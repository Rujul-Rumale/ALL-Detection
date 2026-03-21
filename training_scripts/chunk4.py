    # Save all args to JSON so any downstream tool can reconstruct the run.
    params_path = os.path.join(output_dir, f"{run_id}_params.json")
    with open(params_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    # ── Criterion ─────────────────────────────────────────────────────────────
    # Combine alpha and normed class weights multiplicatively.
    # alpha=[0.75, 0.25] reflects clinical priority (ALL errors are costlier).
    # loss_weights reflects class frequency (normalised 1/class_count).
    # Together they apply both demographic balance and clinical priority.
    alpha = torch.tensor([0.75, 0.25], dtype=torch.float32)
    combined_weight = alpha * loss_weights.to(device)

    criterion = MildFocalLoss(
        weight=combined_weight,
        gamma=0.5,
        label_smoothing=args.label_smoothing,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Phase 0: backbone fully frozen; only head trains.
    set_backbone_unfreeze_phase(model, UnfreezePhase.FROZEN, timm_name, logger)

    # Two parameter groups with different LRs:
    #   group 0 = backbone (frozen initially; LR will be very small when unfrozen)
    #   group 1 = head (always active; higher LR for faster convergence)
    # filter(requires_grad) on group 0 selects zero params initially (frozen),
    # but the group exists so we can update its params list at phase transitions
    # without rebuilding the optimizer (which would reset momentum buffers).
    optimizer = optim.AdamW(
        [
            {
                "params": filter(
                    lambda p: p.requires_grad, model[0].parameters()
                ),
                "lr": args.lr_backbone,
            },
            {
                "params": model[1].parameters(),
                "lr": args.lr_head,
            },
        ],
        weight_decay=args.weight_decay,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    # Phase 1 (warmup): linear ramp from 0.5× to 1× LR over warmup_epochs.
    # Reason: the head is randomly initialised; large early gradients can
    # corrupt the first backbone updates when unfreezing begins. Warmup
    # keeps the head LR low until it has begun to converge.
    warmup_sched = LinearLR(
        optimizer,
        start_factor=0.5,
        total_iters=args.warmup_epochs,
    )

    # Phase 2 (cosine): decays from base LR to eta_min=1e-7 over
    # cosine_t_max_effective steps, which starts immediately after warmup.
    # ORIGINAL BUG: cosine T_max was set to the raw cosine_t_max (e.g. 150).
    # The warmup period consumed the first warmup_epochs steps, so the cosine
    # scheduler only had (150 - 10) = 140 steps to decay — but its internal
    # period was 150, meaning it never reached eta_min within the training run.
    # FIX: set T_max = cosine_t_max - warmup_epochs so cosine reaches eta_min
    # at exactly the hard ceiling epoch.
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=cosine_t_max_effective,
        eta_min=1e-7,
    )

    # SequentialLR: runs warmup_sched for warmup_epochs steps, then hands off
    # to cosine_sched for all remaining steps.
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[args.warmup_epochs],
    )

    # GradScaler for AMP. Disabled on CPU (enabled=False = identity operations).
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── System monitor ────────────────────────────────────────────────────────
    monitor_state = {
        "cpu_pct": 0.0, "ram_used": 0.0, "ram_total": 0.0,
        "gpu_util": 0.0, "vram_used": 0.0, "vram_total": 0.0, "gpu_temp": 0.0,
    }
    sys_log_path = os.path.join(output_dir, f"{run_id}_system.log")
    stop_event   = threading.Event()
    monitor_thread = threading.Thread(
        target=system_monitor,
        args=(sys_log_path, stop_event, monitor_state),
        daemon=True,
        # daemon=True: thread is killed automatically if main process dies
        # (e.g. OOM, keyboard interrupt). Non-daemon threads block process exit.
    )
    monitor_thread.start()

    # ── Training state ────────────────────────────────────────────────────────
    best_auc        = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(output_dir, f"{run_id}_best.pth")
    metrics_path    = os.path.join(output_dir, f"{run_id}_metrics.csv")

    with open(metrics_path, "w") as f:
        # Column name "train_acc_50_50" makes the synthetic distribution
        # explicit in any downstream CSV analysis.
        f.write(
            "epoch,train_loss,train_acc_50_50,val_loss,val_acc,"
            "auc,sensitivity,specificity,f1,epoch_time\n"
        )

    best_state    = {}
    current_state = {}   # Must be defined before run_epoch (nonlocal target)
    epoch_times   = []

    # Rolling windows for the loss-AUC decoupling detector.
    # Kept as module-scope lists (closured in run_epoch) rather than globals
    # for cleaner scoping — they are only meaningful within this training run.
    _auc_history  = []
    _loss_history = []

    global_start = time.time()

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")

        # weights_only=False REQUIRED for own checkpoints on PyTorch >= 2.6.
        # REASON: PyTorch 2.6 changed the default of weights_only from False
        # to True. The checkpoint saved by this script includes
        # optimizer_state_dict, which contains internal PyTorch objects (e.g.
        # torch.Size, torch.dtype). These are not in the default safe list for
        # weights_only=True, causing an UnpicklingError at load time.
        # This file is written by this script and is a trusted source —
        # weights_only=False is safe here. Do NOT use weights_only=False on
        # third-party or user-provided checkpoints.
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        if "epoch" in checkpoint:
            resumed_epoch = checkpoint["epoch"]
            start_epoch   = resumed_epoch + 1
            logger.info(
                f"Checkpoint was at epoch {resumed_epoch}; "
                f"resuming from epoch {start_epoch}."
            )

            # Align backbone freeze state to match the epoch we are resuming
            # from. Without this, the backbone would start fully frozen on
            # resume even if Phase 2 was already active when the run stopped.
            if resumed_epoch >= args.phase2_start:
                set_backbone_unfreeze_phase(
                    model, UnfreezePhase.FULL, timm_name, logger
                )
                optimizer.param_groups[0]["params"] = list(
                    model[0].parameters()
                )
                logger.info("Resume: aligned to Phase FULL")

            elif resumed_epoch >= args.phase1_5_start:
                set_backbone_unfreeze_phase(
                    model, UnfreezePhase.PARTIAL4, timm_name, logger
                )
                optimizer.param_groups[0]["params"] = list(
                    p for p in model[0].parameters() if p.requires_grad
                )
                logger.info("Resume: aligned to Phase PARTIAL4")

            elif resumed_epoch >= args.phase1_start:
                set_backbone_unfreeze_phase(
                    model, UnfreezePhase.PARTIAL2, timm_name, logger
                )
                optimizer.param_groups[0]["params"] = list(
                    p for p in model[0].parameters() if p.requires_grad
                )
                logger.info("Resume: aligned to Phase PARTIAL2")

            else:
                logger.info("Resume: staying in Phase FROZEN (head only)")

        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Optimizer state restored from checkpoint.")
            except Exception as e:
                # This can happen if the optimizer param groups changed
                # (e.g. different backbone freeze state). Log and continue;
                # the optimizer will start fresh with the current LR schedule.
                logger.warning(
                    f"Could not restore optimizer state: {e}. "
                    f"Optimizer will start fresh — LR schedule resets."
                )

        if "auc" in checkpoint:
            best_auc = checkpoint["auc"]
            logger.info(f"Resume: best AUC = {best_auc:.4f}")

    # ── Epoch function ────────────────────────────────────────────────────────
    def run_epoch(epoch):
        """
        Runs one complete train + val epoch and updates all training state.

        Declared as a nested function (closure) to avoid passing a large
        number of objects as arguments. All mutable state (best_auc,
        patience_counter, best_state, current_state) is explicitly declared
        nonlocal to avoid the Python scoping bug where assignment inside a
        function creates a LOCAL variable instead of updating the outer one.

        ORIGINAL BUG (current_state):
            nonlocal best_auc, patience_counter, best_state
            # current_state was missing from this list.
            # The line `current_state = {...}` created a new local variable.
            # The outer current_state (read by any external logger/display)
            # was never updated — it always contained the initial empty dict.
        """
        nonlocal best_auc, patience_counter, best_state, current_state

        # ── Phase transitions ──────────────────────────────────────────────
        # Each transition:
        #   1. Updates which backbone parameters have requires_grad=True
        #   2. Rebuilds the backbone optimizer param group to include only
        #      the newly-unfrozen params (previously frozen params had
        #      requires_grad=False and were excluded)
        #   3. Sets the backbone LR and resets the cosine scheduler base
        #      (see apply_phase_lr docstring for why this is necessary)
        if epoch == args.phase1_start:
            set_backbone_unfreeze_phase(
                model, UnfreezePhase.PARTIAL2, timm_name, logger
            )
            optimizer.param_groups[0]["params"] = list(
                p for p in model[0].parameters() if p.requires_grad
            )
            # 0.5× backbone LR: last 2 blocks are still fragile at this stage;
            # a gentler LR prevents large weight updates that could undo
            # ImageNet feature representations accumulated over epochs 1–10.
            apply_phase_lr(
                optimizer, args.lr_backbone * 0.5, scheduler, logger
            )

        elif epoch == args.phase1_5_start:
            set_backbone_unfreeze_phase(
                model, UnfreezePhase.PARTIAL4, timm_name, logger
            )
            optimizer.param_groups[0]["params"] = list(
                p for p in model[0].parameters() if p.requires_grad
            )
            # 0.75× LR: blocks 3 and 4 are deeper and more general-purpose;
            # slightly higher LR than Phase 1 to allow faster adaptation
            # without destabilising the already-fine-tuned blocks 5 and 6.
            apply_phase_lr(
                optimizer, args.lr_backbone * 0.75, scheduler, logger
            )

        elif epoch == args.phase2_start:
            set_backbone_unfreeze_phase(
                model, UnfreezePhase.FULL, timm_name, logger
            )
            optimizer.param_groups[0]["params"] = list(
                model[0].parameters()
            )
            # Full backbone LR: by Phase 2 the head is stable and can absorb
            # full backbone gradients without losing learned representations.
            apply_phase_lr(optimizer, args.lr_backbone, scheduler, logger)

        # ── Train + val ────────────────────────────────────────────────────
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, acc, auc, sens, spec, f1 = validate(
            model, val_loader, criterion, device, use_amp
        )

        # Step scheduler AFTER validation so the LR shown in the log line
        # is the LR that will be used in the NEXT epoch (not the current one).
        scheduler.step()

        # Periodic memory release: prevents VRAM fragmentation accumulating
        # over 150 epochs, which can cause OOM errors in later epochs even
        # when peak per-epoch usage is within limits.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
