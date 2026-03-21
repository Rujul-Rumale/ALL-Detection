        epoch_time = int(time.time() - t0)
        epoch_times.append(epoch_time)

        # ETA: rolling average over last 5 epochs (more stable than global avg
        # because epoch time changes when backbone is unfrozen mid-training).
        rolling   = epoch_times[-min(5, len(epoch_times)):]
        avg_ep    = sum(rolling) / len(rolling)
        remaining = max(0, (args.epochs + args.patience) - epoch)
        eta_str   = format_time(avg_ep * remaining) if remaining > 0 else "done"

        # ── Loss-AUC decoupling detector ───────────────────────────────────
        # WHAT IT DETECTS:
        #   "Ghost training" — the loss is still falling (which looks like
        #   progress) but val_AUC is no longer improving (meaning nothing
        #   useful is being learned). This happens because:
        #     1. Focal loss self-suppression reduces gradients as the model
        #        becomes confident, so the loss falls even with tiny updates.
        #     2. The model has genuinely converged and is only fine-tuning
        #        marginal confidence values without changing predictions.
        #
        # DETECTION LOGIC:
        #   If over the last 10 epochs:
        #     - val_loss moved by > 5e-4 (it is still changing)
        #     - val_AUC changed by < 0.001 (but AUC is frozen)
        #   → the loss signal has decoupled from generalisation.
        #
        # THRESHOLD RATIONALE:
        #   5e-4 loss movement: below this, loss is essentially flat too
        #   (numerical noise), so decoupling doesn't apply.
        #   0.001 AUC: changes smaller than this are within epoch-to-epoch
        #   noise and should not be counted as real improvements.
        _auc_history.append(auc)
        _loss_history.append(val_loss)
        if len(_auc_history) >= 10:
            auc_delta  = max(_auc_history[-10:]) - min(_auc_history[-10:])
            loss_delta = max(_loss_history[-10:]) - min(_loss_history[-10:])
            if loss_delta > 5e-4 and auc_delta < 0.001:
                logger.warning(
                    f"[DECOUPLE E{epoch}] val_loss moving ({loss_delta:.4f}) "
                    f"but val_AUC flat ({auc_delta:.4f}) over last 10 epochs. "
                    f"Loss is no longer a reliable training signal. "
                    f"Consider stopping if patience also indicates plateau."
                )

        # ── Update training state ──────────────────────────────────────────
        # current_state is declared nonlocal above — this assignment correctly
        # updates the outer scope variable. Without nonlocal, this line would
        # create a new local variable and the outer current_state would never
        # be updated (the original bug).
        current_state = {
            "epoch": epoch, "auc": auc,  "acc": acc,
            "sens":  sens,  "spec": spec, "f1":  f1,
            "train_loss": train_loss, "val_loss": val_loss,
        }

        is_best = auc > best_auc
        if is_best:
            best_auc   = auc
            best_state = current_state.copy()
            patience_counter = 0   # reset on any improvement, not just post-min_epochs

            # Save checkpoint: includes optimizer state for exact resume.
            # weights_only=False at load time is required (see resume block).
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "auc":                  auc,
                    "args":                 vars(args),
                },
                checkpoint_path,
            )
        else:
            # Only increment patience counter after min_epochs.
            # Reason: the model may still be in an active learning phase
            # (Phase 1/1.5 → Phase 2 transition) and temporarily plateau
            # before the full backbone kicks in. Premature patience would
            # stop the run before the most productive phase begins.
            if epoch >= args.min_epochs:
                patience_counter += 1

        # ── Compose log line ───────────────────────────────────────────────
        bLR = optimizer.param_groups[0]["lr"]
        hLR = optimizer.param_groups[1]["lr"]
        ms  = monitor_state   # hardware snapshot from monitor thread

        epoch_line = (
            f"E {epoch:>3} | "
            # [50/50] tag: reminds the reader that train_acc is on a synthetic
            # balanced distribution, not the real data distribution.
            f"TrL={train_loss:.4f} TrA={train_acc:.4f}[50/50] | "
            f"VlA={acc:.4f} F1={f1:.4f} "
            f"Se={sens:.4f} Sp={spec:.4f} AUC={auc:.4f} | "
            f"bLR={bLR:.6f} hLR={hLR:.6f} | "
            f"{epoch_time}s ETA={eta_str} | "
            f"pat={patience_counter}/{args.patience} | "
            f"CPU={ms['cpu_pct']:.0f}% "
            f"RAM={ms['ram_used']:.1f}GB "
            f"GPU={ms['gpu_util']:.0f}% "
            f"VRAM={ms['vram_used']/1024:.1f}GB "
            f"T={ms['gpu_temp']:.0f}C"
            + (" * BEST" if is_best else "")
        )

        # Append epoch to CSV metrics file.
        # train_acc_50_50 column name makes the synthetic distribution
        # explicit in any downstream data analysis.
        with open(metrics_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},"
                f"{acc:.4f},{auc:.4f},{sens:.4f},{spec:.4f},"
                f"{f1:.4f},{epoch_time}\n"
            )

        return epoch_line

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TRAINING | {run_id}")
    print(
        f"  min_epochs={args.min_epochs}  "
        f"patience={args.patience}  "
        f"hard_ceiling={args.epochs}"
    )
    print(f"{'='*70}\n")

    epoch = start_epoch
    while True:
        print(f"\n--- Epoch {epoch} ---", flush=True)
        epoch_line = run_epoch(epoch)
        logger.info(epoch_line)

        # STOPPING LOGIC:
        # ─────────────────
        # Early stopping: fires only after min_epochs, preventing premature
        # stops during Phase 2 ramp-up. patience_counter counts consecutive
        # epochs without val_AUC improvement (resetting on any improvement).
        if epoch >= args.min_epochs and patience_counter >= args.patience:
            logger.info(
                f"Early stopping at epoch {epoch}: val_AUC has not improved "
                f"for {patience_counter} consecutive epochs "
                f"(min_epochs={args.min_epochs} satisfied). "
                f"Best AUC: {best_auc:.4f} (saved to {checkpoint_path})."
            )
            break

        # Hard ceiling: prevents runaway training if patience never fires
        # (e.g. AUC oscillates by tiny amounts keeping patience=0 forever).
        if epoch >= args.epochs + args.patience:
            logger.info(
                f"Hard ceiling reached at epoch {epoch} "
                f"({args.epochs} guaranteed + {args.patience} max patience). "
                f"Stopping."
            )
            break

        epoch += 1

    # ── Post-training: threshold optimisation + ONNX export ───────────────────
    stop_event.set()
    monitor_thread.join()
    logger.info(f"System hardware log: {sys_log_path}")

    # Load the best checkpoint (not the final epoch's weights).
    logger.info("Loading best checkpoint for threshold optimisation...")
    ckpt = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Collect TTA-averaged P(ALL) scores over the full validation set.
    # These are used to find the optimal decision threshold via Youden's J.
    all_targets_t, all_scores_t = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            variants = [
                images,
                torch.flip(images, [3]),
                torch.flip(images, [2]),
                torch.flip(images, [2, 3]),
            ]
            avg_probs = torch.zeros(images.size(0), 2, device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                for v in variants:
                    avg_probs += torch.softmax(model(v).float(), dim=1)
            avg_probs /= 4
            all_targets_t.extend(labels.cpu().numpy())
            all_scores_t.extend(avg_probs[:, 0].cpu().numpy())

    all_targets_t = np.array(all_targets_t)
    all_scores_t  = np.array(all_scores_t)

    # Youden's J = Sensitivity + Specificity - 1.
    # The threshold that maximises J is the optimal operating point for
    # clinical use: it balances sensitivity (catching real ALL) against
    # specificity (not over-calling HEM as ALL).
    logger.info("ALL-positive ROC threshold optimisation (Youden's J):")
    logger.info(f"{'Threshold':>10} {'Sens':>8} {'Spec':>8} {'J':>8}")

    optimal_metrics = sweep_all_positive_thresholds(all_targets_t, all_scores_t)

    for thresh in np.arange(0.35, 0.76, 0.05):
        tm  = compute_all_positive_metrics(
            all_targets_t, all_scores_t, threshold=thresh
        )
        j = tm["sens"] + tm["spec"] - 1.0
        logger.info(
            f"{thresh:>10.2f} {tm['sens']:>8.4f} "
            f"{tm['spec']:>8.4f} {j:>8.4f}"
        )

    optimal_threshold = optimal_metrics["threshold"]
    logger.info(
        f"Optimal ALL threshold: {optimal_threshold:.2f} "
        f"(J={optimal_metrics['j']:.4f})"
    )

    # Save optimal threshold into the params JSON so inference pipelines can
    # load a single file to get both the hyperparameters and decision threshold.
    with open(params_path, "r") as f:
        params = json.load(f)
    params["optimal_threshold"] = float(optimal_threshold)
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

    # ONNX export for deployment / inference without PyTorch runtime.
    # opset_version=13: stable, widely supported by ONNX Runtime >= 1.10.
    # dynamic_axes: allows variable batch sizes at inference time without
    # re-exporting.
    logger.info("Exporting best model to ONNX...")
    onnx_path = os.path.join(output_dir, f"{run_id}_best.onnx")
    dummy = torch.randn(1, 3, args.res, args.res).to(device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=13,
    )
    logger.info(f"ONNX saved: {onnx_path}")

    logger.info(
        f"\n[COMPLETE] "
        f"Best AUC: {best_auc:.4f} | "
        f"Threshold: {optimal_threshold:.2f} | "
        f"Outputs: {output_dir}"
    )


if __name__ == "__main__":
    main()
