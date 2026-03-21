    WHY AN ENUM INSTEAD OF INTEGERS OR FLOATS:
    -------------------------------------------
    The original code used phase=1.5 as a float literal to represent the
    "last 4 blocks" state. Float comparisons are fragile (IEEE 754 rounding
    can make 1.5 != 1.5 in edge cases involving arithmetic), and a float
    has no intrinsic meaning — an agentic tool or reviewer cannot know what
    "1.5" means without reading the entire function.

    An Enum is:
      - Type-safe: cannot accidentally pass 1.6 or "1.5"
      - Self-documenting: UnfreezePhase.PARTIAL4 is unambiguous
      - IDE-navigable: jump-to-definition works; float literals don't
    """
    FROZEN   = 0   # all backbone parameters frozen; head only
    PARTIAL2 = 1   # last 2 semantic blocks unfrozen
    PARTIAL4 = 2   # last 4 semantic blocks unfrozen
    FULL     = 3   # entire backbone unfrozen


# Named parameter prefixes for each backbone.
#
# WHY NOT backbone.children() (ORIGINAL BUG):
# --------------------------------------------
# backbone.children() returns the immediate child modules of the timm model
# object. For EfficientNet-B0, these are:
#   conv_stem, bn1, blocks, conv_head, bn2, global_pool
#
# Note: "blocks" is a SINGLE child module containing ALL MBConv stages.
# So list(backbone.children())[-2:] gives [bn2, global_pool] — two utility
# layers — NOT the last two MBConv stages. The intended partial unfreeze was
# never executed; the model was effectively frozen except for BN + pooling.
#
# named_parameters() walks the full parameter tree with dot-separated names
# like "blocks.6.0.conv_dw.weight". Matching against prefix "blocks.6"
# correctly selects exactly MBConv stage 6 across all timm variants.
#
# HOW TO VERIFY/EXTEND THIS TABLE:
#   import timm
#   m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
#   print([n for n, _ in m.named_parameters()][:20])
# This prints the actual parameter names for any timm model. Add entries
# for new models by identifying the last N stage prefixes from this output.
UNFREEZE_PREFIXES = {
    "mobilenetv3_large_100": {
        UnfreezePhase.PARTIAL2: ["blocks.6", "blocks.5"],
        UnfreezePhase.PARTIAL4: ["blocks.6", "blocks.5", "blocks.4", "blocks.3"],
    },
    "efficientnet_b0": {
        UnfreezePhase.PARTIAL2: ["blocks.6", "blocks.5"],
        UnfreezePhase.PARTIAL4: ["blocks.6", "blocks.5", "blocks.4", "blocks.3"],
    },
    "efficientnet_b4": {
        UnfreezePhase.PARTIAL2: ["blocks.6", "blocks.5"],
        UnfreezePhase.PARTIAL4: ["blocks.6", "blocks.5", "blocks.4", "blocks.3"],
    },
    "resnet50": {
        # ResNet uses layer1–layer4 naming instead of blocks.N.
        UnfreezePhase.PARTIAL2: ["layer4", "layer3"],
        UnfreezePhase.PARTIAL4: ["layer4", "layer3", "layer2", "layer1"],
    },
}


def set_backbone_unfreeze_phase(
    model, phase: UnfreezePhase, timm_name: str, logger
):
    """
    Sets requires_grad on backbone parameters according to `phase`.

    Always starts by freezing everything, then selectively unfreezes.
    This ensures idempotency: calling this function twice with the same
    phase produces the same result regardless of prior state.
    """
    backbone = model[0]

    # Step 1: freeze everything. Unconditional — ensures clean state.
    for param in backbone.parameters():
        param.requires_grad = False

    if phase == UnfreezePhase.FROZEN:
        logger.info("Backbone: FROZEN (head-only training)")
        return

    if phase == UnfreezePhase.FULL:
        # Unfreeze all backbone parameters at once.
        for param in backbone.parameters():
            param.requires_grad = True
        n = sum(p.numel() for p in backbone.parameters())
        logger.info(f"Backbone: FULL — {n:,} parameters now trainable")
        return

    # PARTIAL2 or PARTIAL4: look up the prefix list for this model.
    prefixes = UNFREEZE_PREFIXES.get(timm_name, {}).get(phase, [])
    if not prefixes:
        # Safety fallback: unknown model or missing entry in the table.
        # Warn loudly so the operator can add the correct prefixes.
        logger.warning(
            f"No prefix table for {timm_name}/{phase.name}. "
            f"Falling back to FULL unfreeze — ADD THIS MODEL TO "
            f"UNFREEZE_PREFIXES to enable proper partial unfreezing."
        )
        for param in backbone.parameters():
            param.requires_grad = True
        return

    # Unfreeze only parameters whose names start with any listed prefix.
    for name, param in backbone.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad = True

    n = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    logger.info(
        f"Backbone: {phase.name} — {n:,} params trainable "
        f"(prefixes: {prefixes})"
    )


def apply_phase_lr(optimizer, backbone_lr, scheduler, logger):
    """
    Sets backbone group LR and resets the cosine scheduler's internal state
    so cosine decay starts from the new LR rather than the original one.

    WHY THIS IS NEEDED (ORIGINAL BUG):
    ------------------------------------
    At each phase transition the original code did:

        optimizer.param_groups[0]["lr"] = args.lr_backbone * 0.5

    But at the END of the same epoch, run_epoch() calls scheduler.step().
    SequentialLR (once past its milestone) delegates to CosineAnnealingLR,
    which recomputes the LR from its internal base_lrs and last_epoch counter
    — OVERWRITING the manual assignment. The phase LR change had zero effect.

    Fix: after setting the LR on the optimizer group, also update:
      - active_sched.base_lrs[0]: the value cosine decays FROM
      - active_sched.last_epoch = 0: restart the cosine period from this LR

    This means cosine will decay from backbone_lr to eta_min starting at the
    phase transition epoch, which is the intended behaviour.
    """
    optimizer.param_groups[0]["lr"] = backbone_lr

    # scheduler.schedulers[-1] is CosineAnnealingLR once the warmup
    # milestone is passed. Index -1 is safe because SequentialLR always
    # has at least one scheduler after the milestone.
    active_sched = scheduler.schedulers[-1]
    if hasattr(active_sched, "base_lrs"):
        active_sched.base_lrs[0] = backbone_lr
        active_sched.last_epoch  = 0

    logger.info(
        f"Backbone LR set to {backbone_lr:.2e}; "
        f"cosine base_lrs and last_epoch reset."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════
def get_loaders(args):
    """
    Builds train and val DataLoaders from the cross-validation split JSON.

    CLASS IMBALANCE HANDLING:
    --------------------------
    The CNMC dataset has approximately a 2:1 ALL:HEM ratio. Two mechanisms
    are applied together (they are complementary, not redundant):

    1. WeightedRandomSampler: resamples the training set to produce
       approximately 50/50 class batches. This ensures the model sees each
       class equally during training, preventing it from learning to predict
       the majority class always.

    2. MildFocalLoss(weight=...): applies per-class loss weights derived
       from class_counts. Even with a balanced sampler, within each batch
       the model may be more or less confident about each class; the loss
       weight provides additional gradient pressure toward harder classes.

    IMPORTANT: because WeightedRandomSampler produces a 50/50 distribution,
    train_acc reflects performance on an ARTIFICIAL balanced world that does
    NOT exist at inference time. The [50/50] tag on the log line marks this.
    Val_acc reflects the real ~2:1 distribution. The two numbers are NOT
    comparable and the train/val accuracy gap is NOT a reliable measure of
    overfitting — use val_AUC and val_F1 for all convergence decisions.
    """
    with open(args.splits_json, "r") as f:
        splits = json.load(f)

    fold_key = f"fold_{args.fold}"
    if fold_key not in splits["folds"]:
        raise ValueError(
            f"Fold '{fold_key}' not found in {args.splits_json}. "
            f"Available keys: {list(splits['folds'].keys())}"
        )

    fold_data   = splits["folds"][fold_key]
    train_pairs = fold_data["train_images"]   # list of [path, label]
    val_pairs   = fold_data["val_images"]

    train_transform, val_transform = get_transforms(args.res)
    train_ds = CNMCDataset(train_pairs, transform=train_transform)
    val_ds   = CNMCDataset(val_pairs,   transform=val_transform)

    # Build sample weights: each sample's weight = 1 / count_of_its_class.
    # This makes ALL samples (minority or majority) equally likely to be drawn
    # per epoch regardless of class frequency.
    labels       = [p[1] for p in train_pairs]
    class_counts = np.bincount(labels)
    sample_weights = np.array([1.0 / class_counts[l] for l in labels])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        # replacement=True: allows oversampling the minority class beyond
        # its natural count, which is necessary for true class balance.
    )

    # prefetch_factor=2: each worker pre-fetches 2 batches ahead.
    # Reason: GPU processing time ≈ 1–2 batches at res=320, so 2 pre-fetched
    # batches keep the GPU fed without excessive memory buffering.
    prefetch   = 2 if args.num_workers > 0 else None
    persistent = args.num_workers > 0
    # persistent_workers=True: keeps worker processes alive between epochs.
    # Reason: spawning 4 workers takes ~2–3 seconds on some systems. Over
    # 150 epochs this adds 5–7 minutes of pure overhead. Persistent workers
    # avoid this at the cost of slightly higher idle RAM.

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,          # sampler is mutually exclusive with shuffle
        num_workers=args.num_workers,
        pin_memory=True,          # pin_memory: enables faster host→GPU copies
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        drop_last=True,
        # drop_last=True: discards the last incomplete batch each epoch.
        # Reason: BatchNorm1d in the head raises an error on batch_size=1.
        # Even if the last batch is size > 1, its gradient contribution is
        # disproportionate (smaller batch → noisier gradient estimate).
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,            # no shuffle: val order doesn't matter for metrics
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        # drop_last=False for val: we want metrics over ALL val samples.
    )

    # normed_weights used by MildFocalLoss as the `weight` argument to CE.
    # Formula: n_classes / (class_count * n_classes) = 1 / class_count,
    # normalised so weights sum to n_classes (PyTorch convention).
    normed_weights = len(class_counts) / (class_counts * 2.0)

    # Class breakdown for the startup log — confirms the split loaded correctly.
    train_all = sum(1 for _, l in train_pairs if l == 0)
    train_hem = sum(1 for _, l in train_pairs if l == 1)
    val_all   = sum(1 for _, l in val_pairs   if l == 0)
    val_hem   = sum(1 for _, l in val_pairs   if l == 1)

    info = {
        "train_total": len(train_pairs), "val_total": len(val_pairs),
        "train_all":   train_all,        "train_hem": train_hem,
        "val_all":     val_all,          "val_hem":   val_hem,
    }

    return (
        train_loader, val_loader,
        torch.tensor(normed_weights, dtype=torch.float32),
        class_counts, info,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  System Monitor Thread
# ═══════════════════════════════════════════════════════════════════════════════
def system_monitor(log_path, stop_event, monitor_state, interval=5):
    """
    Background thread that samples CPU/RAM/GPU/VRAM/temp every `interval`
    seconds and writes to a CSV for post-hoc hardware analysis.

    Runs as a daemon thread so it dies automatically if the main process
    crashes without calling stop_event.set().

    monitor_state is a plain dict updated in-place and read by run_epoch()
    to append hardware info to each epoch log line. No locking is used:
    dict updates in CPython are atomic for simple key/value pairs (GIL
    prevents torn reads/writes on dict.__setitem__).
    """
    with open(log_path, "w") as f:
        f.write(
            "timestamp,cpu_pct,ram_used_gb,ram_total_gb,"
            "gpu_util_pct,vram_used_mb,vram_total_mb,gpu_temp_c\n"
        )

    while not stop_event.is_set():
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
            ram     = psutil.virtual_memory()
            ram_used  = ram.used  / 1024 ** 3
            ram_total = ram.total / 1024 ** 3

            # nvidia-smi subprocess: more reliable than pynvml for reading
            # VRAM and temperature across driver versions.
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,utilization.gpu,memory.used,"
                    "memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True,
            )
            parts      = [p.strip() for p in result.stdout.strip().split(",")]
            timestamp  = parts[0]
            gpu_util   = parts[1]
            vram_used  = parts[2]
            vram_total = parts[3]
            gpu_temp   = parts[4]

            with open(log_path, "a") as f:
                f.write(
                    f"{timestamp},{cpu_pct},{ram_used:.2f},{ram_total:.2f},"
                    f"{gpu_util},{vram_used},{vram_total},{gpu_temp}\n"
                )

            # update() replaces all keys atomically under CPython GIL.
            monitor_state.update({
                "cpu_pct":    float(cpu_pct),
                "ram_used":   ram_used,
                "ram_total":  ram_total,
                "gpu_util":   float(gpu_util),
                "vram_used":  float(vram_used),
                "vram_total": float(vram_total),
                "gpu_temp":   float(gpu_temp),
            })
        except Exception:
            # Silently skip failed samples (e.g. nvidia-smi not found on CPU
            # machines). The CSV will have gaps but training continues.
            pass

        stop_event.wait(interval)


def format_time(seconds):
    """Human-readable duration string for ETA and elapsed time logging."""
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # ── Run identity ──────────────────────────────────────────────────────────
    # Timestamp in run_id prevents collisions when the same run_name is used
    # for multiple experiments on the same day.
    run_id = (
        f"{args.run_name}_fold{args.fold}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = os.path.join(args.output_root, args.run_name)
    logger = setup_logging(output_dir, run_id)

    # ── Device & performance flags ────────────────────────────────────────────
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    )

    # CPU thread caps: 8 intra-op, 4 inter-op.
    # Reason: DataLoader workers + PyTorch's own thread pools can over-subscribe
    # CPU cores, causing context-switch overhead that slows both CPU and GPU
    # pipelines. Explicit caps keep total threads predictable.
    torch.set_num_threads(8)
    torch.set_num_interop_threads(4)

    # TF32 on Ampere+ GPUs: uses tensor cores for matmul/conv at ~3× the
    # throughput of FP32, with negligible accuracy difference for training.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True

    # cudnn.benchmark: profiles convolution algorithms on the first few batches
    # and selects the fastest for the fixed input shape (res×res).
    # Only beneficial when input shapes are constant — which they are here.
    torch.backends.cudnn.benchmark = True

    use_amp = device.type == "cuda"
    # AMP only on CUDA: MPS (Apple Silicon) and CPU do not support
    # torch.amp.GradScaler and will error if use_amp=True.

    # ── Data ──────────────────────────────────────────────────────────────────
    (train_loader, val_loader,
     loss_weights, class_counts, data_info) = get_loaders(args)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, backbone_params, head_params, timm_name, in_features = get_model(args)
    model = model.to(device)
    n_backbone = sum(p.numel() for p in backbone_params)
    n_head     = sum(p.numel() for p in head_params)

    # ── Startup log ───────────────────────────────────────────────────────────
    # Written once at the top of every run. Agentic tools that parse log files
    # can use this block to reconstruct the exact hyperparameter configuration
    # without needing to re-parse CLI args.
    other_folds = [f for f in [1, 2, 3] if f != args.fold]
    folds_train = "+".join(str(f) for f in other_folds)
    # Effective cosine T_max: the warmup period is handled by LinearLR, so
    # CosineAnnealingLR should span only the remaining (non-warmup) epochs.
    cosine_t_max_effective = max(1, args.cosine_t_max - args.warmup_epochs)

    for line in [
        "=" * 70,
        "  ALL Leukemia Edge Classifier — train.py",
        f"  Run ID:            {run_id}",
        f"  Model:             {args.model}  ({timm_name})",
        f"  Fold:              {args.fold} (val) | Folds {folds_train} (train)",
        f"  Resolution:        {args.res}x{args.res}",
        f"  Batch size:        {args.batch_size}",
        f"  Min epochs:        {args.min_epochs}  (patience cannot fire before this)",
        f"  Hard ceiling:      {args.epochs}  (never exceeded regardless of patience)",
        f"  Patience:          {args.patience}",
        f"  LR backbone:       {args.lr_backbone}",
        f"  LR head:           {args.lr_head:.0e}",
        f"  Weight decay:      {args.weight_decay:.0e}",
        f"  Label smoothing:   {args.label_smoothing}",
        f"  Warmup epochs:     {args.warmup_epochs}",
        f"  Phase 1 start:     {args.phase1_start}  (last 2 blocks)",
        f"  Phase 1.5 start:   {args.phase1_5_start}  (last 4 blocks)",
        f"  Phase 2 start:     {args.phase2_start}  (full backbone)",
        f"  Cosine T_max raw:  {args.cosine_t_max}",
        f"  Cosine T_max eff:  {cosine_t_max_effective}  (= raw - warmup)",
        f"  Cosine eta_min:    1e-7",
        f"  AMP:               {use_amp}",
        f"  Device:            {device} ({gpu_name})",
        f"  Workers:           {args.num_workers}",
        f"  Splits JSON:       {args.splits_json}",
        f"  Decouple detector: fires WARNING when val_loss Δ>5e-4 and",
        f"                     val_AUC Δ<0.001 over the last 10 epochs.",
        f"                     This indicates ghost training (loss ≠ AUC).",
        "=" * 70,
        "  Dataset",
        (f"  Train: {data_info['train_total']}  "
         f"(ALL={data_info['train_all']}, HEM={data_info['train_hem']})"),
        (f"  Val:   {data_info['val_total']}  "
         f"(ALL={data_info['val_all']},   HEM={data_info['val_hem']})"),
        (f"  Loss weights:  ALL={loss_weights[0]:.6f}, "
         f"HEM={loss_weights[1]:.6f}  (from class_counts)"),
        "=" * 70,
        "  Model",
        f"  Backbone params: {n_backbone:>10,}",
        f"  Head params:     {n_head:>10,}",
        f"  Total params:    {n_backbone + n_head:>10,}",
        f"  Head:  BN1d -> Drop(0.4) -> Linear(512) -> ReLU -> Drop(0.3) -> Linear(2)",
        "=" * 70,
        "  Loss: MildFocalLoss(gamma=0.5, label_smoothing=0.05)",
        "  Alpha: [0.75, 0.25]  (ALL errors penalised 3x more than HEM)",
        "  NOTE: train_acc is measured on a SYNTHETIC 50/50 distribution.",
        "        WeightedRandomSampler resamples to class balance for training.",
        "        DO NOT compare train_acc to val_acc — different distributions.",
        "        Use val_AUC and val_F1 for all convergence and stopping decisions.",
        "=" * 70,
    ]:
        logger.info(line)
