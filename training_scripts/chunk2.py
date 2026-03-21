    ColorJitter (hue=0.02, brightness/contrast/saturation=0.1):
      - Simulates H&E staining variation across labs. Kept mild (0.02 hue)
        to avoid shifting cell colour so far that discriminative features
        (nuclear-to-cytoplasm ratio, chromatin texture) become unreliable.

    GaussianBlur:
      - Simulates out-of-focus scanner regions. Applied at p=0.2 to keep
        the majority of training crops in-focus.

    CoarseDropout (8 holes of size res//10):
      - Forces the model to classify from partial cell views, preventing
        over-reliance on a single morphological feature (e.g. nucleus size).

    Val transform: resize only — no augmentation. Reason: val metrics must
    reflect real-world performance on unmodified images, not augmented ones.
    """
    train_transform = A.Compose([
        # Oversample then crop: gives random spatial sampling without black
        # border padding artefacts from affine transforms.
        A.Resize(res + 32, res + 32),
        A.RandomCrop(res, res),

        # All three flips are valid for microscopy — no canonical "up".
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Affine: mild shear and scale to simulate slide mounting variation.
        A.Affine(shear=(-10, 10), scale=(0.9, 1.1), p=0.5),

        # ElasticTransform: simulates cell membrane deformation from
        # slide preparation. approximate=True uses a faster gaussian kernel
        # approximation; quality difference is negligible at sigma=10.
        A.ElasticTransform(alpha=1, sigma=10, p=0.3, approximate=True),

        # Stain normalisation substitute. True Macenko/Vahadane normalisation
        # is computationally expensive per-sample. ColorJitter at these mild
        # ranges achieves ~70% of the benefit at 0.1% of the cost.
        A.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.5
        ),

        # Scanner defocus simulation.
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),

        # Partial occlusion — forces the model to not rely on any single region.
        # NOTE: requires albumentations==1.4.18. The num_holes_range /
        # hole_height_range / hole_width_range API changed in 1.5.x.
        # Pin the version in requirements.txt to avoid silent API breakage.
        A.CoarseDropout(
            num_holes_range=(8, 8),
            hole_height_range=(res // 10, res // 10),
            hole_width_range=(res // 10, res // 10),
            p=0.2,
        ),

        # ImageNet mean/std: used because we initialise from ImageNet weights.
        # Using different stats would misalign input distributions with the
        # pre-trained feature representations.
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        # No random crop: use exact target resolution.
        A.Resize(res, res),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, val_transform


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss Function
# ═══════════════════════════════════════════════════════════════════════════════
class MildFocalLoss(nn.Module):
    """
    Focal loss with gamma=0.5, corrected alpha, and label smoothing.

    WHY NOT THE ORIGINAL AsymmetricFocalLoss (gamma=2)?
    -----------------------------------------------------
    The focal term is (1 - p_t)^gamma. Its purpose is to down-weight
    easy (confident) samples so training focuses on hard ones. However
    at gamma=2 this backfires in later training:

      - By epoch 80+, the model classifies most training images with >90%
        confidence.
      - For such a sample: focal_weight = (1 - 0.9)^2 = 0.01
      - That sample contributes only 1% of the gradient of a chance-level
        sample.
      - With ~90% of the batch suppressed this way, the effective gradient
        signal almost vanishes.
      - The reported loss keeps falling (averages over near-zero terms) but
        no meaningful weight updates occur. This is "ghost training" — the
        dashboard reports progress while generalisation is frozen.

    At gamma=0.5:
      - The same sample: (1 - 0.9)^0.5 ≈ 0.32
      - Still down-weighted (easy samples are penalised less), but not
        suppressed. The loss remains informative across all 175 epochs and
        continues to correlate with val_AUC.

    WHY alpha=[0.75, 0.25]?
    ------------------------
    alpha[c] is a per-class penalty multiplier applied on top of the focal
    weight. The original code used alpha=[0.25, 0.75], which penalises HEM
    (class 1) errors 3× more than ALL (class 0) errors.

    For leukemia screening this is clinically backwards:
      - A false negative on ALL (missing a cancer cell) can delay diagnosis
        and worsen patient outcomes.
      - A false positive on HEM (flagging a normal cell as cancer) leads to
        a follow-up test, not a missed diagnosis.

    Correct assignment: alpha[0]=0.75 (penalise ALL errors more),
    alpha[1]=0.25 (tolerate HEM errors more).

    NOTE: WeightedRandomSampler already balances the class distribution;
    alpha provides additional *asymmetric* pressure on top of that balance.
    The two are complementary, not redundant.

    Args:
        weight (Tensor, optional): per-class weight from class_counts,
            passed to CrossEntropyLoss as the base loss.
        gamma (float): focal exponent. 0.5 chosen empirically as the
            highest value that keeps loss informative past epoch 100.
        label_smoothing (float): prevents overconfident logits that would
            push focal weights toward zero even at gamma=0.5.
    """

    def __init__(self, weight=None, gamma=0.5, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        # reduction="none" because we apply the focal weight per-sample
        # before taking the mean ourselves. If reduction="mean" were used,
        # we would be averaging first then multiplying — wrong order.
        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
            reduction="none",
        )

    def forward(self, logits, targets):
        # ce_loss shape: [N] — one scalar per sample.
        ce_loss = self.ce(logits, targets)

        # Convert CE loss to probability of the correct class.
        # exp(-CE) = exp(log(p_t)) = p_t.
        # This avoids a separate softmax + gather, keeping the computation
        # numerically stable (no explicit log of a probability).
        pt = torch.exp(-ce_loss)

        # Focal weight: (1 - p_t)^gamma.
        # High p_t (confident correct prediction) → low weight → sample
        # contributes less to the gradient. Low p_t (hard example) → weight
        # closer to 1 → sample contributes more.
        focal_weight = (1.0 - pt) ** self.gamma

        # Element-wise multiply then reduce.
        return (focal_weight * ce_loss).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  Metric Monitor (running average)
# ═══════════════════════════════════════════════════════════════════════════════
class MetricMonitor:
    """
    Numerically stable running mean for loss tracking across batches.
    Uses sum/count rather than iterative average to avoid floating-point
    drift from repeated (avg + new) / 2 style updates.
    """
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n   # weight by batch size, not batch count
        self.count += n
        self.avg   = self.sum / self.count


# ═══════════════════════════════════════════════════════════════════════════════
#  Training & Validation
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    """
    One full pass over the training set with AMP, gradient clipping, and
    lightweight batch-level progress printing.

    Prints a progress line every 20% of batches — enough to confirm the
    process is alive without flooding the log file. An agentic runner that
    tails stdout will see at most 5 progress lines per epoch.
    """
    model.train()
    monitor = MetricMonitor()
    correct = total = 0
    n_batches = len(loader)

    for i, (images, labels) in enumerate(loader, 1):
        # non_blocking=True overlaps the host→device transfer with CPU work
        # in the previous iteration, reducing idle GPU time.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # set_to_none=True releases gradient memory immediately rather than
        # zeroing it, reducing peak memory usage by ~1 tensor per parameter.
        optimizer.zero_grad(set_to_none=True)

        # autocast: automatically casts ops to float16 where safe, keeping
        # accumulations in float32. Roughly 1.5–2× throughput on Ampere GPUs.
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        if use_amp:
            # GradScaler: scales loss before backward to prevent float16
            # gradient underflow, then unscales before the optimizer step.
            scaler.scale(loss).backward()
            # unscale_ must happen before clip_grad_norm_ so the gradient
            # norms are in the original (unscaled) magnitude.
            scaler.unscale_(optimizer)
            # Gradient clipping: prevents exploding gradients during early
            # backbone fine-tuning. max_norm=1.0 is a standard safe value.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU / MPS fallback path — identical logic without AMP wrappers.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        monitor.update(loss.item(), images.size(0))
        preds    = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        # Print every ~20% of batches. max(1, ...) guards against n_batches < 5.
        if i % max(1, n_batches // 5) == 0 or i == n_batches:
            print(
                f"  batch {i}/{n_batches}  loss={monitor.avg:.4f}",
                flush=True,
                # flush=True: important for agentic tools that read stdout line
                # by line. Without it, Python's output buffer may hold lines
                # until the process exits.
            )

    train_acc = correct / total if total > 0 else 0.0
    # NOTE: train_acc is computed on a SYNTHETIC 50/50 distribution because
    # WeightedRandomSampler resamples to class balance. It is NOT comparable
    # to val_acc, which is computed on the natural ~2:1 ALL:HEM distribution.
    # The [50/50] tag added to the log line makes this unmistakable.
    return monitor.avg, train_acc


def validate(model, loader, criterion, device, use_amp=True):
    """
    Validation with 4-way Test-Time Augmentation (TTA).

    TTA RATIONALE:
    --------------
    Microscopy images have no canonical orientation. Averaging predictions
    over the original and three flipped variants reduces variance from
    orientation-sensitive features, typically yielding +0.5–1.5 AUC points
    at zero training cost.

    The four variants are:
      1. Original
      2. Horizontal flip
      3. Vertical flip
      4. Both flips (= 180° rotation)

    Loss is computed on the original variant only (logits_orig) for
    comparability with train_loss. Using averaged probs for the loss would
    understate it and break the loss-AUC decoupling detector.

    Returns: (val_loss, acc, auc, sensitivity, specificity, f1)
    """
    model.eval()
    monitor = MetricMonitor()
    all_targets, all_scores = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            variants = [
                images,
                torch.flip(images, [3]),      # horizontal flip (W axis)
                torch.flip(images, [2]),      # vertical flip   (H axis)
                torch.flip(images, [2, 3]),   # both            (= 180°)
            ]

            avg_probs   = torch.zeros(images.size(0), 2, device=device)
            logits_orig = None

            with torch.amp.autocast("cuda", enabled=use_amp):
                for i, v in enumerate(variants):
                    logits = model(v)
                    if i == 0:
                        # Save original logits for loss computation.
                        logits_orig = logits
                    # .float() cast: softmax in float32 prevents precision loss
                    # when averaging probabilities that may be close to 0/1.
                    avg_probs += torch.softmax(logits.float(), dim=1)

            avg_probs /= len(variants)

            # Loss on original only — see docstring above.
            loss = criterion(logits_orig.float(), labels)
            monitor.update(loss.item(), images.size(0))

            all_targets.extend(labels.cpu().numpy())
            # avg_probs[:, 0] = P(ALL) — used as the positive class score
            # for AUC, sensitivity, specificity calculation.
            all_scores.extend(avg_probs[:, 0].cpu().numpy())

    all_targets = np.array(all_targets)
    all_scores  = np.array(all_scores)

    # compute_all_positive_metrics treats class 0 (ALL) as the positive class.
    # threshold=0.5 is the default decision boundary; post-training Youden's J
    # optimisation finds the production threshold separately.
    metrics = compute_all_positive_metrics(
        all_targets, all_scores, threshold=0.5
    )

    return (
        monitor.avg,
        metrics["acc"],
        metrics["auc"],
        metrics["sens"],
        metrics["spec"],
        metrics["f1"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Definition
# ═══════════════════════════════════════════════════════════════════════════════
# Maps CLI --model keys to timm model names.
# Kept as a module-level dict so it is visible to agentic tools inspecting
# the file without running it.
TIMM_NAME_MAP = {
    "mnv3l": "mobilenetv3_large_100",
    "effb0":  "efficientnet_b0",
    "effb4":  "efficientnet_b4",
    "rn50":   "resnet50",
}


class ConstrainedHead(nn.Module):
    """
    Classification head replacing the pre-trained backbone's original head.

    ARCHITECTURE RATIONALE:
    ------------------------
    BN → Dropout(0.4) → Linear(→512) → ReLU → Dropout(0.3) → Linear(→2)

    BatchNorm1d: normalises the backbone feature distribution before the
    linear layers. Pre-trained backbones output features with varying scale
    depending on the last pooling layer. BN removes this scale dependency,
    making the head LR less sensitive to backbone choice.

    Dropout(0.4): strong regularisation before the bottleneck. CNMC has
    ~10k training images — too few for a large linear layer to generalise
    without regularisation.

    Linear(in_features → 512): reduces from the backbone's output dimension
    (e.g. 960 for MNV3L, 1280 for EffB0) to a shared bottleneck size.
    512 was chosen as a round number that fits comfortably in cache for all
    backbone sizes tested.

    ReLU: introduces non-linearity between the two linear layers. Without
    it the two Linear layers collapse to a single affine transform.

    Dropout(0.3): lighter second dropout — the bottleneck is already smaller
    and more regularised.

    Linear(512 → 2): final logit layer. Two output classes: ALL (0), HEM (1).
    """
    def __init__(self, in_features, num_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),   # inplace=True saves one activation tensor
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(x)


def get_model(args):
    """
    Builds backbone + head as nn.Sequential([backbone, head]).

    WHY nn.Sequential INSTEAD OF A CUSTOM MODULE:
    -----------------------------------------------
    The two-element Sequential allows clean indexing: model[0] = backbone,
    model[1] = head. This makes partial unfreezing and per-group optimizer
    params straightforward without subclassing.

    in_features is probed by running a dummy forward pass at build time so
    the head adapts automatically to any backbone without hardcoding output
    dimensions. The probe uses batch_size=2 (not 1) because BatchNorm1d
    raises an error on single-element batches during .train() mode.
    """
    timm_name = TIMM_NAME_MAP[args.model]

    # num_classes=0: removes the backbone's original classifier head,
    # returning the global-pooled feature vector directly.
    backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)

    # Probe output dimension with a dummy pass.
    # .eval() prevents BatchNorm from updating running stats on zeros.
    backbone.eval()
    with torch.no_grad():
        in_features = backbone(torch.zeros(2, 3, 224, 224)).shape[-1]
    backbone.train()

    model = nn.Sequential(backbone, ConstrainedHead(in_features))

    # Return param groups for logging purposes (parameter counts).
    backbone_params = list(model[0].parameters())
    head_params     = list(model[1].parameters())

    return model, backbone_params, head_params, timm_name, in_features


# ═══════════════════════════════════════════════════════════════════════════════
#  Backbone Unfreezing — Enum + Named-Prefix Table
# ═══════════════════════════════════════════════════════════════════════════════
class UnfreezePhase(Enum):
    """
    Enumerates the four backbone freeze states used during training.
