"""
defenses/pgd_at.py

PGD Adversarial Training (PGD-AT) — fine-tunes an already-compressed model on PGD
adversarial inputs for a fixed number of epochs.

All parameters come from configs/base.yaml — never hardcode values here.
Compression must be applied before calling this module.

Usage:
    from defenses.pgd_at import pgd_adversarial_train
    hardened_model = pgd_adversarial_train(model, train_loader, config)
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Resolve project root so sibling packages import cleanly whether this module
# is imported from the project root or from defenses/.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torchattacks
from models.loader import load_config, LogitsWrapper  # noqa: F401 — load_config re-exported for convenience


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Perturbation sanity check
# ---------------------------------------------------------------------------

def _check_pgd_perturbation(
    pgd: torchattacks.PGD,
    train_loader: torch.utils.data.DataLoader,
    at_eps: float,
    model_device: str,
    mean: list,
    std: list,
) -> None:
    """Assert that the PGD L-inf perturbation in pixel space is within 10% of at_eps.

    Grabs one batch from train_loader, generates adversarial examples, then
    un-normalises both clean and adversarial images to [0, 1] pixel space.
    Asserts that the L-inf of (adv − clean) lies in [at_eps * 0.9, at_eps * 1.1].

    This check fires before any training epoch.  If it fails a ValueError is
    raised immediately so no compute is wasted on a misconfigured run.

    Args:
        pgd:          torchattacks.PGD instance with set_normalization_used
                      already called.
        train_loader: DataLoader yielding ImageNet-normalised (mean/std) images.
        at_eps:       Configured epsilon (defense_pgd.at_eps from base.yaml).
        model_device: Device string for moving tensors to match the model.
        mean:         ImageNet normalisation mean — 3-element list.
        std:          ImageNet normalisation std  — 3-element list.

    Raises:
        ValueError: If the measured L-inf is outside at_eps ± 10%.
    """
    print("[PGD-AT] Running PGD perturbation sanity check …")

    images, labels = next(iter(train_loader))
    images = images.to(model_device)
    labels = labels.to(model_device)

    # Generate adversarial examples (torchattacks handles grad internally).
    adv_images = pgd(images, labels)

    # Un-normalise to [0, 1] pixel space for measurement.
    mean_t = torch.tensor(mean, dtype=images.dtype, device=model_device).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  dtype=images.dtype, device=model_device).view(1, 3, 1, 1)
    images_px = (images     * std_t + mean_t).clamp(0.0, 1.0)
    adv_px    = (adv_images * std_t + mean_t).clamp(0.0, 1.0)

    linf = (adv_px - images_px).abs().max().item()

    lo = at_eps * 0.9
    hi = at_eps * 1.1
    print(
        f"[PGD-AT] Perturbation L-inf (pixel space) : {linf:.5f}  "
        f"(expected {at_eps:.5f} ± 10%  →  [{lo:.5f}, {hi:.5f}])"
    )

    if not (lo <= linf <= hi):
        raise ValueError(
            f"PGD perturbation sanity check FAILED — training aborted.\n"
            f"  Measured L-inf (pixel space) : {linf:.5f}\n"
            f"  Expected range               : [{lo:.5f}, {hi:.5f}]\n"
            f"  Configured at_eps            : {at_eps:.5f}  "
            f"({round(at_eps * 255)}/255)\n"
            "  Likely cause: set_normalization_used() was not called on the\n"
            "  PGD attack, so perturbations were applied in normalised space\n"
            "  (~[-2.1, 2.6]) instead of pixel space ([0, 1])."
        )

    print("[PGD-AT] Perturbation sanity check PASSED.\n")


# ---------------------------------------------------------------------------
# Clean accuracy helper
# ---------------------------------------------------------------------------

def _measure_clean_acc(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> float:
    """Measure clean accuracy on the given loader."""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            if hasattr(logits, "logits"):
                logits = logits.logits
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    if was_training:
        model.train()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    epoch: int,
    compression: str,
    checkpoint_dir: str,
) -> str:
    """Save model checkpoint after a training epoch."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if compression == "int4":
        filename = f"pgdat_{compression}_epoch{epoch:02d}_full_model.pt"
        ckpt_path = os.path.join(checkpoint_dir, filename)
        torch.save(model, ckpt_path)
    else:
        filename = f"pgdat_{compression}_epoch{epoch:02d}.pt"
        ckpt_path = os.path.join(checkpoint_dir, filename)
        torch.save(model.state_dict(), ckpt_path)
    return os.path.abspath(ckpt_path)


# ---------------------------------------------------------------------------
# Layer-freeze helper
# ---------------------------------------------------------------------------

def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers, then unfreeze the last 4 transformer blocks and head."""
    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        blocks = model.vit.encoder.layer
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        blocks = model.encoder.layer
    elif hasattr(model, "blocks"):
        blocks = model.blocks
    else:
        raise ValueError(
            "[PGD-AT] Cannot detect model architecture for layer freezing."
        )

    for param in model.parameters():
        if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
            param.requires_grad = False

    for block in list(blocks)[-4:]:
        for param in block.parameters():
            if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                param.requires_grad = True

    for name, param in model.named_parameters():
        if "classifier" in name or "head" in name:
            if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[PGD-AT] Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def pgd_adversarial_train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    config: dict,
    compression: str = "fp32",
) -> nn.Module:
    """Fine-tune an already-compressed model using PGD adversarial inputs.

    Training protocol:
      - Backbone frozen; only last 4 transformer blocks + classifier head trained.
      - Optimizer : AdamW (weight_decay from config["defense_pgd"])
      - LR schedule: linear warmup — lr/10 for epoch 1, full lr from epoch 2.
      - Loss       : CrossEntropy on PGD-perturbed inputs using AMP.
      - Epochs     : config["defense_pgd"]["epochs"]  (7)
      - Checkpoint saved after every epoch to config["paths"]["checkpoints_pgdat_dir"]
    """
    _set_seeds(config["seed"])

    defense_cfg = config["defense_pgd"]
    ds_cfg      = config["dataset"]
    ckpt_dir    = config["paths"]["checkpoints_pgdat_dir"]
    
    epochs: int            = defense_cfg["epochs"]
    weight_decay: float    = defense_cfg["weight_decay"]
    at_eps: float          = defense_cfg["at_eps"]
    at_alpha: float        = defense_cfg["at_alpha"]
    at_steps: int          = defense_cfg["at_steps"]
    warmup_epochs: int     = int(defense_cfg.get("warmup_epochs", 1))
    save_every_epoch: bool = defense_cfg.get("save_every_epoch", True)
    mean: list             = ds_cfg["mean"]
    std: list              = ds_cfg["std"]

    # Per-compression learning rate.
    lr_cfg = defense_cfg["lr"]
    if isinstance(lr_cfg, dict):
        if compression not in lr_cfg:
            raise KeyError(
                f"[PGD-AT] No LR configured for compression='{compression}'."
            )
        lr: float = float(lr_cfg[compression])
    else:
        lr = float(lr_cfg)

    model_device = next(model.parameters()).device
    device_type = "cuda" if "cuda" in str(model_device) else "cpu"

    # Build PGD attack bound to this model.
    # random_start=True is critical to prevent catastrophic overfitting.
    pgd_attack = torchattacks.PGD(LogitsWrapper(model), eps=at_eps, alpha=at_alpha, steps=at_steps, random_start=True)
    pgd_attack.set_normalization_used(mean=mean, std=std)

    _freeze_backbone(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
    )

    def _warmup_lambda(epoch_idx: int) -> float:
        return 0.1 if epoch_idx < warmup_epochs else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_lambda)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize AMP scaler for efficiency
    scaler = torch.amp.GradScaler(device=device_type) if device_type == "cuda" else None

    print(f"[PGD-AT] Starting PGD adversarial training — {epochs} epoch(s)")
    print(f"[PGD-AT] compression  : {compression}")
    print(f"[PGD-AT] device       : {model_device} (AMP enabled)")
    print(f"[PGD-AT] optimizer    : AdamW  lr={lr}  weight_decay={weight_decay}")
    print(f"[PGD-AT] warmup_epochs: {warmup_epochs}  (epoch 1 uses lr={lr * 0.1:.2e})")
    print(f"[PGD-AT] PGD settings : eps={at_eps:.5f}, alpha={at_alpha:.5f}, steps={at_steps}")
    print(f"[PGD-AT] checkpoints  : {ckpt_dir}\n")

    _check_pgd_perturbation(pgd_attack, train_loader, at_eps, model_device, mean, std)

    print("[PGD-AT] Measuring baseline clean accuracy (500-image subset) …")
    baseline_loader = DataLoader(
        Subset(train_loader.dataset, range(500)),
        batch_size=64,
        shuffle=False,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
    )
    baseline_clean_acc = _measure_clean_acc(model, baseline_loader, str(model_device))
    print(f"[PGD-AT] Baseline clean_acc : {baseline_clean_acc:.4f}\n")

    for epoch in range(1, epochs + 1):
        model.train()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[PGD-AT] Epoch {epoch}/{epochs} — effective lr={current_lr:.2e}")

        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            leave=True,
            dynamic_ncols=True,
        )

        for images, labels in loop:
            images = images.to(model_device)
            labels = labels.to(model_device)

            # Generate PGD adversarial examples
            adv_images = pgd_attack(images, labels)

            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                    logits = model(adv_images)
                    if hasattr(logits, "logits"):
                        logits = logits.logits
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(adv_images)
                if hasattr(logits, "logits"):
                    logits = logits.logits
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            loop.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{correct / total:.4f}",
            )

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc  = correct / total
        print(
            f"[PGD-AT] Epoch {epoch}/{epochs} — "
            f"loss={epoch_loss:.4f}  train_adv_acc={epoch_acc:.4f}"
        )

        epoch_clean_acc = _measure_clean_acc(model, baseline_loader, str(model_device))
        clean_drop = baseline_clean_acc - epoch_clean_acc
        print(
            f"[PGD-AT] Epoch {epoch} clean_acc={epoch_clean_acc:.4f}  "
            f"(baseline={baseline_clean_acc:.4f}  drop={clean_drop:+.4f})"
        )

        if clean_drop > 0.15:
            print(
                f"\n[PGD-AT] *** WARNING: clean_acc dropped {clean_drop:.4f} "
                f"(> 0.15 threshold) after epoch {epoch}. ***\n"
            )

        if save_every_epoch:
            ckpt_path = save_checkpoint(model, epoch, compression, ckpt_dir)
            print(f"[PGD-AT] Checkpoint saved → {ckpt_path}")

    model.eval()
    print(f"\n[PGD-AT] Training complete.  Model returned in eval mode.")
    return model
