"""
experiments/eval_phase1.py

Phase 1: no-defense robustness sweep across compression levels × attacks.

For each (compression, attack) pair the script records:
    model, compression, attack, clean_acc, robust_acc, asr, robustness_gap

Results are written to results/phase1_results.csv immediately after each pair
completes.  Already-written rows are detected on startup and skipped, so the
script is safe to interrupt and re-run (resumable).

Usage:
    python experiments/eval_phase1.py [--model deit_small|deit_base]
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

# Ensure project root is on sys.path so sibling packages resolve correctly
# whether the script is run from the project root or from experiments/.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models.loader import load_config, load_model, LogitsWrapper
import attacks.fgsm as fgsm_mod
from utils.metrics import (
    clean_accuracy,
    robust_accuracy,
    attack_success_rate,
    robustness_gap,
    save_results_to_csv,
)

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_FILE = "results/phase1_results.csv"
ATTACK_NAMES = ["fgsm", "pgd", "patch"]


# ── Data loading ──────────────────────────────────────────────────────────────

# When using ImageNette (10-class subset) instead of full ImageNet-1k, the
# ImageFolder class indices (0–9) don't match the pretrained model's output
# indices.  This table remaps each synset to its correct ImageNet-1k position.
_IMAGENETTE_TO_IMAGENET: dict[str, int] = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def _is_synset_id(name: str) -> bool:
    """Return True if name looks like an ImageNet synset ID (e.g. 'n01440764')."""
    return len(name) == 9 and name[0] == "n" and name[1:].isdigit()


def _remap_subset_labels(dataset: ImageFolder) -> ImageFolder:
    """Remap ImageFolder targets to ImageNet-1k indices.

    Two cases:

    1. Full ImageNet-1k (synset-ID folder names like 'n01440764'):
       ImageFolder alphabetical order == ImageNet-1k label order, so NO
       remapping is needed.  Detected by checking folder name format, NOT
       class count (count-based check breaks with <1000 classes downloaded).

    2. ImageNette (10-class subset with synset-ID folders):
       ImageFolder assigns indices 0-9 alphabetically, which do NOT match
       the model's expected ImageNet-1k indices.  Remap using the lookup table.
    """
    sample_class = dataset.classes[0] if dataset.classes else ""

    if _is_synset_id(sample_class):
        # Folder names are synset IDs — alphabetical order == ImageNet label order.
        # No remapping needed regardless of how many classes are present.
        print(f"[data] Synset-ID folders detected ({len(dataset.classes)} classes) "
              f"— labels already correct, skipping remap.")
        return dataset

    # Folder names are NOT synset IDs (e.g. human-readable ImageNette names).
    # Remap using the 10-entry lookup table.
    print(f"[data] Non-synset folders detected — applying ImageNette→ImageNet remap.")
    new_samples = []
    for path, lbl in dataset.samples:
        synset = dataset.classes[lbl]
        new_lbl = _IMAGENETTE_TO_IMAGENET.get(synset, lbl)
        new_samples.append((path, new_lbl))
    dataset.samples = new_samples
    dataset.targets = [lbl for _, lbl in new_samples]
    return dataset


def build_val_loader(cfg: dict, device: str) -> DataLoader:
    """Build a deterministic subset loader for the validation set.

    Works with both full ImageNet-1k and ImageNette (10-class subset).
    Labels are remapped to ImageNet-1k indices automatically when a subset
    is detected (< 1000 classes).

    The subset is drawn with seed=42 via randperm, matching the fixed split
    described in configs/base.yaml so results are reproducible across runs.
    """
    ds_cfg = cfg["dataset"]
    eval_cfg = cfg["eval"]

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(ds_cfg["image_size"]),
        T.ToTensor(),
        T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
    ])

    full_dataset = ImageFolder(root=str(_ROOT / ds_cfg["val_dir"]), transform=transform)
    full_dataset = _remap_subset_labels(full_dataset)

    rng = torch.Generator()
    rng.manual_seed(cfg["seed"])
    n = min(ds_cfg["val_subset_size"], len(full_dataset))
    indices = torch.randperm(len(full_dataset), generator=rng)[:n].tolist()
    print(f"[phase1] Val subset : {n} images, seed={cfg['seed']}, first 5 indices={indices[:5]}")
    subset = Subset(full_dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=eval_cfg["num_workers"],
        pin_memory=(device == "cuda"),
    )
    return loader


# ── Resumability helpers ──────────────────────────────────────────────────────

def load_completed_runs(results_path: str) -> set:
    """Return the set of (model, compression, attack) tuples already in the CSV.

    Works with the 9-column schema written by save_results_to_csv:
        timestamp, model, compression, defense, attack, clean_acc,
        robust_acc, asr, robustness_gap, phase
    The resumability key tuple (model, compression, attack) is unchanged.
    """
    completed: set = set()
    if not os.path.isfile(results_path):
        return completed
    with open(results_path, newline="") as f:
        for row in csv.DictReader(f):
            completed.add((row["model"], row["compression"], row["attack"]))
    return completed


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_model_device(model) -> str:
    """Return the device string for the first model parameter found."""
    for p in model.parameters():
        return str(p.device)
    return "cpu"


@torch.no_grad()
def run_clean_eval(
    model,
    loader: DataLoader,
    model_device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect clean logits and labels for the full loader.

    Returns:
        all_logits:  (N, C) tensor on CPU.
        all_labels:  (N,)   tensor on CPU.
    """
    model.eval()   # enforce eval mode — guards against accidental model.train() upstream
    logits_list, labels_list = [], []
    for images, labels in loader:
        images = images.to(model_device)
        logits = model(images)
        logits_list.append(logits.cpu())
        labels_list.append(labels.cpu())
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def run_adv_eval(
    attack,
    model,
    loader: DataLoader,
    model_device: str,
) -> torch.Tensor:
    """Run the attack on every batch and collect adversarial logits.

    Gradient context is managed by the attack objects themselves; this function
    does not suppress gradients.

    Returns:
        all_adv_logits: (N, C) tensor on CPU.
    """
    adv_logits_list = []
    for images, labels in loader:
        images = images.to(model_device)
        labels = labels.to(model_device)
        adv_images = attack(images, labels)
        with torch.no_grad():
            adv_logits = model(adv_images)
        adv_logits_list.append(adv_logits.cpu())
    return torch.cat(adv_logits_list, dim=0)


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(results_path: str, model_name: str) -> None:
    """Print a formatted table of all phase1 rows for the given model."""
    if not os.path.isfile(results_path):
        return
    header = f"\n{'compression':<12} {'attack':<8} {'clean_acc':>10} {'robust_acc':>11} {'asr':>8} {'gap':>8}"
    print(header)
    print("-" * len(header.strip()))
    with open(results_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] != model_name:
                continue
            print(
                f"{row['compression']:<12} {row['attack']:<8} "
                f"{float(row['clean_acc']):>10.4f} "
                f"{float(row['robust_acc']):>11.4f} "
                f"{float(row['asr']):>8.4f} "
                f"{float(row['robustness_gap']):>8.4f}"
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 — no-defense robustness sweep: compression × attack."
    )
    parser.add_argument(
        "--model",
        default="deit_small",
        choices=["deit_small", "deit_base"],
        help="Model to evaluate (default: deit_small)",
    )
    args = parser.parse_args()
    model_name: str = args.model

    cfg = load_config(str(_ROOT / "configs/base.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[phase1] device      : {device}")
    print(f"[phase1] model       : {model_name}")
    print(f"[phase1] results     : {RESULTS_FILE}")
    print()

    loader = build_val_loader(cfg, device)
    completed = load_completed_runs(RESULTS_FILE)

    if completed:
        print(f"[phase1] Resuming — {len(completed)} combination(s) already done, skipping those.\n")

    compression_levels: list[str] = cfg["compression"]["levels"]

    for compression in compression_levels:
        remaining = [a for a in ATTACK_NAMES if (model_name, compression, a) not in completed]

        if not remaining:
            print(f"[phase1] {compression:<6}: all attacks already done — skipping model load.")
            continue

        # ── Load model (once per compression level) ───────────────────────────
        print(f"[phase1] {compression:<6}: loading {model_name} …")
        try:
            raw_model = load_model(model_name, compression, cfg, device=device)
        except Exception as exc:
            print(f"[phase1] {compression:<6}: load failed — {exc}")
            continue

        model = LogitsWrapper(raw_model)
        model.eval()
        model_device = infer_model_device(raw_model)
        print(f"[phase1] {compression:<6}: model on {model_device}")

        # ── Clean accuracy (computed once, reused for all attacks) ────────────
        print(f"[phase1] {compression:<6}: evaluating clean accuracy …")
        try:
            clean_logits, clean_labels = run_clean_eval(model, loader, model_device)
        except Exception as exc:
            print(f"[phase1] {compression:<6}: clean eval failed — {exc}")
            del raw_model, model
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        c_acc = clean_accuracy(clean_logits, clean_labels)
        print(f"[phase1] {compression:<6}: clean_acc = {c_acc:.4f}")

        # ── Per-attack robustness sweep ────────────────────────────────────────
        for attack_name in remaining:
            print(f"[phase1] {compression:<6} × {attack_name:<5}: building attack …")

            if attack_name == "fgsm":
                attack = fgsm_mod.build_attack(model, cfg)
            elif attack_name == "pgd":
                import attacks.pgd as pgd_mod   # lazy import — pgd.py may not exist yet
                attack = pgd_mod.build_attack(model, cfg)
            elif attack_name == "patch":
                import attacks.patch as patch_mod  # lazy import — patch.py may not exist yet
                attack = patch_mod.build_attack(model, cfg)
            else:
                raise ValueError(f"Unknown attack: {attack_name!r}")

            print(f"[phase1] {compression:<6} × {attack_name:<5}: running on {len(loader.dataset)} images …")
            try:
                adv_logits = run_adv_eval(attack, model, loader, model_device)
            except Exception as exc:
                print(f"[phase1] {compression:<6} × {attack_name:<5}: attack failed — {exc}")
                continue

            rob_acc = robust_accuracy(adv_logits, clean_labels)
            asr = attack_success_rate(clean_logits, adv_logits, clean_labels)
            rob_gap = robustness_gap(clean_logits, adv_logits, clean_labels)

            print(
                f"[phase1] {compression:<6} × {attack_name:<5}: "
                f"robust_acc={rob_acc:.4f}  asr={asr:.4f}  gap={rob_gap:.4f}"
            )

            save_results_to_csv(
                results_dir=str(_ROOT / "results"),
                model=model_name,
                compression=compression,
                defense="none",           # phase 1 has no defense
                attack=attack_name,
                clean_acc=c_acc,
                robust_acc=rob_acc,
                asr=asr,
                robustness_gap_val=rob_gap,
                phase=1,
                filename="phase1_results.csv",
            )
            print(f"[phase1] {compression:<6} × {attack_name:<5}: saved → {RESULTS_FILE}")

        # ── Free GPU memory before loading the next compression level ──────────
        del raw_model, model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n[phase1] All combinations complete.")
    print_summary(RESULTS_FILE, model_name)


if __name__ == "__main__":
    main()
