"""
scripts/setup_data_colab.py

Downloads ImageNet-1k images and saves them in torchvision ImageFolder format:
  • 5 000 validation images → data/imagenet/val/
  • 10 000 training images  → data/imagenet/train/   (needed for AT / AT+KD)

Run once in Colab before any experiment:
    !python scripts/setup_data_colab.py --method a   # HuggingFace (recommended)
    !python scripts/setup_data_colab.py --method b   # Kaggle (val only)

Both produce:
    data/imagenet/val/
    ├── n01440764/
    │   ├── img_00000.JPEG
    │   └── ...
    data/imagenet/train/
    ├── n01440764/
    │   ├── img_00000.JPEG
    │   └── ...
"""

import argparse
import json
import os
import random
import shutil
import sys
import urllib.request
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
N_IMAGES = 5000        # validation images  — must match dataset.val_subset_size in base.yaml
N_TRAIN_IMAGES = 10000 # training images    — must match dataset.train_subset_size in base.yaml
VAL_DIR   = "data/imagenet/val"
TRAIN_DIR = "data/imagenet/train"

# ── Label index → synset ID mapping ─────────────────────────────────────────
#
# HuggingFace imagenet-1k dataset.features["label"].names returns HUMAN-READABLE
# strings like "tench", "goldfish" — NOT synset IDs like "n01440764".
# Using those names as folder names causes ImageFolder's alphabetical sort to
# produce a completely different label order from what the model expects,
# giving clean_acc ≈ 0.0002 (worse than random).
#
# Fix: use PyTorch's canonical imagenet_class_index.json which maps
# integer label → [synset_id, human_name] in the correct ImageNet-1k order.
# e.g. {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
# Folder names become n01440764/, n01443537/, … which sort alphabetically
# in the SAME order as ImageNet-1k labels — matching what the model expects.
# ─────────────────────────────────────────────────────────────────────────────

_CLASS_INDEX_URLS = [
    # Google Storage — most reliably accessible from Colab
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
    # PyTorch Hub fallback
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_class_index.json",
    # AWS fallback
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
]


def _get_label_to_synset() -> dict:
    """Return {label_int: synset_id_str} from a canonical imagenet_class_index.json.

    Tries three URLs in order so a single CDN outage does not break the run.
    All three sources share the same format:
        {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
    """
    for url in _CLASS_INDEX_URLS:
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                class_index = json.load(resp)
            mapping = {int(k): v[0] for k, v in class_index.items()}
            print(f"[setup] Label → synset mapping loaded ({len(mapping)} classes) from:\n[setup]   {url}")
            return mapping
        except Exception as exc:
            print(f"[setup] Could not fetch {url}: {exc} — trying next URL …")

    # All URLs failed — make it impossible to miss
    print("[setup] " + "=" * 70)
    print("[setup] FATAL: all class-index URLs failed.")
    print("[setup] Folder names will be class_0000/ (WRONG) — accuracy will be ~0.")
    print("[setup] Do NOT proceed with evaluation. Fix network access and re-run.")
    print("[setup] " + "=" * 70)
    return {i: f"class_{i:04d}" for i in range(1000)}



# ─────────────────────────────────────────────────────────────────────────────
# METHOD A — HuggingFace datasets streaming (recommended)
#
# Streams only the images you need — no multi-GB download.
# Requires:
#   1. A HuggingFace account
#   2. Accept terms at: https://huggingface.co/datasets/imagenet-1k
#   3. Your HF token (from https://huggingface.co/settings/tokens)
# ─────────────────────────────────────────────────────────────────────────────

def setup_method_a():
    # ── Install deps if missing ───────────────────────────────────────────────
    try:
        from huggingface_hub import login
    except ImportError:
        print("[setup] Installing huggingface_hub …")
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import login

    try:
        from datasets import load_dataset
    except ImportError:
        print("[setup] Installing datasets …")
        os.system("pip install -q datasets pillow")
        from datasets import load_dataset

    # ── Login ─────────────────────────────────────────────────────────────────
    # Priority: env var HF_TOKEN → Colab secret HF_TOKEN → interactive prompt
    token = os.environ.get("HF_TOKEN", None)

    if token is None:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
            print("[setup] Using HF_TOKEN from Colab secrets.")
        except Exception:
            pass

    if token:
        login(token=token, add_to_git_credential=False)
    else:
        print("[setup] No HF_TOKEN found — prompting for login.")
        print("[setup] Get your token at: https://huggingface.co/settings/tokens")
        login()  # interactive prompt

    # ── Stream dataset ────────────────────────────────────────────────────────
    print(f"\n[setup] Streaming {N_IMAGES} images from imagenet-1k validation split …")

    dataset = load_dataset(
        "imagenet-1k",
        split="validation",
        streaming=True,
    )
    dataset = dataset.shuffle(seed=SEED, buffer_size=5000)

    # IMPORTANT: do NOT use dataset.features["label"].names for folder names.
    # That returns human-readable strings ("tench", "goldfish") which sort
    # alphabetically in a different order than ImageNet-1k label indices,
    # causing clean_acc ≈ 0.0002.  Use synset IDs from PyTorch instead.
    label_to_synset = _get_label_to_synset()

    saved = 0
    for example in dataset:
        if saved >= N_IMAGES:
            break

        label_id: int = example["label"]
        image = example["image"]
        synset = label_to_synset.get(label_id, f"class_{label_id:04d}")

        class_dir = Path(VAL_DIR) / synset
        class_dir.mkdir(parents=True, exist_ok=True)

        img_path = class_dir / f"img_{saved:05d}.JPEG"
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(img_path, "JPEG")

        saved += 1
        if saved % 100 == 0:
            print(f"[setup]   {saved}/{N_IMAGES} saved …")

    print(f"[setup] Done — {saved} images written to {VAL_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# METHOD B — Kaggle API
#
# Downloads the full validation tar (~6 GB) then keeps N_IMAGES images.
# Requires:
#   1. kaggle.json (from kaggle.com → Settings → API → Create New Token)
#   2. Accept competition terms at:
#      https://www.kaggle.com/competitions/imagenet-object-localization-challenge
# ─────────────────────────────────────────────────────────────────────────────

def setup_method_b():
    # ── Place kaggle.json ─────────────────────────────────────────────────────
    kaggle_cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_cfg.exists():
        print("[setup] kaggle.json not found — opening Colab file picker …")
        try:
            from google.colab import files
            uploaded = files.upload()
            kaggle_cfg.parent.mkdir(parents=True, exist_ok=True)
            for fname, data in uploaded.items():
                kaggle_cfg.write_bytes(data)
                print(f"[setup] Saved → {kaggle_cfg}")
        except ImportError:
            sys.exit(
                "[setup] Not in Colab and ~/.kaggle/kaggle.json missing.\n"
                "        Place your kaggle.json there and re-run."
            )
    kaggle_cfg.chmod(0o600)

    # ── Install kaggle CLI ────────────────────────────────────────────────────
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[setup] Installing kaggle …")
        os.system("pip install -q kaggle")

    # ── Download val tar ──────────────────────────────────────────────────────
    print("[setup] Downloading ImageNet val split from Kaggle (~6 GB) …")
    os.makedirs("data/imagenet", exist_ok=True)
    os.system(
        "kaggle competitions download "
        "imagenet-object-localization-challenge "
        "-f ILSVRC/Data/CLS-LOC/val.tar "
        "-p data/imagenet"
    )

    # ── Extract flat ──────────────────────────────────────────────────────────
    flat_dir = "data/imagenet/val_flat"
    os.makedirs(flat_dir, exist_ok=True)
    print("[setup] Extracting …")
    os.system(f"tar -xf data/imagenet/val.tar -C {flat_dir} --strip-components=5")

    # ── Download synset label mapping ─────────────────────────────────────────
    print("[setup] Downloading label mapping …")
    os.system(
        "wget -q https://raw.githubusercontent.com/tensorflow/models/master/"
        "research/slim/datasets/imagenet_2012_validation_synset_labels.txt "
        "-O data/imagenet/val_labels.txt"
    )

    label_file = Path("data/imagenet/val_labels.txt")
    if not label_file.exists():
        sys.exit("[setup] Could not download label mapping.")

    synset_per_image = label_file.read_text().strip().splitlines()
    flat_images = sorted(Path(flat_dir).glob("*.JPEG"))
    print(f"[setup] Found {len(flat_images)} extracted images.")

    random.seed(SEED)
    selected = random.sample(range(len(flat_images)), min(N_IMAGES, len(flat_images)))

    print(f"[setup] Moving {N_IMAGES} images into ImageFolder structure …")
    out_dir = Path(VAL_DIR)
    for idx in selected:
        img_path = flat_images[idx]
        synset = synset_per_image[idx]
        dest_dir = out_dir / synset
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, dest_dir / img_path.name)

    print(f"[setup] Done — {N_IMAGES} images in {VAL_DIR}")
    print("[setup] Cleaning up temp files …")
    shutil.rmtree(flat_dir, ignore_errors=True)


# ── Training data (Method A only) ────────────────────────────────────────────
#
# AT and AT+KD (Phase 2) require 10 000 training images.
# Uses the same HuggingFace streaming approach as setup_method_a() — no full
# dataset download; only the images you need are fetched.
# ─────────────────────────────────────────────────────────────────────────────

def setup_train():
    """Stream 10 000 ImageNet-1k training images into data/imagenet/train/.

    Requires the same HuggingFace login as setup_method_a().
    Skips automatically if TRAIN_DIR already exists and is non-empty.
    """
    try:
        from huggingface_hub import login
    except ImportError:
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import login

    try:
        from datasets import load_dataset
    except ImportError:
        os.system("pip install -q datasets pillow")
        from datasets import load_dataset

    # Reuse same token logic as setup_method_a()
    token = os.environ.get("HF_TOKEN", None)
    if token is None:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            pass
    if token:
        login(token=token, add_to_git_credential=False)
    else:
        login()  # interactive prompt

    print(f"\n[setup] Streaming {N_TRAIN_IMAGES} images from imagenet-1k train split …")

    dataset = load_dataset(
        "imagenet-1k",
        split="train",
        streaming=True,
    )
    dataset = dataset.shuffle(seed=SEED, buffer_size=20000)

    # IMPORTANT: same fix as setup_method_a() — use synset IDs, not human names.
    label_to_synset = _get_label_to_synset()

    saved = 0
    for example in dataset:
        if saved >= N_TRAIN_IMAGES:
            break

        label_id = example["label"]
        synset = label_to_synset.get(label_id, f"class_{label_id:04d}")
        class_dir = Path(TRAIN_DIR) / synset
        class_dir.mkdir(parents=True, exist_ok=True)

        img = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(class_dir / f"img_{saved:05d}.JPEG", "JPEG")

        saved += 1
        if saved % 500 == 0:
            print(f"[train] {saved}/{N_TRAIN_IMAGES} saved …")

    print(f"[setup] Done — {saved} training images written to {TRAIN_DIR}")


# ── Verification ──────────────────────────────────────────────────────────────

def verify():
    """Verify val dir loads correctly and folder names look like synset IDs."""
    try:
        from torchvision.datasets import ImageFolder
        ds = ImageFolder(VAL_DIR)
        print(f"\n[verify] ImageFolder loaded OK")
        print(f"[verify]   images  : {len(ds)}")
        print(f"[verify]   classes : {len(ds.classes)}")
        if len(ds) < 100:
            print("[verify] WARNING: fewer than 100 images — check the setup.")
        # Sanity-check folder names: synset IDs start with 'n' followed by 8 digits.
        # Human-readable names (tench, goldfish …) would indicate the old bug.
        sample_class = ds.classes[0]
        if not (sample_class.startswith("n") and len(sample_class) == 9 and sample_class[1:].isdigit()):
            print(
                f"[verify] ERROR: first folder is '{sample_class}' — expected a synset ID "
                "like 'n01440764'.  This means label_to_synset used human-readable names.\n"
                "[verify] Delete the data directory and re-run setup to fix accuracy."
            )
        else:
            print(f"[verify]   folder names look correct (e.g. '{sample_class}') ✓")
    except Exception as e:
        print(f"[verify] FAILED: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["a", "b"],
        default="a",
        help="a = HuggingFace streaming (default)  |  b = Kaggle download",
    )
    args = parser.parse_args()

    # ── Validation data ───────────────────────────────────────────────────────
    if Path(VAL_DIR).exists() and any(Path(VAL_DIR).iterdir()):
        print(f"[setup] {VAL_DIR} already exists and is non-empty — skipping val download.")
    else:
        if args.method == "a":
            setup_method_a()
        else:
            setup_method_b()

    # ── Training data (Method A only — Kaggle method downloads val only) ───────
    if args.method == "a":
        if Path(TRAIN_DIR).exists() and any(Path(TRAIN_DIR).iterdir()):
            print(f"[setup] {TRAIN_DIR} already exists and is non-empty — skipping train download.")
        else:
            setup_train()
    else:
        print(
            "[setup] Method B downloads val data only.\n"
            f"[setup] To get training data, re-run with --method a or\n"
            f"[setup] manually place 10 000 train images under {TRAIN_DIR}/ in ImageFolder format."
        )

    verify()
