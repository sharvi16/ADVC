"""
models/loader.py

Loads DeiT-S or DeiT-B at a specified compression level (fp32, int8, int4).
All parameters come from configs/base.yaml — never hardcode values here.

Usage:
    from models.loader import load_config, load_model
    cfg = load_config()
    model = load_model(model_name="deit_small", compression="int8", config=cfg)
"""

import torch
import torch.nn as nn
import timm
import yaml
from pathlib import Path
from typing import Literal

# Absolute path to configs/base.yaml — resolved relative to this file so that
# load_config() works regardless of the caller's working directory.
_DEFAULT_CFG = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


CompressionLevel = Literal["fp32", "int8", "int4"]
ModelName = Literal["deit_small"]


class LogitsWrapper(nn.Module):
    """Unwrap HuggingFace ImageClassifierOutput to a plain (N, C) logits tensor.

    timm models return plain tensors. HuggingFace INT8/INT4 models loaded via
    bitsandbytes return a dataclass with a .logits attribute.  This wrapper
    makes both interfaces identical for torchattacks and eval loops.

    Centralised here so adversarial_training.py, at_kd.py, and eval_phase1.py
    all share one canonical implementation — fixing a bug here fixes it everywhere.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out.logits if hasattr(out, "logits") else out


def load_config(config_path: str = None) -> dict:
    """
    Load the base YAML config.

    Args:
        config_path: Absolute or relative path to base.yaml.  Defaults to the
                     configs/base.yaml that lives next to this file, so callers
                     do not need to pass anything when running from any directory.

    Returns:
        config: Parsed config as a dictionary.
    """
    path = config_path if config_path is not None else str(_DEFAULT_CFG)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(
    model_name: ModelName,
    compression: CompressionLevel,
    config: dict,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Load a DeiT model at the specified compression level.

    Args:
        model_name:  "deit_small"
        compression: "fp32", "int8", or "int4"
        config:      Parsed base.yaml config dict
        device:      "cuda" or "cpu"

    Returns:
        model: torch.nn.Module in eval mode, moved to device.
    """
    model_cfg = config["models"][model_name]
    timm_name = model_cfg["timm_name"]

    if compression == "fp32":
        model = _load_fp32(timm_name, device)
    elif compression == "int8":
        model = _load_int8(timm_name, config, device)
    elif compression == "int4":
        model = _load_int4(timm_name, config, device)
    else:
        raise ValueError(
            f"Unknown compression level: {compression!r}. "
            "Choose from: fp32, int8, int4"
        )

    model.eval()
    return model


def _load_fp32(timm_name: str, device: str) -> torch.nn.Module:
    """Load full-precision model via timm."""
    model = timm.create_model(timm_name, pretrained=True)
    model = model.to(device)
    return model


def _load_int8(timm_name: str, config: dict, device: str) -> torch.nn.Module:
    """
    Load INT8 quantized model.
    Uses bitsandbytes if available, falls back to torch static quantization.
    """
    backend = config["compression"]["int8"]["backend"]

    if backend == "bitsandbytes":
        try:
            from transformers import AutoModelForImageClassification, BitsAndBytesConfig
            import bitsandbytes  # noqa: F401

            hf_name = _get_hf_name(timm_name, config)

            # load_in_8bit as a direct kwarg was removed in newer transformers;
            # it must be passed via BitsAndBytesConfig instead.
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            # Pin all layers to one device to avoid silent multi-GPU splits when
            # device_map="auto" is used with multi-GPU hosts.
            model = AutoModelForImageClassification.from_pretrained(
                hf_name,
                quantization_config=bnb_config,
                device_map={"" : device},
            )
            return model

        except (ImportError, Exception) as exc:
            # Do NOT fall back to torch static quantization — it runs on CPU only
            # and uses a different quantisation scheme (fbgemm vs bnb), making
            # results incomparable to bitsandbytes INT8 at 30-60x slower speed.
            raise RuntimeError(
                f"[loader] bitsandbytes INT8 failed: {exc}\n"
                "Install with: pip install bitsandbytes\n"
                "Ensure CUDA version matches: bitsandbytes requires CUDA >= 11.0\n"
                "Do NOT fall back to torch static quantization — results are not comparable."
            ) from exc

    raise ValueError(f"Unknown INT8 backend: {backend!r}")


def _load_int4(timm_name: str, config: dict, device: str) -> torch.nn.Module:
    """Load INT4 (NF4) quantized model via bitsandbytes."""
    from transformers import AutoModelForImageClassification, BitsAndBytesConfig

    int4_cfg = config["compression"]["int4"]
    compute_dtype = (
        torch.float16
        if int4_cfg["bnb_4bit_compute_dtype"] == "float16"
        else torch.bfloat16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=int4_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    hf_name = _get_hf_name(timm_name, config)

    # Pin all layers to one device.  device_map="auto" can split layers across
    # multiple GPUs on multi-GPU hosts, causing next(model.parameters()).device
    # to return only the first layer's device and breaking tensor routing.
    model = AutoModelForImageClassification.from_pretrained(
        hf_name,
        quantization_config=bnb_config,
        device_map={"" : device},
    )
    return model


def _get_hf_name(timm_name: str, config: dict) -> str:
    """Resolve a timm model name to its HuggingFace repo name via config."""
    for _, model_cfg in config["models"].items():
        if model_cfg["timm_name"] == timm_name:
            return model_cfg["hf_name"]
    raise ValueError(f"No HuggingFace name found for timm model: {timm_name!r}")


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Return model parameter size in MB.

    Args:
        model: Any torch.nn.Module.

    Returns:
        Size in megabytes, rounded to 2 decimal places.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(total_bytes / (1024 ** 2), 2)


def print_model_info(
    model: torch.nn.Module,
    model_name: str,
    compression: str,
) -> None:
    """Print a quick summary of the loaded model."""
    size_mb = get_model_size_mb(model)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    device = next(model.parameters()).device
    print(f"[loader] {model_name} @ {compression}")
    print(f"         Params : {num_params:.1f}M")
    print(f"         Size   : {size_mb} MB")
    print(f"         Device : {device}")


# Sanity check — run directly to verify everything loads
if __name__ == "__main__":
    cfg = load_config()
    print("=== Sanity check: DeiT-S at all compression levels ===\n")

    for level in ["fp32", "int8", "int4"]:
        try:
            model = load_model("deit_small", level, cfg)
            print_model_info(model, "deit_small", level)

            dummy = torch.randn(1, 3, 224, 224)
            if level == "fp32":
                dummy = dummy.cuda()
            with torch.no_grad():
                out = model(dummy)
            print(f"         Output : {out.logits.shape if hasattr(out, 'logits') else out.shape}\n")

        except Exception as e:
            print(f"[loader] {level} failed: {e}\n")
