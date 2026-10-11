from pathlib import Path

import torch
from torch import nn

from traffic_sign_quantization.checkpoints import load_quantized_weights, load_weights


def load_checkpoint_pair(
    fp32_model: nn.Module,
    quantized_model: nn.Module,
    fp32_path: str | Path,
    quantized_path: str | Path,
    images: torch.Tensor,
) -> tuple[nn.Module, nn.Module]:
    """Load a model pair and check that both produce logits for the same images."""
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("images must be a non-empty NCHW batch")

    try:
        fp32_model = load_weights(fp32_model, fp32_path, device="cpu").eval()
    except RuntimeError as error:
        raise ValueError("full-precision checkpoint does not match the model") from error

    try:
        quantized_model = load_quantized_weights(quantized_model, quantized_path).eval()
    except RuntimeError as error:
        raise ValueError("quantized checkpoint does not match the converted model") from error

    with torch.inference_mode():
        fp32_logits = fp32_model(images.cpu())
        quantized_logits = quantized_model(images.cpu())

    if (
        fp32_logits.ndim != 2
        or quantized_logits.ndim != 2
        or fp32_logits.shape != quantized_logits.shape
    ):
        raise ValueError("model pair must return matching batch and class dimensions")

    return fp32_model, quantized_model
