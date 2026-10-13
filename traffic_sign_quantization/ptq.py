from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from traffic_sign_quantization.checkpoints import load_weights, save_quantized_weights
from traffic_sign_quantization.models import create_model


def quantize_baseline(
    model: nn.Module,
    calibration_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    backend: str = "qnnpack",
) -> nn.Module:
    """Calibrate and convert a CPU copy of a model with static quantization."""
    if backend not in torch.backends.quantized.supported_engines:
        raise ValueError(f"unsupported quantization backend: {backend}")

    torch.backends.quantized.engine = backend
    quantized_model = deepcopy(model).cpu().eval()
    batches = iter(calibration_loader)
    first = next(batches, None)
    if first is None or len(first[0]) == 0:
        raise ValueError("calibration_loader must contain at least one batch")
    example = first[0][:1].cpu()
    prepared = prepare_fx(
        quantized_model,
        get_default_qconfig_mapping(backend),
        example_inputs=(example,),
    )

    sample_count = 0
    with torch.inference_mode():
        prepared(first[0].cpu())
        sample_count += len(first[0])
        for batch in batches:
            images = batch[0].cpu()
            prepared(images)
            sample_count += len(images)

    if sample_count == 0:
        raise ValueError("calibration_loader must contain at least one batch")

    return convert_fx(prepared).eval()


def quantize_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    calibration_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    backend: str = "qnnpack",
) -> nn.Module:
    """Convert baseline weights and save a separate INT8 checkpoint."""
    if Path(checkpoint).resolve() == Path(output).resolve():
        raise ValueError("output must differ from the full-precision checkpoint")
    model = load_weights(create_model("baseline_cnn"), checkpoint).eval()
    model = quantize_baseline(model, calibration_loader, backend)
    save_quantized_weights(model, output)
    return model
