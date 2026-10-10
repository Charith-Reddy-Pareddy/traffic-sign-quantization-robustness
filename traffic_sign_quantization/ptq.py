from collections.abc import Iterable
from copy import deepcopy

import torch
from torch import nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx


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
    example = torch.zeros(1, 3, 32, 32)
    prepared = prepare_fx(
        quantized_model,
        get_default_qconfig_mapping(backend),
        example_inputs=(example,),
    )

    sample_count = 0
    with torch.inference_mode():
        for batch in calibration_loader:
            images = batch[0].cpu()
            prepared(images)
            sample_count += len(images)

    if sample_count == 0:
        raise ValueError("calibration_loader must contain at least one batch")

    return convert_fx(prepared).eval()
