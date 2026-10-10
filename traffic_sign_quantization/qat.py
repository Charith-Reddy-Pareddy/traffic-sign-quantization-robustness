from collections.abc import Iterable
from copy import deepcopy

import torch
from torch import nn
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx

from traffic_sign_quantization.training import EpochSummary, TrainConfig, fit


def _check_backend(backend: str) -> None:
    if backend not in torch.backends.quantized.supported_engines:
        raise ValueError(f"unsupported quantization backend: {backend}")
    torch.backends.quantized.engine = backend


def prepare_qat_model(model: nn.Module, backend: str = "qnnpack") -> nn.Module:
    """Prepare a copy of a model for quantization-aware training."""
    _check_backend(backend)
    qat_model = deepcopy(model).cpu().train()
    example = torch.zeros(1, 3, 32, 32)
    return prepare_qat_fx(
        qat_model,
        get_default_qat_qconfig_mapping(backend),
        example_inputs=(example,),
    )


def convert_qat_model(model: nn.Module, backend: str = "qnnpack") -> nn.Module:
    """Convert a trained QAT model for quantized CPU inference."""
    _check_backend(backend)
    qat_model = deepcopy(model).cpu().eval()
    return convert_fx(qat_model).eval()


def fit_qat(
    model: nn.Module,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    config: TrainConfig,
    backend: str = "qnnpack",
) -> tuple[nn.Module, tuple[EpochSummary, ...]]:
    """Train a prepared model with the existing loop on CPU."""
    qat_model = prepare_qat_model(model, backend)
    history = fit(qat_model, train_loader, validation_loader, config, device="cpu")
    return qat_model, history
