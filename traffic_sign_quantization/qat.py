from collections.abc import Iterable
from copy import deepcopy
from itertools import chain

import torch
from torch import nn
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx

from traffic_sign_quantization.training import EpochSummary, TrainConfig, fit
from traffic_sign_quantization.data import GTSRBDataset


def _check_backend(backend: str) -> None:
    if backend not in torch.backends.quantized.supported_engines:
        raise ValueError(f"unsupported quantization backend: {backend}")
    torch.backends.quantized.engine = backend


def prepare_qat_model(
    model: nn.Module, backend: str = "qnnpack", images: torch.Tensor | None = None
) -> nn.Module:
    """Prepare a copy of a model for quantization-aware training."""
    _check_backend(backend)
    qat_model = deepcopy(model).cpu().train()
    if images is None:
        images = GTSRBDataset(split="train")[0][0].unsqueeze(0)
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("images must be a non-empty NCHW batch")
    example = images[:1].detach().cpu()
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
    batches = iter(train_loader)
    batch = next(batches, None)
    if batch is None:
        raise ValueError("train_loader must contain at least one batch")
    qat_model = prepare_qat_model(model, backend, images=batch[0])
    if batches is train_loader:
        train_loader = chain((batch,), batches)
    history = fit(qat_model, train_loader, validation_loader, config, device="cpu")
    return qat_model, history
