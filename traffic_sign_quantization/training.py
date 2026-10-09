import random
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class EpochResult:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class EpochSummary:
    training: EpochResult
    validation: EpochResult


def _run_epoch(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochResult:
    device = torch.device(device)
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    grad_context = torch.enable_grad if training else torch.no_grad

    with grad_context():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.detach().item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += batch_size

    if total_items == 0:
        raise ValueError("data loader produced no batches")
    return EpochResult(
        loss=total_loss / total_items,
        accuracy=total_correct / total_items,
    )


def train_epoch(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> EpochResult:
    return _run_epoch(model, loader, device, optimizer)


def evaluate(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str = "cpu",
) -> EpochResult:
    return _run_epoch(model, loader, device)


def fit(
    model: torch.nn.Module,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    config: TrainConfig,
    device: torch.device | str = "cpu",
) -> tuple[EpochSummary, ...]:
    device = torch.device(device)
    set_seed(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history = []

    for _ in range(config.epochs):
        training = train_epoch(model, train_loader, optimizer, device)
        validation = evaluate(model, validation_loader, device)
        history.append(EpochSummary(training, validation))

    return tuple(history)
