from pathlib import Path

import torch


def save_weights(model: torch.nn.Module, path: str | Path) -> None:
    torch.save(model.state_dict(), Path(path))


def load_weights(
    model: torch.nn.Module,
    path: str | Path,
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    device = torch.device(device)
    model.to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def save_quantized_weights(model: torch.nn.Module, path: str | Path) -> None:
    """Save weights from a converted quantized model."""
    torch.save(model.state_dict(), Path(path))


def load_quantized_weights(
    model: torch.nn.Module,
    path: str | Path,
) -> torch.nn.Module:
    """Load weights into a matching, already converted model."""
    model.cpu().eval()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model
