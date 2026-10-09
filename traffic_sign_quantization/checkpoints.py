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
