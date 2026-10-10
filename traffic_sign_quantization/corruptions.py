import torch


def _check_input(images: torch.Tensor, severity: int) -> None:
    if images.ndim not in (3, 4) or not images.is_floating_point():
        raise ValueError("images must be a floating point CHW or NCHW tensor")
    if severity not in range(1, 6):
        raise ValueError("severity must be between 1 and 5")


def add_gaussian_noise(
    images: torch.Tensor,
    severity: int,
    seed: int = 42,
) -> torch.Tensor:
    _check_input(images, severity)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(images.shape, generator=generator, dtype=images.dtype)
    return (images + noise.to(images.device) * (0.02 * severity)).clamp(0, 1)


def adjust_brightness(images: torch.Tensor, severity: int) -> torch.Tensor:
    _check_input(images, severity)
    return (images + 0.04 * severity).clamp(0, 1)


def adjust_contrast(images: torch.Tensor, severity: int) -> torch.Tensor:
    _check_input(images, severity)
    dims = tuple(range(images.ndim - 3, images.ndim))
    mean = images.mean(dim=dims, keepdim=True)
    scale = 1 + 0.15 * severity
    return ((images - mean) * scale + mean).clamp(0, 1)
