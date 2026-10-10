import argparse
from collections.abc import Iterable
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from traffic_sign_quantization.checkpoints import load_weights
from traffic_sign_quantization.corruptions import (
    add_gaussian_noise,
    adjust_brightness,
    adjust_contrast,
)
from traffic_sign_quantization.data import GTSRBDataset
from traffic_sign_quantization.models import create_model
from traffic_sign_quantization.training import EpochResult, evaluate


def evaluate_corruptions(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> dict[str, dict[int, EpochResult]]:
    device = torch.device(device)
    model.to(device)
    transforms = {
        "gaussian_noise": add_gaussian_noise,
        "brightness": adjust_brightness,
        "contrast": adjust_contrast,
    }
    results = {}

    for name, transform in transforms.items():
        results[name] = {}
        for severity in range(1, 6):
            def batches():
                for batch_idx, (images, labels) in enumerate(loader):
                    if name == "gaussian_noise":
                        images = transform(images, severity, seed + batch_idx)
                    else:
                        images = transform(images, severity)
                    yield images, labels

            results[name][severity] = evaluate(model, batches(), device)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure traffic sign accuracy under image corruptions"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("baseline_cnn", "mobilenet_v3_small"),
        default="baseline_cnn",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    device = torch.device(args.device)
    dataset = GTSRBDataset(root=args.data_root, split="test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = load_weights(create_model(args.model), args.weights, device)
    results = evaluate_corruptions(model, loader, device)

    for name, severities in results.items():
        for severity, result in severities.items():
            print(
                f"{name} severity={severity} "
                f"loss={result.loss:.4f} accuracy={result.accuracy:.4f}"
            )


if __name__ == "__main__":
    main()
