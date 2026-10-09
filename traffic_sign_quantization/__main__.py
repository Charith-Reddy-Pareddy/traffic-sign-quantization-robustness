import argparse
from pathlib import Path

import torch

from traffic_sign_quantization.data import GTSRBDataset, create_data_loaders
from traffic_sign_quantization.models import create_model
from traffic_sign_quantization.training import TrainConfig, fit, set_seed


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a traffic sign classifier")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--model",
        choices=("baseline_cnn", "mobilenet_v3_small"),
        default="baseline_cnn",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=_default_device())
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    set_seed(config.seed)
    dataset = GTSRBDataset(root=args.data_root, split="train")
    train_loader, val_loader = create_data_loaders(
        dataset,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    model = create_model(args.model)
    history = fit(model, train_loader, val_loader, config, args.device)

    for epoch, result in enumerate(history, start=1):
        print(
            f"epoch {epoch}: "
            f"train loss={result.training.loss:.4f} "
            f"train accuracy={result.training.accuracy:.4f} "
            f"validation loss={result.validation.loss:.4f} "
            f"validation accuracy={result.validation.accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
