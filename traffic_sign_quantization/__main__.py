import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from traffic_sign_quantization.checkpoints import save_weights
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
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--output", type=Path)
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
        validation_fraction=args.validation_fraction,
        seed=config.seed,
    )
    if args.dry_run:
        print(json.dumps({
            "model": args.model,
            "device": args.device,
            "config": asdict(config),
            "validation_fraction": args.validation_fraction,
            "split": "stratified_tracks",
            "train_samples": len(train_loader.dataset),
            "validation_samples": len(val_loader.dataset),
        }, indent=2, sort_keys=True))
        return
    model = create_model(args.model)
    output = args.output or Path("artifacts") / f"{args.model}_seed{args.seed}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_epoch = 0

    def on_epoch(epoch, result):
        nonlocal best_accuracy, best_epoch
        print(
            f"epoch {epoch}: "
            f"train loss={result.training.loss:.4f} "
            f"train accuracy={result.training.accuracy:.4f} "
            f"validation loss={result.validation.loss:.4f} "
            f"validation accuracy={result.validation.accuracy:.4f}",
            flush=True,
        )
        if result.validation.accuracy > best_accuracy:
            save_weights(model, output)
            best_accuracy = result.validation.accuracy
            best_epoch = epoch

    history = fit(model, train_loader, val_loader, config, args.device, on_epoch)
    report = {
        "model": args.model,
        "device": args.device,
        "torch_version": str(torch.__version__),
        "config": asdict(config),
        "split": "stratified_tracks",
        "validation_fraction": args.validation_fraction,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "history": [asdict(result) for result in history],
    }
    output.with_name(output.name + ".json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved best weights from epoch {best_epoch} to {output}")


if __name__ == "__main__":
    main()
