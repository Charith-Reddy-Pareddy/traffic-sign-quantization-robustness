import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from traffic_sign_quantization.checkpoints import load_weights
from traffic_sign_quantization.data import GTSRBDataset
from traffic_sign_quantization.models import create_model
from traffic_sign_quantization.training import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a traffic sign classifier")
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
    result = evaluate(model, loader, device)
    print(f"test loss={result.loss:.4f} accuracy={result.accuracy:.4f}")


if __name__ == "__main__":
    main()
