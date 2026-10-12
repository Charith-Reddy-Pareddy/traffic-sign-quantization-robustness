import argparse
import json
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from traffic_sign_quantization.calibration import create_calibration_loader
from traffic_sign_quantization.data import GTSRBDataset
from traffic_sign_quantization.models import create_model
from traffic_sign_quantization.paired_benchmark import (
    benchmark_checkpoint_pair,
    load_checkpoint_pair,
)
from traffic_sign_quantization.ptq import quantize_baseline
from traffic_sign_quantization.qat import convert_qat_model, prepare_qat_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare full-precision and quantized checkpoints on GTSRB"
    )
    parser.add_argument("--fp32-checkpoint", type=Path, required=True)
    parser.add_argument("--quantized-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--model",
        choices=("baseline_cnn", "mobilenet_v3_small"),
        default="baseline_cnn",
    )
    parser.add_argument("--quantization", choices=("qat", "ptq"), default="qat")
    parser.add_argument("--backend", default="qnnpack")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def _checkpoint_sizes(fp32: Path, quantized: Path) -> dict[str, int | float]:
    fp32_size = fp32.stat().st_size
    quantized_size = quantized.stat().st_size
    if fp32_size < 1 or quantized_size < 1:
        raise ValueError("checkpoint files must not be empty")
    return {
        "fp32_checkpoint_size_bytes": fp32_size,
        "quantized_checkpoint_size_bytes": quantized_size,
        "checkpoint_size_reduction": (fp32_size - quantized_size) / fp32_size,
    }


def _run_config(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {
        "dataset": "GTSRB",
        "split": "test",
        "model": args.model,
        "quantization": args.quantization,
        "backend": args.backend,
        "fp32_checkpoint": args.fp32_checkpoint.name,
        "quantized_checkpoint": args.quantized_checkpoint.name,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "runs": args.runs,
        "max_batches": args.max_batches,
    }
    if args.quantization == "ptq":
        config["calibration_samples"] = args.calibration_samples
    return config


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.warmup < 0 or args.runs < 2:
        parser.error("--warmup must be nonnegative and --runs must be at least 2")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    dataset = GTSRBDataset(root=args.data_root, split="test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    if args.max_batches is not None:
        loader = islice(loader, args.max_batches)
    first_images, _ = dataset[0]
    first_images = first_images.unsqueeze(0)
    fp32_model = create_model(args.model)
    quantized_model = create_model(args.model)

    if args.quantization == "qat":
        quantized_model = prepare_qat_model(quantized_model, args.backend)
        with torch.inference_mode():
            quantized_model(first_images)
        quantized_model = convert_qat_model(quantized_model, args.backend)
    else:
        calibration_loader = create_calibration_loader(
            root=args.data_root,
            sample_count=args.calibration_samples,
            batch_size=args.batch_size,
        )
        quantized_model = quantize_baseline(
            quantized_model, calibration_loader, backend=args.backend
        )

    fp32_model, quantized_model = load_checkpoint_pair(
        fp32_model,
        quantized_model,
        args.fp32_checkpoint,
        args.quantized_checkpoint,
        first_images,
    )
    report = benchmark_checkpoint_pair(
        fp32_model,
        quantized_model,
        loader,
        warmup=args.warmup,
        runs=args.runs,
    )
    report["config"] = _run_config(args)
    report["config"].update(
        _checkpoint_sizes(args.fp32_checkpoint, args.quantized_checkpoint)
    )
    result = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
