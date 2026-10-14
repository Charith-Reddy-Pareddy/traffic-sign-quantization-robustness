import argparse
import json
import pickle
import platform
from itertools import islice
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from traffic_sign_quantization.calibration import create_calibration_loader
from traffic_sign_quantization.data import GTSRBDataset, create_data_loaders
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
    parser.add_argument("--cpu-threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def _runtime_info() -> dict[str, str | int]:
    return {
        "device": "cpu",
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "system": platform.system(),
        "architecture": platform.machine(),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }


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
        "split": getattr(args, "split", "test"),
        "model": args.model,
        "quantization": args.quantization,
        "backend": args.backend,
        "fp32_checkpoint": args.fp32_checkpoint.name,
        "quantized_checkpoint": args.quantized_checkpoint.name,
        "batch_size": args.batch_size,
        "cpu_threads": getattr(args, "cpu_threads", torch.get_num_threads()),
        "warmup": args.warmup,
        "runs": args.runs,
        "max_batches": args.max_batches,
    }
    if args.quantization == "ptq":
        config["calibration_samples"] = args.calibration_samples
    config["split_seed"] = getattr(args, "split_seed", 42)
    config["validation_fraction"] = getattr(args, "validation_fraction", 0.2)
    return config


def _validate_inputs(args: argparse.Namespace) -> None:
    for option, path in (
        ("--fp32-checkpoint", args.fp32_checkpoint),
        ("--quantized-checkpoint", args.quantized_checkpoint),
    ):
        if not path.is_file():
            raise ValueError(f"{option}: checkpoint file not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"{option}: checkpoint file is empty: {path}")
    splits = ("Test", "Train") if args.quantization == "ptq" else ("Test",)
    if getattr(args, "split", "test") == "validation":
        splits = ("Train",)
    for split in splits:
        path = args.data_root.expanduser() / f"{split}.csv"
        if not path.is_file():
            raise ValueError(f"--data-root: required GTSRB CSV not found: {path}")
    if (
        args.backend not in torch.backends.quantized.supported_engines
        or args.backend == "none"
    ):
        raise ValueError(f"unsupported quantization backend: {args.backend}")
    if args.quantization == "ptq" and args.calibration_samples < 1:
        raise ValueError("--calibration-samples must be positive for PTQ")


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.warmup < 0 or args.runs < 2:
        parser.error("--warmup must be nonnegative and --runs must be at least 2")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")

    try:
        torch.set_num_threads(args.cpu_threads)
        _validate_inputs(args)
        _run_benchmark(args)
    except (
        OSError, ValueError, RuntimeError, pickle.UnpicklingError, EOFError
    ) as error:
        parser.error(str(error))


def _run_benchmark(args: argparse.Namespace) -> None:
    dataset = GTSRBDataset(
        root=args.data_root, split="train" if args.split == "validation" else "test"
    )
    if args.split == "validation":
        _, loader = create_data_loaders(
            dataset, batch_size=args.batch_size,
            validation_fraction=args.validation_fraction, seed=args.split_seed,
        )
        selected = loader.dataset
    else:
        selected = dataset
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    for i in range(len(selected)):
        index = selected.indices[i] if args.split == "validation" else i
        path, _ = dataset.records[index]
        if not path.is_file():
            raise ValueError(f"GTSRB image file not found: {path}")
    if args.max_batches is not None:
        loader = islice(loader, args.max_batches)
    first_images, _ = selected[0]
    first_images = first_images.unsqueeze(0)
    fp32_model = create_model(args.model)
    quantized_model = create_model(args.model)

    if args.quantization == "qat":
        quantized_model = prepare_qat_model(quantized_model, args.backend, images=first_images)
        with torch.inference_mode():
            quantized_model(first_images)
        quantized_model = convert_qat_model(quantized_model, args.backend)
    else:
        calibration_loader = create_calibration_loader(
            root=args.data_root,
            sample_count=args.calibration_samples,
            batch_size=args.batch_size,
            split_seed=args.split_seed,
            validation_fraction=args.validation_fraction,
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
    report["config"][f"{args.split}_split_samples"] = len(selected)
    report["config"].update(
        _checkpoint_sizes(args.fp32_checkpoint, args.quantized_checkpoint)
    )
    report["runtime"] = _runtime_info()
    result = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
