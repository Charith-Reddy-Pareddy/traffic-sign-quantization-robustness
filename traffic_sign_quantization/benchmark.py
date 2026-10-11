import json
import math
import statistics
import time
from pathlib import Path

import torch


def benchmark_inference(
    model: torch.nn.Module,
    images: torch.Tensor,
    warmup: int = 10,
    runs: int = 100,
    device: torch.device | str = "cpu",
) -> dict[str, float | int]:
    """Measure repeated inference on a batch of dataset images."""
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("images must be a non-empty NCHW batch")
    if warmup < 0 or runs < 2:
        raise ValueError("warmup must be nonnegative and runs must be at least 2")

    device = torch.device(device)
    was_training = model.training
    model.to(device)
    model.eval()
    images = images.to(device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        with torch.inference_mode():
            for _ in range(warmup):
                model(images)

            times = []
            for _ in range(runs):
                synchronize()
                start = time.perf_counter()
                model(images)
                synchronize()
                times.append((time.perf_counter() - start) * 1000)
    finally:
        model.train(was_training)

    mean_ms = statistics.mean(times)
    return {
        "batch_size": int(images.shape[0]),
        "mean_ms": mean_ms,
        "std_ms": statistics.stdev(times),
        "p50_ms": statistics.median(times),
        "p95_ms": sorted(times)[math.ceil(0.95 * runs) - 1],
        "images_per_second": images.shape[0] * 1000 / mean_ms,
    }


def compare_models(
    full_precision: torch.nn.Module,
    quantized: torch.nn.Module,
    images: torch.Tensor,
    warmup: int = 10,
    runs: int = 100,
) -> dict[str, dict[str, float | int] | float]:
    """Compare loaded full-precision and quantized models on the same CPU batch."""
    full_stats = benchmark_inference(
        full_precision, images, warmup=warmup, runs=runs, device="cpu"
    )
    quantized_stats = benchmark_inference(
        quantized, images, warmup=warmup, runs=runs, device="cpu"
    )
    return {
        "full_precision": full_stats,
        "quantized": quantized_stats,
        "speedup": full_stats["mean_ms"] / quantized_stats["mean_ms"],
    }



def save_benchmark_report(
    report: dict[str, dict[str, float | int] | float],
    path: str | Path,
) -> None:
    """Write benchmark results as a readable JSON file."""
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
