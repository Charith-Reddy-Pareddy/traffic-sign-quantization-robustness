from collections.abc import Iterable
from itertools import chain
from pathlib import Path

import torch
from torch import nn

from traffic_sign_quantization.benchmark import benchmark_inference
from traffic_sign_quantization.checkpoints import load_quantized_weights, load_weights


def load_checkpoint_pair(
    fp32_model: nn.Module,
    quantized_model: nn.Module,
    fp32_path: str | Path,
    quantized_path: str | Path,
    images: torch.Tensor,
) -> tuple[nn.Module, nn.Module]:
    """Load a model pair and check that both produce logits for the same images."""
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("images must be a non-empty NCHW batch")

    try:
        fp32_model = load_weights(fp32_model, fp32_path, device="cpu").eval()
    except RuntimeError as error:
        raise ValueError("full-precision checkpoint does not match the model") from error

    try:
        quantized_model = load_quantized_weights(quantized_model, quantized_path).eval()
    except RuntimeError as error:
        raise ValueError("quantized checkpoint does not match the converted model") from error

    with torch.inference_mode():
        fp32_logits = fp32_model(images.cpu())
        quantized_logits = quantized_model(images.cpu())

    if (
        fp32_logits.ndim != 2
        or quantized_logits.ndim != 2
        or fp32_logits.shape != quantized_logits.shape
    ):
        raise ValueError("model pair must return matching batch and class dimensions")

    return fp32_model, quantized_model


def benchmark_checkpoint_pair(
    fp32_model: nn.Module,
    quantized_model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    warmup: int = 10,
    runs: int = 100,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Compare accuracy, prediction agreement, and latency on one real-data loader."""
    batches = iter(loader)
    try:
        first_images, first_labels = next(batches)
    except StopIteration as error:
        raise ValueError("data loader must contain at least one batch") from error

    if first_images.ndim != 4 or first_images.shape[0] < 1:
        raise ValueError("loader images must be a non-empty NCHW batch")
    if first_labels.reshape(-1).numel() != first_images.shape[0]:
        raise ValueError("image and label batch sizes must match")

    fp32_timing = benchmark_inference(
        fp32_model, first_images, warmup=warmup, runs=runs, device=device
    )
    quantized_timing = benchmark_inference(
        quantized_model, first_images, warmup=warmup, runs=runs, device=device
    )

    device = torch.device(device)
    correct_fp32 = 0
    correct_quantized = 0
    agree = 0
    samples = 0
    class_counts: torch.Tensor | None = None
    fp32_class_correct: torch.Tensor | None = None
    quantized_class_correct: torch.Tensor | None = None
    fp32_mode = fp32_model.training
    quantized_mode = quantized_model.training
    fp32_model.eval()
    quantized_model.eval()
    try:
        with torch.inference_mode():
            for images, labels in chain(((first_images, first_labels),), batches):
                if images.ndim != 4 or images.shape[0] < 1:
                    raise ValueError("loader images must be non-empty NCHW batches")
                labels = labels.reshape(-1)
                if labels.numel() != images.shape[0]:
                    raise ValueError("image and label batch sizes must match")

                images = images.to(device)
                labels = labels.to(device)
                fp32_logits = fp32_model(images)
                quantized_logits = quantized_model(images)
                if (
                    fp32_logits.ndim != 2
                    or quantized_logits.ndim != 2
                    or fp32_logits.shape != quantized_logits.shape
                    or fp32_logits.shape[0] != labels.numel()
                ):
                    raise ValueError(
                        "model pair must return matching batch and class dimensions"
                    )
                if labels.is_floating_point() and not torch.equal(
                    labels, labels.round()
                ):
                    raise ValueError("labels must contain integer class IDs")
                if labels.numel() and (
                    labels.min() < 0 or labels.max() >= fp32_logits.shape[1]
                ):
                    raise ValueError("labels must be valid model class IDs")

                if class_counts is None:
                    class_counts = torch.zeros(fp32_logits.shape[1], dtype=torch.long)
                    fp32_class_correct = torch.zeros_like(class_counts)
                    quantized_class_correct = torch.zeros_like(class_counts)
                elif fp32_logits.shape[1] != class_counts.numel():
                    raise ValueError("model pair must keep the same class count")

                fp32_preds = fp32_logits.argmax(dim=1)
                quantized_preds = quantized_logits.argmax(dim=1)
                fp32_correct = fp32_preds == labels
                quantized_correct = quantized_preds == labels
                correct_fp32 += int(fp32_correct.sum().item())
                correct_quantized += int(quantized_correct.sum().item())
                agree += int((fp32_preds == quantized_preds).sum().item())
                samples += labels.numel()
                labels_cpu = labels.detach().to(device="cpu", dtype=torch.long)
                class_counts += torch.bincount(
                    labels_cpu, minlength=class_counts.numel()
                )
                fp32_class_correct += torch.bincount(
                    labels_cpu[fp32_correct.cpu()], minlength=class_counts.numel()
                )
                quantized_class_correct += torch.bincount(
                    labels_cpu[quantized_correct.cpu()], minlength=class_counts.numel()
                )
    finally:
        fp32_model.train(fp32_mode)
        quantized_model.train(quantized_mode)

    per_class_accuracy = {}
    for class_id, count in enumerate(class_counts.tolist()):
        if count:
            fp32_acc = int(fp32_class_correct[class_id]) / count
            quantized_acc = int(quantized_class_correct[class_id]) / count
            per_class_accuracy[class_id] = {
                "samples": count,
                "fp32_accuracy": fp32_acc,
                "quantized_accuracy": quantized_acc,
                "accuracy_delta": quantized_acc - fp32_acc,
            }

    return {
        "samples": samples,
        "fp32_accuracy": correct_fp32 / samples,
        "quantized_accuracy": correct_quantized / samples,
        "accuracy_delta": (correct_quantized - correct_fp32) / samples,
        "prediction_agreement": agree / samples,
        "per_class_accuracy": per_class_accuracy,
        "fp32_timing": fp32_timing,
        "quantized_timing": quantized_timing,
        "speedup": fp32_timing["mean_ms"] / quantized_timing["mean_ms"],
    }
