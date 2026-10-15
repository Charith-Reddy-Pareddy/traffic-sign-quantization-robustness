import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from traffic_sign_quantization.data import GTSRBDataset
from traffic_sign_quantization.models import create_model
from traffic_sign_quantization.paired_benchmark import (
    benchmark_checkpoint_pair,
    load_checkpoint_pair,
)
from traffic_sign_quantization.qat import convert_qat_model, prepare_qat_model


def test_checkpoint_pair_reports_per_class_accuracy_on_real_gtsrb():
    data_root = Path(os.environ.get("GTSRB_DATA_ROOT", "data/raw"))
    fp32_path = Path(
        os.environ.get(
            "FP32_CHECKPOINT", "checkpoints/baseline_cnn_fp32_gtsrb_smoke.pt"
        )
    )
    quantized_path = Path(
        os.environ.get(
            "QUANTIZED_CHECKPOINT",
            "checkpoints/baseline_cnn_qat_gtsrb_smoke_converted.pt",
        )
    )
    required = (data_root / "Test.csv", fp32_path, quantized_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.skip(f"real GTSRB benchmark inputs are unavailable: {', '.join(missing)}")

    dataset = GTSRBDataset(root=data_root, split="test")
    if len(dataset) < 4:
        pytest.skip("real GTSRB test split has fewer than four images")
    loader = DataLoader(Subset(dataset, range(4)), batch_size=2, shuffle=False)
    example = torch.stack((dataset[0][0], dataset[1][0]))

    fp32_model = create_model("baseline_cnn")
    quantized_model = prepare_qat_model(create_model("baseline_cnn"))
    with torch.inference_mode():
        quantized_model(example)
    quantized_model = convert_qat_model(quantized_model)
    fp32_model, quantized_model = load_checkpoint_pair(
        fp32_model,
        quantized_model,
        fp32_path,
        quantized_path,
        example[:1],
    )

    report = benchmark_checkpoint_pair(
        fp32_model, quantized_model, loader, warmup=0, runs=2
    )

    assert report["samples"] == 4
    assert report["per_class_accuracy"]
    assert sum(
        metrics["samples"] for metrics in report["per_class_accuracy"].values()
    ) == report["samples"]
    fp32_weighted = sum(
        metrics["samples"] * metrics["fp32_accuracy"]
        for metrics in report["per_class_accuracy"].values()
    ) / report["samples"]
    quantized_weighted = sum(
        metrics["samples"] * metrics["quantized_accuracy"]
        for metrics in report["per_class_accuracy"].values()
    ) / report["samples"]
    assert fp32_weighted == pytest.approx(report["fp32_accuracy"])
    assert quantized_weighted == pytest.approx(report["quantized_accuracy"])
