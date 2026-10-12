from argparse import Namespace
from pathlib import Path

from traffic_sign_quantization.benchmark_pair_cli import _run_config


def _args(quantization: str) -> Namespace:
    return Namespace(
        model="baseline_cnn",
        quantization=quantization,
        backend="qnnpack",
        fp32_checkpoint=Path("/private/models/fp32.pt"),
        quantized_checkpoint=Path("/private/models/int8.pt"),
        batch_size=32,
        warmup=4,
        runs=20,
        max_batches=3,
        calibration_samples=128,
    )


def test_run_config_records_settings_without_local_paths():
    config = _run_config(_args("qat"))

    assert config["dataset"] == "GTSRB"
    assert config["split"] == "test"
    assert config["model"] == "baseline_cnn"
    assert config["quantization"] == "qat"
    assert config["fp32_checkpoint"] == "fp32.pt"
    assert config["quantized_checkpoint"] == "int8.pt"
    assert config["batch_size"] == 32
    assert config["max_batches"] == 3
    assert "calibration_samples" not in config
    assert "/private/" not in str(config)


def test_run_config_records_ptq_calibration_size():
    config = _run_config(_args("ptq"))

    assert config["calibration_samples"] == 128
