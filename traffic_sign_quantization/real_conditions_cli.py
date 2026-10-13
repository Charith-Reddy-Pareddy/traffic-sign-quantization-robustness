import argparse
import hashlib
import json
from pathlib import Path

from traffic_sign_quantization.benchmark_pair_cli import _checkpoint_sizes, _runtime_info
from traffic_sign_quantization.calibration import create_calibration_loader
from traffic_sign_quantization.checkpoints import load_quantized_weights, load_weights
from traffic_sign_quantization.models import create_model
from traffic_sign_quantization.paired_benchmark import benchmark_checkpoint_pair
from traffic_sign_quantization.ptq import quantize_baseline
from traffic_sign_quantization.real_conditions import create_low_luminance_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FP32/PTQ on original low-brightness GTSRB images")
    parser.add_argument("--fp32-checkpoint", type=Path, required=True)
    parser.add_argument("--quantized-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    try:
        if args.warmup < 0 or args.runs < 2:
            raise ValueError("warmup must be nonnegative and runs must be at least 2")
        paths = [args.fp32_checkpoint, args.quantized_checkpoint]
        metadata_path = args.fp32_checkpoint.with_name(args.fp32_checkpoint.name + ".json")
        if args.output.resolve() in {p.resolve() for p in paths + [metadata_path]}:
            raise ValueError("output must not overwrite a checkpoint or its training report")
        metadata = json.loads(metadata_path.read_text())
        if metadata["model"] != "baseline_cnn" or metadata["split"] != "stratified_tracks":
            raise ValueError("requires a baseline checkpoint trained with separate validation tracks")
        loader, selection = create_low_luminance_loader(
            args.data_root, args.threshold, args.batch_size
        )
        calibration = create_calibration_loader(
            args.data_root, batch_size=args.batch_size,
            split_seed=metadata["config"]["seed"],
            validation_fraction=metadata["validation_fraction"],
        )
        fp32 = load_weights(create_model("baseline_cnn"), args.fp32_checkpoint).eval()
        int8 = load_quantized_weights(
            quantize_baseline(fp32, calibration), args.quantized_checkpoint
        )
        report = benchmark_checkpoint_pair(fp32, int8, loader, args.warmup, args.runs)
        report["selection"] = selection
        report["runtime"] = _runtime_info()
        report["config"] = {
            "model": "baseline_cnn", "quantization": "ptq", "backend": "qnnpack",
            "batch_size": args.batch_size, "warmup": args.warmup, "runs": args.runs,
            "calibration_samples": len(calibration.dataset),
            "calibration_seed": 42, "split_seed": metadata["config"]["seed"],
            "validation_fraction": metadata["validation_fraction"],
            "latency_scope": "first selected batch; model inference only",
            **_checkpoint_sizes(*paths),
        }
        report["provenance"] = {
            "checkpoints": [
                {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in paths
            ],
            "test_csv_sha256": hashlib.sha256(
                (loader.dataset.dataset.root / "Test.csv").read_bytes()
            ).hexdigest(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: report[k] for k in (
            "samples", "fp32_accuracy", "quantized_accuracy", "accuracy_delta", "speedup"
        )}, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
