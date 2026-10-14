import argparse
import hashlib
import json
from pathlib import Path

import torch

from traffic_sign_quantization.calibration import create_calibration_loader
from traffic_sign_quantization.checkpoints import load_quantized_weights, load_weights
from traffic_sign_quantization.models import create_model
from traffic_sign_quantization.ptq import quantize_baseline
from traffic_sign_quantization.qat import convert_qat_model
from traffic_sign_quantization.qat_setup import prepare_baseline_qat


def compare_checkpoints(fp32_path, ptq_path, qat_path, root="data/raw", batch_size=64):
    """Evaluate three baseline checkpoints on exactly the same validation batches."""
    fp32_path, ptq_path, qat_path = map(Path, (fp32_path, ptq_path, qat_path))
    metadata = json.loads(qat_path.with_name(qat_path.name + ".json").read_text())
    source_hash = hashlib.sha256(fp32_path.read_bytes()).hexdigest()
    if metadata["status"] != "complete" or metadata["source_sha256"] != source_hash:
        raise ValueError("QAT must be complete and derived from the supplied FP32 checkpoint")
    qat, _, val, setup = prepare_baseline_qat(fp32_path, root, batch_size)
    if metadata["training_csv_sha256"] != setup["training_csv_sha256"]:
        raise ValueError("QAT training CSV differs from the current dataset")
    if (metadata["split_seed"], metadata["validation_fraction"]) != (setup["split_seed"], setup["validation_fraction"]):
        raise ValueError("QAT and FP32 validation splits differ")
    load_weights(qat, qat_path)
    int8_path = qat_path.with_name(qat_path.stem + "_int8.pt")
    qat = load_quantized_weights(convert_qat_model(qat), int8_path)
    fp32 = load_weights(create_model("baseline_cnn"), fp32_path).eval()
    calibration = create_calibration_loader(
        root, batch_size=batch_size, split_seed=setup["split_seed"],
        validation_fraction=setup["validation_fraction"],
    )
    ptq = load_quantized_weights(quantize_baseline(fp32, calibration), ptq_path)
    models = {"fp32": fp32, "ptq": ptq, "qat": qat}
    correct = dict.fromkeys(models, 0)
    samples = 0
    with torch.inference_mode():
        for images, labels in val:
            for name, model in models.items():
                logits = model(images)
                if logits.shape != (len(labels), 43) or not torch.isfinite(logits).all():
                    raise ValueError(f"{name} checkpoint returned invalid logits")
                correct[name] += int((logits.argmax(1) == labels).sum())
            samples += len(labels)
    report = {
        "samples": samples, "accuracy": {name: count / samples for name, count in correct.items()},
        "dataset": "GTSRB", "split": "validation", "model": "baseline_cnn",
        "split_seed": setup["split_seed"], "validation_fraction": setup["validation_fraction"],
        "batch_size": batch_size, "backend": "qnnpack", "cpu_threads": torch.get_num_threads(),
        "torch_version": str(torch.__version__),
        "training_csv_sha256": setup["training_csv_sha256"],
        "validation_indices_sha256": hashlib.sha256(json.dumps(val.dataset.indices).encode()).hexdigest(),
        "qat_extra_epochs": metadata["config"]["epochs"],
        "limitation": "QAT received extra fine-tuning; differences do not isolate quantization effects.",
        "checkpoints": [
            {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in (fp32_path, ptq_path, qat_path, int8_path)
        ],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare FP32, PTQ and QAT on real validation images")
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--ptq", type=Path, required=True)
    parser.add_argument("--qat", type=Path, required=True, help="trained prepared QAT checkpoint")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protected = [args.fp32, args.ptq, args.qat, args.qat.with_name(args.qat.stem + "_int8.pt")]
    protected += [p.with_name(p.name + ".json") for p in protected]
    if args.output.resolve() in {p.resolve() for p in protected}:
        parser.error("output must not overwrite checkpoints or their reports")
    try:
        report = compare_checkpoints(args.fp32, args.ptq, args.qat, args.data_root, args.batch_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
