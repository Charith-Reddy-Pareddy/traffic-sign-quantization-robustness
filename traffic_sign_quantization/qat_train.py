import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from traffic_sign_quantization.checkpoints import load_weights, save_quantized_weights, save_weights
from traffic_sign_quantization.qat import convert_qat_model
from traffic_sign_quantization.qat_setup import prepare_baseline_qat
from traffic_sign_quantization.training import TrainConfig, evaluate, fit


def train_baseline_qat(checkpoint, output, root="data/raw", epochs=5, learning_rate=1e-4, batch_size=64):
    """Fine-tune QAT on real training images and retain the best validation epoch."""
    checkpoint, output = Path(checkpoint), Path(output)
    int8_path = output.with_name(output.stem + "_int8.pt")
    report_path = output.with_name(output.name + ".json")
    source_report = checkpoint.with_name(checkpoint.name + ".json")
    if {p.resolve() for p in (output, int8_path, report_path)} & {p.resolve() for p in (checkpoint, source_report)}:
        raise ValueError("outputs must not overwrite the source checkpoint or report")
    config = TrainConfig(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
    model, train, val, report = prepare_baseline_qat(checkpoint, root, batch_size)
    config = TrainConfig(**{**asdict(config), "seed": report["split_seed"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    report.update({
        "status": "training", "config": asdict(config), "history": [],
        "torch_version": str(torch.__version__),
        "cpu_threads": torch.get_num_threads(),
        "selection_metric": "prepared_model_validation_accuracy",
        "validation_observers_frozen": True,
        "best_epoch": 0, "best_validation_accuracy": -1.0,
        "prepared_checkpoint": output.name, "int8_checkpoint": int8_path.name,
    })

    def write_report():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def on_epoch(epoch, result):
        report["history"].append(asdict(result))
        if result.validation.accuracy > report["best_validation_accuracy"]:
            save_weights(model, output)
            report["best_epoch"] = epoch
            report["best_validation_accuracy"] = result.validation.accuracy
        write_report()
        print(
            f"epoch {epoch}/{epochs}: train loss={result.training.loss:.4f} "
            f"accuracy={result.training.accuracy:.4f}; "
            f"validation loss={result.validation.loss:.4f} "
            f"accuracy={result.validation.accuracy:.4f}", flush=True,
        )

    write_report()
    fit(model, train, val, config, on_epoch=on_epoch)
    load_weights(model, output)
    int8 = convert_qat_model(model)
    save_quantized_weights(int8, int8_path)
    report["converted_validation"] = asdict(evaluate(int8, val))
    report["status"] = "complete"
    write_report()
    print(f"saved best prepared epoch {report['best_epoch']} and converted INT8 weights", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Fine-tune baseline QAT on real GTSRB training images")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        train_baseline_qat(args.checkpoint, args.output, args.data_root, args.epochs, args.learning_rate, args.batch_size)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
