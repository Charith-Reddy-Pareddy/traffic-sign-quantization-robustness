import argparse
import hashlib
import json
from pathlib import Path

from traffic_sign_quantization.checkpoints import load_weights, save_weights
from traffic_sign_quantization.data import GTSRBDataset, create_data_loaders
from traffic_sign_quantization.models import create_model
from traffic_sign_quantization.qat import prepare_qat_model
from traffic_sign_quantization.training import set_seed


def prepare_baseline_qat(checkpoint: str | Path, root: str | Path = "data/raw", batch_size: int = 64):
    """Prepare baseline QAT using the source checkpoint's training split."""
    checkpoint = Path(checkpoint)
    metadata = json.loads(checkpoint.with_name(checkpoint.name + ".json").read_text())
    if metadata["model"] != "baseline_cnn" or metadata["split"] != "stratified_tracks":
        raise ValueError("requires baseline weights trained with separate validation tracks")
    seed = metadata["config"]["seed"]
    fraction = metadata["validation_fraction"]
    set_seed(seed)
    dataset = GTSRBDataset(root=root, split="train")
    train, val = create_data_loaders(dataset, batch_size, fraction, seed)
    if (len(train.dataset), len(val.dataset)) != (metadata["train_samples"], metadata["validation_samples"]):
        raise ValueError("dataset split counts differ from the source training report")
    model = load_weights(create_model("baseline_cnn"), checkpoint)
    images = train.dataset[0][0].unsqueeze(0)
    model = prepare_qat_model(model, images=images)
    report = {
        "model": "baseline_cnn", "backend": "qnnpack", "device": "cpu",
        "status": "prepared_not_trained", "source_checkpoint": checkpoint.name,
        "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_csv_sha256": hashlib.sha256((dataset.root / "Train.csv").read_bytes()).hexdigest(),
        "split": "stratified_tracks", "split_seed": seed,
        "validation_fraction": fraction, "batch_size": batch_size,
        "train_samples": len(train.dataset), "validation_samples": len(val.dataset),
        "tracing_input": "real_training_image",
    }
    return model, train, val, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare baseline QAT without starting training")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output.with_name(args.output.name + ".json")
    source_report = args.checkpoint.with_name(args.checkpoint.name + ".json")
    if {args.output.resolve(), report_path.resolve()} & {args.checkpoint.resolve(), source_report.resolve()}:
        parser.error("output must not overwrite the source checkpoint or its report")
    try:
        model, _, _, report = prepare_baseline_qat(args.checkpoint, args.data_root, args.batch_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_weights(model, args.output)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
