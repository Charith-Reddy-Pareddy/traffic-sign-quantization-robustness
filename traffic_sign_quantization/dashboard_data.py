from collections.abc import Mapping
import json
from math import isclose
from pathlib import Path
from typing import Any


_RESULT_FIELDS = {
    "samples",
    "accuracy",
    "prediction_agreement",
    "per_class",
    "coverage",
}


def normalize_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return comparison results in one shape while retaining report provenance."""
    if "accuracy" not in report and "per_class_accuracy" in report:
        report = _adapt_comparison_report(report)

    required = {
        "samples",
        "accuracy",
        "prediction_agreement",
        "per_class",
        "coverage",
    }
    missing = sorted(required - report.keys())
    if missing:
        raise ValueError(f"report is missing required fields: {', '.join(missing)}")

    samples = report["samples"]
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("report samples must be a positive integer")

    model_scores = report["accuracy"]
    class_scores = report["per_class"]
    if not isinstance(model_scores, Mapping) or not model_scores:
        raise ValueError("report accuracy must contain model scores")
    if not isinstance(class_scores, Mapping) or not class_scores:
        raise ValueError("report per_class must contain class scores")
    if not isinstance(report["prediction_agreement"], Mapping):
        raise ValueError("report prediction_agreement must be a mapping")
    if not isinstance(report["coverage"], Mapping):
        raise ValueError("report coverage must be a mapping")
    for pair, score in report["prediction_agreement"].items():
        if not _valid_score(score):
            raise ValueError(f"prediction agreement for {pair!r} must be between 0 and 1")

    for class_id, metrics in class_scores.items():
        if not isinstance(metrics, Mapping) or not isinstance(
            metrics.get("accuracy"), Mapping
        ):
            raise ValueError(f"per-class metrics for {class_id!r} are incomplete")
        class_samples = metrics.get("samples")
        if (
            not isinstance(class_samples, int)
            or isinstance(class_samples, bool)
            or class_samples < 0
        ):
            raise ValueError(
                f"sample count for class {class_id!r} must be nonnegative"
            )
        for name, score in metrics["accuracy"].items():
            if score is not None and not _valid_score(score):
                raise ValueError(
                    f"accuracy for model {name!r}, class {class_id!r} must be between 0 and 1"
                )

    for name, score in model_scores.items():
        if not _valid_score(score):
            raise ValueError(f"accuracy for {name!r} must be between 0 and 1")

    classes = [
        {
            "class_id": int(class_id),
            "samples": metrics["samples"],
            "accuracy": dict(metrics["accuracy"]),
        }
        for class_id, metrics in sorted(
            class_scores.items(), key=lambda item: int(item[0])
        )
    ]
    class_ids = [row["class_id"] for row in classes]
    if len(class_ids) != len(set(class_ids)):
        raise ValueError("report contains duplicate class ids")
    if sum(row["samples"] for row in classes) != samples:
        raise ValueError("per-class sample counts do not match report samples")
    for row in classes:
        if set(row["accuracy"]) != set(model_scores):
            raise ValueError(
                f"accuracy fields for class {row['class_id']} do not match report models"
            )

    represented = report["coverage"].get("represented_classes")
    observed_ids = {
        row["class_id"] for row in classes if row["samples"] > 0
    }
    if represented is not None and set(map(int, represented)) != observed_ids:
        raise ValueError("coverage classes do not match per-class results")

    for name, score in model_scores.items():
        rows = [
            row
            for row in classes
            if row["samples"] > 0 and _valid_score(row["accuracy"].get(name))
        ]
        class_samples = sum(row["samples"] for row in rows)
        if not class_samples:
            raise ValueError(f"per-class accuracy for {name!r} has no values")
        weighted = sum(
            row["samples"] * row["accuracy"][name] for row in rows
        ) / class_samples
        if class_samples == samples and not isclose(weighted, score, abs_tol=1e-9):
            raise ValueError(
                f"per-class accuracy for {name!r} does not match overall accuracy"
            )

    return {
        "dataset": report.get("dataset"),
        "split": report.get("split"),
        "samples": samples,
        "models": {
            name: {"accuracy": score} for name, score in model_scores.items()
        },
        "agreement": dict(report["prediction_agreement"]),
        "classes": classes,
        "coverage": dict(report["coverage"]),
        "provenance": {
            key: value for key, value in report.items() if key not in _RESULT_FIELDS
        },
    }


def load_report(path: str | Path) -> dict[str, Any]:
    """Load and check a saved comparison report for the dashboard data feed."""
    report_path = Path(path)
    try:
        with report_path.open(encoding="utf-8-sig") as file:
            report = json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"report file not found: {report_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"report file is not valid JSON: {report_path}") from exc

    if not isinstance(report, Mapping):
        raise ValueError(f"report must contain a JSON object: {report_path}")
    return normalize_report(report)


def discover_reports(directory: str | Path) -> list[tuple[Path, dict[str, Any]]]:
    """Return valid saved comparison reports in stable filename order."""
    report_dir = Path(directory)
    found = []
    for path in sorted(report_dir.glob("*.json")):
        try:
            report = load_report(path)
        except (FileNotFoundError, ValueError):
            continue
        found.append((path, report))
    return found


def _adapt_comparison_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Map saved FP32/PTQ comparison output into the dashboard report shape."""
    required = {
        "samples",
        "fp32_accuracy",
        "quantized_accuracy",
        "prediction_agreement",
        "per_class_accuracy",
    }
    missing = sorted(required - report.keys())
    if missing:
        return dict(report)

    rows = report["per_class_accuracy"]
    if not isinstance(rows, Mapping):
        return dict(report)
    classes = {
        class_id: {
            "samples": metrics.get("samples"),
            "accuracy": {
                "fp32": metrics.get("fp32_accuracy"),
                "ptq": metrics.get("quantized_accuracy"),
            },
        }
        for class_id, metrics in rows.items()
        if isinstance(metrics, Mapping)
    }
    selection = report.get("selection")
    if isinstance(selection, Mapping):
        selection = dict(selection)
        images = selection.pop("images", None)
        if images is not None:
            selection["image_records"] = len(images)
    config = report.get("config")
    if not isinstance(config, Mapping):
        config = {}

    result = {
        key: value
        for key, value in report.items()
        if key not in {
            "fp32_accuracy",
            "quantized_accuracy",
            "per_class_accuracy",
            "prediction_agreement",
            "selection",
        }
    }
    result.update(
        {
            "dataset": (
                selection.get("dataset") if isinstance(selection, Mapping) else None
            )
            or config.get("dataset"),
            "split": (
                selection.get("split") if isinstance(selection, Mapping) else None
            )
            or config.get("split"),
            "accuracy": {
                "fp32": report["fp32_accuracy"],
                "ptq": report["quantized_accuracy"],
            },
            "prediction_agreement": {
                "fp32_vs_ptq": report["prediction_agreement"],
            },
            "per_class": classes,
            "coverage": {
                "represented_classes": [
                    int(class_id)
                    for class_id, metrics in classes.items()
                    if metrics["samples"]
                ],
            },
        }
    )
    if selection is not None:
        result["selection"] = selection
    return result


def _valid_score(score: Any) -> bool:
    return (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and 0 <= score <= 1
    )
