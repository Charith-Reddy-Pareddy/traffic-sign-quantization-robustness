from collections.abc import Mapping
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
    model_scores = report["accuracy"]
    class_scores = report["per_class"]
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
    return {
        "dataset": report.get("dataset"),
        "split": report.get("split"),
        "samples": report["samples"],
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
