from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Subset

from traffic_sign_quantization.data import GTSRBDataset


def create_low_luminance_loader(
    root: str | Path = "data/raw",
    threshold: float = 0.15,
    batch_size: int = 64,
) -> tuple[DataLoader, dict]:
    """Select original test photographs by luminance, without altering pixels.

    Low luminance is an image measurement, not a verified weather or night label.
    Model inputs use the standard GTSRB resize and scaling only.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset = GTSRBDataset(root=root, split="test")
    indices, records = [], []
    for i, (path, label) in enumerate(dataset.records):
        if not 0 <= label < 43:
            raise ValueError(f"invalid GTSRB class ID: {label}")
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        luminance = float((rgb @ np.array([0.2126, 0.7152, 0.0722])).mean())
        if luminance <= threshold:
            indices.append(i)
            records.append({
                "path": str(path.relative_to(dataset.root)),
                "class_id": label,
                "mean_luminance": luminance,
            })
    if not indices:
        raise ValueError("no real test images meet the luminance threshold")
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    report = {
        "dataset": "GTSRB",
        "source": "https://benchmark.ini.rub.de/gtsrb_dataset.html",
        "split": "test",
        "selection": "low_measured_luminance",
        "threshold": threshold,
        "luminance_definition": "mean RGB weighted by 0.2126, 0.7152, 0.0722, scaled to 0-1",
        "weather_labels_available": False,
        "generated_corruptions": False,
        "test_samples": len(dataset),
        "selected_samples": len(indices),
        "represented_classes": sorted({row["class_id"] for row in records}),
        "images": records,
    }
    return loader, report
