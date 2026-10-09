import csv
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


def create_data_loaders(
    dataset: Dataset,
    batch_size: int = 64,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")

    size = len(dataset)
    if size < 2:
        raise ValueError("dataset must contain at least two items")

    val_size = min(size - 1, max(1, int(size * validation_fraction)))
    train_size = size - val_size
    split_generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        dataset,
        (train_size, val_size),
        generator=split_generator,
    )
    train_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class GTSRBDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load GTSRB images and labels from the official CSV layout."""

    def __init__(
        self,
        root: str | Path = "data/raw",
        split: Literal["train", "test"] = "train",
        image_size: tuple[int, int] = (32, 32),
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if min(image_size) < 1:
            raise ValueError("image_size values must be positive")

        self.root = Path(root).expanduser().resolve()
        self.image_size = image_size
        self.transform = transform
        self.records: list[tuple[Path, int]] = []
        csv_path = self.root / f"{split.title()}.csv"

        with csv_path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            required = {"ClassId", "Path"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{csv_path} must contain ClassId and Path columns")

            for row_num, row in enumerate(reader, start=2):
                raw_label = row.get("ClassId")
                raw_path = row.get("Path")
                if not raw_label or not raw_label.strip():
                    raise ValueError(f"missing class label on row {row_num}")
                if not raw_path or not raw_path.strip():
                    raise ValueError(f"missing image path on row {row_num}")
                try:
                    label = int(raw_label)
                except ValueError as error:
                    raise ValueError(f"invalid class label on row {row_num}") from error

                rel_path = Path(raw_path)
                if rel_path.is_absolute():
                    raise ValueError(f"invalid image path on row {row_num}")

                img_path = (self.root / rel_path).resolve()
                if not img_path.is_relative_to(self.root):
                    raise ValueError(f"image path escapes dataset root on row {row_num}")
                self.records.append((img_path, label))

        if not self.records:
            raise ValueError(f"no image rows found in {csv_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.records[index]
        with Image.open(img_path) as image:
            image = image.convert("RGB").resize(
                self.image_size, Image.Resampling.BILINEAR
            )

        if self.transform is not None:
            image = self.transform(image)
        if isinstance(image, Image.Image):
            pixels = np.array(image, copy=True)
            image = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)
        if not isinstance(image, torch.Tensor):
            raise TypeError("transform must return a PIL image or torch tensor")
        return image, label
