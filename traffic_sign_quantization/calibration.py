from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from traffic_sign_quantization.data import GTSRBDataset, create_data_loaders


def create_calibration_loader(
    root: str | Path = "data/raw",
    sample_count: int = 512,
    batch_size: int = 64,
    seed: int = 42,
    *,
    validation_fraction: float = 0.2,
    split_seed: int = 42,
) -> DataLoader:
    """Sample real training images while keeping validation tracks separate."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    dataset = GTSRBDataset(root=root, split="train")
    train_loader, _ = create_data_loaders(
        dataset,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )
    pool = train_loader.dataset.indices
    if sample_count > len(pool):
        raise ValueError("sample_count cannot exceed the training partition size")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pool), generator=generator)[:sample_count].tolist()
    indices = [pool[i] for i in order]
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)
