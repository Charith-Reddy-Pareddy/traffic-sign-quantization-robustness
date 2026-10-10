from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from traffic_sign_quantization.data import GTSRBDataset


def create_calibration_loader(
    root: str | Path = "data/raw",
    sample_count: int = 512,
    batch_size: int = 64,
    seed: int = 42,
) -> DataLoader:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    dataset = GTSRBDataset(root=root, split="train")
    if sample_count > len(dataset):
        raise ValueError("sample_count cannot exceed the training dataset size")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)
