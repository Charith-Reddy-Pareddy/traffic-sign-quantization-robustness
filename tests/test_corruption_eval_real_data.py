import os
from pathlib import Path

import pytest
from torch.utils.data import DataLoader, Subset

from traffic_sign_quantization.corruption_eval import evaluate_corruptions
from traffic_sign_quantization.data import GTSRBDataset
from traffic_sign_quantization.models import create_model


def test_corruption_evaluation_uses_real_gtsrb_images():
    root = Path(os.environ.get("GTSRB_DATA_ROOT", "data/raw"))
    if not (root / "Test.csv").is_file():
        pytest.skip("set GTSRB_DATA_ROOT to a real GTSRB dataset")

    dataset = GTSRBDataset(root=root, split="test")
    loader = DataLoader(Subset(dataset, range(2)), batch_size=2, shuffle=False)
    results = evaluate_corruptions(create_model("baseline_cnn"), loader, seed=7)

    assert set(results) == {"gaussian_noise", "brightness", "contrast"}
    assert all(set(levels) == {1, 2, 3, 4, 5} for levels in results.values())
    assert all(
        0 <= result.accuracy <= 1
        for levels in results.values()
        for result in levels.values()
    )
