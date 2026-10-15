import os
from pathlib import Path

import pytest

from traffic_sign_quantization.dashboard import dashboard_html
from traffic_sign_quantization.dashboard_data import load_report


def test_dashboard_shows_saved_real_report():
    path = Path(
        os.environ.get(
            "THREE_WAY_REPORT",
            "artifacts/baseline_cnn_three_way_validation.json",
        )
    )
    if not path.is_file():
        pytest.skip(f"saved GTSRB report is unavailable: {path}")

    page = dashboard_html(load_report(path))

    assert "GTSRB model comparison" in page
    assert "7,290 real samples" in page
    assert "FP32" in page
    assert "PTQ" in page
    assert "QAT" in page
    assert "Report provenance" in page
