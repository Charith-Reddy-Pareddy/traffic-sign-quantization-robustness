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

    assert "Traffic Sign Analytics · GTSRB" in page
    assert "7,290 real images" in page
    assert "Saved comparisons" in page
    assert "<strong>1</strong><span>One report shown</span>" in page
    assert "Model formats" in page
    assert "FP32" in page
    assert "PTQ" in page
    assert "QAT" in page
    assert "Not measured" in page
    assert "Not recorded" in page
    assert 'id="class-search"' in page
    assert 'id="model-filter"' in page
    assert 'id="evaluations"' in page
    assert "Report provenance" in page
