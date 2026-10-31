from pathlib import Path

import pytest

from traffic_sign_quantization.dashboard import create_dashboard


def test_dashboard_selects_a_saved_real_report(tmp_path):
    report_dir = Path("artifacts")
    if not (report_dir / "baseline_cnn_three_way_validation.json").is_file():
        pytest.skip("saved GTSRB comparison reports are unavailable")

    output = create_dashboard(report_dir, tmp_path / "dashboard.html")
    page = output.read_text(encoding="utf-8")
    views = tmp_path / "dashboard_reports"
    three_way = (views / "baseline_cnn_three_way_validation.html").read_text(
        encoding="utf-8"
    )
    ptq = (views / "baseline_cnn_tracksplit_ptq_comparison.html").read_text(
        encoding="utf-8"
    )

    assert 'id="report-select"' in page
    report_options = page.split('<select id="report-select">', 1)[1].split(
        "</select>", 1
    )[0]
    assert report_options.count("<option ") == 3
    assert 'value="dashboard.html" selected' in page
    assert "2,550 real images" in page
    assert "7,290 real images" in three_way
    assert 'value="qat"' in three_way
    assert 'value="qat"' not in ptq
    assert "training_csv_sha256" in three_way
