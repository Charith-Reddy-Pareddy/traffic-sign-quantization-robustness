import os
from pathlib import Path

import pytest

from traffic_sign_quantization.dashboard import _delta, _meter, _percent, dashboard_html
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

    report = load_report(path)
    page = dashboard_html(report)

    assert "Traffic Sign Analytics · GTSRB" in page
    assert "7,290 real images" in page
    assert "Images evaluated" in page
    assert f'<strong>{report["samples"]:,}</strong>' in page
    represented = report["coverage"]["represented_classes"]
    assert "Classes represented" in page
    assert f'<strong>{len(represented)}</strong>' in page
    assert f'of {len(report["classes"])} listed classes' in page
    assert "Saved reports" in page
    assert "<strong>1</strong><span>available for selection</span>" in page
    assert "FP32" in page
    assert "PTQ" in page
    assert "QAT" in page
    assert "Accuracy bars use the same 0–100% scale" in page
    for name, metrics in report["models"].items():
        score = metrics["accuracy"]
        assert f'value="{score:.6f}"' in page
        if name == "fp32":
            assert "reference" in page
        else:
            delta = (score - report["models"]["fp32"]["accuracy"]) * 100
            assert f"{delta:+.1f} pp vs FP32" in page
    assert "Not measured" in page
    assert "Not recorded" in page
    assert 'id="class-search"' in page
    assert 'id="model-filter"' in page
    assert 'id="evaluations"' in page
    assert "Report provenance" in page
    assert "@media (max-width: 640px)" in page
    assert "position: sticky" in page
    assert 'id="class-empty" hidden' in page
    assert 'tbody tr[data-search]' in page

    options = [("first.html", "First"), ("second.html", "Second")]
    multi_page = dashboard_html(report, options, "first.html")
    assert "<strong>2</strong><span>available for selection</span>" in multi_page


def test_dashboard_handles_missing_score_values():
    assert _percent(None) == "—"
    assert _meter("PTQ", None) == (
        '<span class="metric-unavailable">Accuracy unavailable</span>'
    )
    assert _delta(None, 0.9, "ptq") == "score unavailable"


def test_dashboard_shows_saved_real_benchmark_values():
    path = Path("artifacts/baseline_cnn_tracksplit_ptq_comparison.json")
    if not path.is_file():
        pytest.skip(f"saved GTSRB benchmark report is unavailable: {path}")

    report = load_report(path)
    page = dashboard_html(report)
    fp32 = report["provenance"]["fp32_timing"]
    ptq = report["provenance"]["quantized_timing"]
    config = report["provenance"]["config"]

    assert 'id="benchmark"' in page
    assert "Saved measurements" in page
    assert f"{fp32['mean_ms']:.2f} ms" in page
    assert f"{ptq['p95_ms']:.2f} ms" in page
    assert f"{ptq['images_per_second']:,.0f} images/s" in page
    assert f"{config['fp32_checkpoint_size_bytes'] / 1024:,.1f} KiB" in page
    assert f"{config['quantized_checkpoint_size_bytes'] / 1024:,.1f} KiB" in page
    assert f"{report['provenance']['speedup']:.2f}×" in page
    assert f"{config['checkpoint_size_reduction'] * 100:.1f}%" in page
    assert "does not rerun inference" in page
