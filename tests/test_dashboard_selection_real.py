from pathlib import Path

import pytest

from traffic_sign_quantization.dashboard import create_dashboard
from traffic_sign_quantization.dashboard_data import discover_reports


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

    reports = discover_reports(report_dir)
    pages = [
        output,
        *(
            views / f"{path.stem}.html"
            for path, _ in reports[1:]
        ),
    ]
    for (source, report), path in zip(reports, pages):
        rendered = path.read_text(encoding="utf-8")
        assert f"{report['samples']:,} real images" in rendered, source.name
        for model in report["models"].values():
            accuracy = model.get("accuracy")
            if accuracy is not None:
                assert f"<strong>{accuracy:.1%}</strong>" in rendered, source.name

    targets = [f"{path.stem}.html" for path, _ in reports[1:]]
    assert (views / "../dashboard.html").resolve() == output.resolve()
    for path in pages[1:]:
        detail = path.read_text(encoding="utf-8")
        picker = detail.split('<select id="report-select">', 1)[1].split(
            "</select>", 1
        )[0]
        assert picker.count("<option ") == len(reports)
        assert 'value="../dashboard.html"' in picker
        for target in targets:
            assert f'value="{target}"' in picker
            assert (views / target).is_file()
