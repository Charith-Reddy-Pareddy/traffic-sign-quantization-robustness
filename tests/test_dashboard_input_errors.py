import sys

import pytest

from traffic_sign_quantization.dashboard import main


def _run(monkeypatch, capsys, report_path):
    monkeypatch.setattr(sys, "argv", ["dashboard", str(report_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    return capsys.readouterr().err


def test_missing_report_has_clear_cli_error(tmp_path, monkeypatch, capsys):
    message = _run(monkeypatch, capsys, tmp_path / "missing.json")

    assert "report file not found" in message
    assert "Traceback" not in message


def test_invalid_report_has_clear_cli_error(tmp_path, monkeypatch, capsys):
    report = tmp_path / "broken.json"
    report.write_text("{", encoding="utf-8")

    message = _run(monkeypatch, capsys, report)

    assert "report file is not valid JSON" in message
    assert "Traceback" not in message


def test_report_directory_with_no_valid_files_has_clear_cli_error(
    tmp_path, monkeypatch, capsys
):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "broken.json").write_text("{", encoding="utf-8")

    message = _run(monkeypatch, capsys, reports)

    assert "no valid comparison reports found" in message
    assert "Traceback" not in message
