from functools import partial

import pytest

import scripts.serve_dashboard as preview


def test_preview_serves_dashboard_from_loopback(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("dashboard", encoding="utf-8")
    calls = {}

    class FakeServer:
        def __init__(self, address, handler):
            calls["address"] = address
            calls["handler"] = handler

    monkeypatch.setattr(preview, "ThreadingHTTPServer", FakeServer)

    preview.create_server(8765, tmp_path)

    assert calls["address"] == ("127.0.0.1", 8765)
    assert isinstance(calls["handler"], partial)
    assert calls["handler"].keywords["directory"] == str(tmp_path.resolve())


def test_preview_requires_dashboard_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dashboard page not found"):
        preview.create_server(8765, tmp_path)


def test_preview_defaults_to_saved_dashboard_bundle(monkeypatch):
    calls = {}

    class FakeServer:
        def __init__(self, address, handler):
            calls["address"] = address
            calls["handler"] = handler

    monkeypatch.setattr(preview, "ThreadingHTTPServer", FakeServer)
    preview.create_server()

    root = preview.DASHBOARD_DIR.resolve()
    assert calls["address"] == ("127.0.0.1", 8000)
    assert calls["handler"].keywords["directory"] == str(root)
    assert (root / "index.html").is_file()
    assert len(list((root / "dashboard_reports").glob("*.html"))) == 2
