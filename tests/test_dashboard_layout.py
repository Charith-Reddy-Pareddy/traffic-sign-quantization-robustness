import os
from pathlib import Path


ROOT = Path(os.environ.get("DASHBOARD_DIR", "docs/dashboard")).resolve()
PAGES = [ROOT / "index.html", *sorted((ROOT / "dashboard_reports").glob("*.html"))]


def test_saved_reports_keep_small_screen_layout_rules():
    assert len(PAGES) == 3
    for page in PAGES:
        source = page.read_text(encoding="utf-8")
        assert 'name="viewport" content="width=device-width, initial-scale=1"' in source
        assert ".table-wrap { overflow-x: auto;" in source
        assert "@media (max-width: 640px)" in source
        assert "@media (max-width: 480px)" in source
