import os
from pathlib import Path


ROOT = Path(os.environ.get("DASHBOARD_DIR", "docs/dashboard")).resolve()
PAGES = [ROOT / "index.html", *sorted((ROOT / "dashboard_reports").glob("*.html"))]


def test_saved_report_controls_stay_connected():
    controls = (
        'document.querySelector("#class-search")',
        'document.querySelector("#model-filter")',
        'document.querySelector("#class-sort")',
        'document.querySelector("#report-select")',
        'search.addEventListener("input", updateClasses)',
        'modelFilter.addEventListener("change", updateClasses)',
        'sortSelect.addEventListener("change", updateClasses)',
        'window.location.href = reportSelect.value;',
    )

    assert len(PAGES) == 3
    for page in PAGES:
        source = page.read_text(encoding="utf-8")
        assert all(control in source for control in controls), page.name
