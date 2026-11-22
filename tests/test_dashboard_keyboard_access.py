import os
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(os.environ.get("DASHBOARD_DIR", "docs/dashboard")).resolve()
PAGES = [ROOT / "index.html", *sorted((ROOT / "dashboard_reports").glob("*.html"))]


class Controls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.labels = set()
        self.fields = {}
        self.sort_buttons = []
        self.sort_headers = 0
        self.regions = []
        self.live_count = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "label" and attrs.get("for"):
            self.labels.add(attrs["for"])
        if tag in {"input", "select"} and attrs.get("id"):
            self.fields[attrs["id"]] = tag, attrs
        if tag == "button" and "data-sort-column" in attrs:
            self.sort_buttons.append(attrs)
        if tag == "th" and attrs.get("aria-sort") is not None:
            self.sort_headers += 1
        if attrs.get("role") == "region":
            self.regions.append(attrs)
        if attrs.get("id") == "class-count":
            self.live_count = attrs.get("aria-live")


def test_dashboard_controls_are_keyboard_accessible():
    assert len(PAGES) == 3
    for page in PAGES:
        parser = Controls()
        parser.feed(page.read_text(encoding="utf-8"))

        controls = {"class-search", "model-filter", "class-sort", "report-select"}
        assert controls <= parser.fields.keys(), page.name
        assert controls <= parser.labels, page.name
        assert parser.fields["class-search"][1].get("type") == "search"
        if page == ROOT / "index.html":
            assert len(parser.sort_buttons) == parser.sort_headers == 12
        assert parser.regions and all(region.get("tabindex") == "0" for region in parser.regions)
        assert parser.live_count == "polite"
