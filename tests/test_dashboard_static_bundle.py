import os
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(os.environ.get("DASHBOARD_DIR", "docs/dashboard")).resolve()


class ReportChoices(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_report_select = False
        self.choices = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "select":
            self.in_report_select = attrs.get("id") == "report-select"
        elif tag == "option" and self.in_report_select:
            self.choices.append((attrs.get("value"), "selected" in attrs))

    def handle_endtag(self, tag):
        if tag == "select":
            self.in_report_select = False


def test_dashboard_report_choices_point_to_each_static_page():
    pages = [ROOT / "index.html", *sorted((ROOT / "dashboard_reports").glob("*.html"))]

    assert len(pages) == 3
    for page in pages:
        parser = ReportChoices()
        parser.feed(page.read_text(encoding="utf-8"))
        choices = [(path, selected) for path, selected in parser.choices if path]

        assert len(choices) == len(pages)
        selected = [path for path, active in choices if active]
        assert selected == [Path(os.path.relpath(page, page.parent)).as_posix()]
        for path, _ in choices:
            target = (page.parent / path).resolve()
            target.relative_to(ROOT)
            assert target.is_file()

    assert "2,550 real images" in pages[0].read_text(encoding="utf-8")
    assert all("7,290 real images" in page.read_text(encoding="utf-8") for page in pages[1:])
