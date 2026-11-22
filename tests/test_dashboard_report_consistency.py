import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(os.environ.get("DASHBOARD_DIR", "docs/dashboard")).resolve()


class ReportPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_overview = False
        self.in_cell = False
        self.in_subheading = False
        self.in_pre = False
        self.cells = None
        self.cell = []
        self.href = None
        self.rows = []
        self.subtitle = []
        self.provenance = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table" and attrs.get("id") == "report-overview":
            self.in_overview = True
        elif self.in_overview and tag == "tr":
            self.cells = []
            self.href = None
        elif self.in_overview and self.cells is not None and tag in {"td", "th"}:
            self.in_cell = True
            self.cell = []
        elif self.in_overview and self.in_cell and tag == "a":
            self.href = attrs.get("href")
        elif tag == "p" and attrs.get("class") == "subheading":
            self.in_subheading = True
        elif tag == "pre":
            self.in_pre = True

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)
        if self.in_subheading:
            self.subtitle.append(data)
        if self.in_pre:
            self.provenance.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.in_cell:
            self.cells.append("".join(self.cell).strip())
            self.in_cell = False
        elif tag == "tr" and self.cells is not None:
            if self.href:
                self.rows.append((self.href, self.cells))
            self.cells = None
        elif tag == "table" and self.in_overview:
            self.in_overview = False
        elif tag == "p" and self.in_subheading:
            self.in_subheading = False
        elif tag == "pre":
            self.in_pre = False


def read_page(path):
    page = ReportPage()
    page.feed(path.read_text(encoding="utf-8"))
    return page


def test_report_overview_matches_page_metadata_and_counts():
    overview = read_page(ROOT / "index.html")
    assert len(overview.rows) == 3

    for href, cells in overview.rows:
        target = (ROOT / href).resolve()
        target.relative_to(ROOT)
        report = read_page(target)
        match = re.match(
            r"(?P<dataset>.+?) (?P<split>test|validation) · (?P<samples>[\d,]+) real images",
            "".join(report.subtitle),
        )
        assert match, target.name
        assert cells[1] == match["dataset"]
        assert cells[2] == match["split"]
        assert cells[4] == match["samples"]

        provenance = json.loads("".join(report.provenance))
        assert cells[1] == provenance["dataset"]
        assert cells[2] == provenance["split"]
        selection = provenance.get("selection")
        if selection:
            assert cells[3].casefold() == selection["selection"].replace("_", " ")
            assert cells[4].replace(",", "") == str(selection["image_records"])
        else:
            assert cells[3] == "Not recorded"
