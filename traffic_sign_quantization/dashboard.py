import argparse
from collections.abc import Mapping
from html import escape
import json
from math import isfinite
from pathlib import Path
from typing import Any
from urllib.parse import quote

from traffic_sign_quantization.dashboard_data import discover_reports, load_report


def dashboard_html(
    report: Mapping[str, Any],
    report_options: list[tuple[str, str]] | None = None,
    selected_report: str | None = None,
    report_summaries: list[tuple[str, str, Mapping[str, Any]]] | None = None,
) -> str:
    """Render a saved model comparison with the fields its report provides."""
    models = report["models"]
    model_names = {name: name.upper() for name in models}
    dataset = escape(str(report.get("dataset") or "Dataset"))
    split = escape(str(report.get("split") or "unspecified split"))
    sample_count = report["samples"]
    coverage = report["coverage"]
    represented = coverage.get("represented_classes")
    represented_count = len(represented) if represented is not None else sum(
        item["samples"] > 0 for item in report["classes"]
    )
    report_count = len(report_options) if report_options else 1
    model_name = escape(str(report["provenance"].get("model") or "Not recorded"))
    reference = models.get("fp32", {}).get("accuracy")
    model_cards = "".join(
        f'<article class="model-card"><span>{escape(label)}</span>'
        f'<strong>{_percent(metrics.get("accuracy"))}</strong>'
        f'{_meter(label, metrics.get("accuracy"))}'
        f'<span class="card-note">Overall accuracy · '
        f'{_delta(metrics.get("accuracy"), reference, name)}</span></article>'
        for name, metrics in models.items()
        for label in (model_names[name],)
    )
    agreements = "".join(
        f'<li><span>{escape(name.replace("_", " → ").upper())}</span>'
        f'<strong>{_percent(value)}</strong></li>'
        for name, value in report["agreement"].items()
    ) or "<li>No prediction-agreement values were recorded.</li>"
    model_headers = "".join(
        f'<th data-model="{escape(name)}" scope="col">{escape(label)}</th>'
        for name, label in model_names.items()
    )
    class_rows = "".join(
        '<tr data-search="class '
        f'{item["class_id"]}" {_class_data_attributes(item)}>'
        f'<th scope="row">{item["class_id"]}</th>'
        f'<td>{item["samples"]:,}</td>'
        + "".join(
            f'<td data-model="{escape(name)}">'
            f'{_percent(item["accuracy"].get(name))}</td>'
            for name in models
        )
        + "</tr>"
        for item in report["classes"]
    )
    class_chart = "".join(
        _class_comparison(item, models, model_names)
        for item in report["classes"]
    )
    empty_class_row = (
        f'<tr id="class-empty" hidden><td colspan="{len(models) + 2}">'
        "No classes match your search.</td></tr>"
    )
    evaluation_rows = "".join(
        "<tr>"
        f"<td>{dataset} {split}</td>"
        f"<td>{model_name}</td>"
        f"<td>{escape(label)}</td>"
        f"<td>{_percent(metrics.get('accuracy'))}</td>"
        "<td>Not measured</td>"
        "<td>Not recorded</td>"
        "</tr>"
        for name, metrics in models.items()
        for label in (model_names[name],)
    )
    provenance = escape(
        json.dumps(report["provenance"], indent=2, sort_keys=True, default=str)
    )
    report_picker = ""
    if report_options and len(report_options) > 1:
        options = "".join(
            f'<option value="{escape(href, quote=True)}"'
            f'{" selected" if href == selected_report else ""}>'
            f"{escape(label)}</option>"
            for href, label in report_options
        )
        report_picker = (
            '<section class="report-picker" aria-label="Saved reports">'
            '<label for="report-select">Choose a saved report'
            f'<select id="report-select">{options}</select></label></section>'
        )
    benchmark_panel = _benchmark_section(report)
    benchmark_nav = '<a href="#benchmark">Benchmark</a>'
    report_nav = '<a href="#saved-reports">Reports</a>' if report_summaries else ""
    report_summary = _saved_report_section(report_summaries, selected_report)

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Traffic Sign Analytics · {dataset}</title>
    <style>
      :root {{ color-scheme: light; font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #202a35; background: #f5f7fa; }}
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; min-width: 320px; }}
      a {{ color: inherit; text-decoration: none; }}
      .topbar {{ display: flex; align-items: center; justify-content: space-between; min-height: 68px; padding: 0 max(6vw, 20px); background: #fff; border-bottom: 1px solid #e5e9ef; }}
      .brand, nav, .status {{ display: flex; align-items: center; }}
      .brand {{ gap: 11px; font-weight: 650; }}
      .brand-mark {{ display: grid; width: 34px; height: 34px; place-items: center; border-radius: 10px; background: #e7f0ff; color: #275daf; font-size: 12px; font-weight: 750; }}
      nav {{ align-self: stretch; gap: 30px; color: #667383; font-size: 13px; }}
      nav a {{ display: grid; align-items: center; border-bottom: 2px solid transparent; }}
      nav a:hover, nav a:focus {{ border-bottom-color: #356ac3; color: #244f91; }}
      main {{ width: min(1120px, 88vw); margin: 48px auto 72px; }}
      .intro {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; margin-bottom: 28px; }}
      .eyebrow {{ margin: 0 0 10px; color: #6e7d90; font-size: 11px; font-weight: 700; letter-spacing: .11em; }}
      h1, h2, h3, p {{ margin-top: 0; }}
      h1 {{ margin-bottom: 8px; font-size: clamp(27px, 4vw, 34px); letter-spacing: -.035em; }}
      .subheading, .panel-heading p {{ margin-bottom: 0; color: #748091; font-size: 14px; }}
      .status {{ gap: 8px; padding: 8px 11px; border: 1px solid #e1e7ee; border-radius: 20px; background: #fff; color: #657385; font-size: 12px; white-space: nowrap; }}
      .status-dot {{ width: 7px; height: 7px; border-radius: 50%; background: #357b54; }}
      .report-picker {{ display: flex; margin: 0 0 20px; padding: 15px 18px; border: 1px solid #e4e9ef; border-radius: 10px; background: #fff; }}
      .report-picker select {{ min-width: min(360px, 80vw); }}
      .summary {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-bottom: 20px; }}
      .summary-card, .panel {{ border: 1px solid #e4e9ef; border-radius: 12px; background: #fff; box-shadow: 0 2px 7px rgb(25 39 56 / 3%); }}
      .summary-card {{ padding: 19px 20px; }}
      .summary-card p {{ margin-bottom: 13px; color: #657385; font-size: 13px; }}
      .summary-card strong {{ display: block; margin-bottom: 5px; font-size: 27px; font-weight: 650; }}
      .summary-card span {{ color: #8994a2; font-size: 12px; }}
      .panel {{ margin-top: 20px; overflow: hidden; }}
      .panel-heading {{ display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 20px 22px; border-bottom: 1px solid #edf0f4; }}
      .panel-heading h2 {{ margin-bottom: 5px; font-size: 16px; }}
      .panel-heading p {{ font-size: 12px; }}
      .model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 14px; padding: 20px 22px; }}
      .model-card {{ display: grid; gap: 4px; padding: 17px; border: 1px solid #e4e9ef; border-radius: 10px; }}
      .model-card > span:first-child {{ color: #657385; font-size: 12px; font-weight: 650; }}
      .model-card strong {{ font-size: 25px; }}
      .model-card meter {{ width: 100%; height: 10px; accent-color: #356ac3; }}
      .model-card .card-note {{ color: #8994a2; font-size: 11px; }}
      .agreement {{ display: flex; flex-wrap: wrap; gap: 12px 28px; margin: 0; padding: 0 22px 20px; list-style: none; }}
      .agreement li {{ display: flex; gap: 10px; color: #657385; font-size: 13px; }}
      .agreement strong {{ color: #202a35; }}
      .table-tools {{ display: flex; align-items: end; flex-wrap: wrap; gap: 12px; padding: 16px 22px 8px; }}
      .class-chart {{ display: grid; gap: 10px; padding: 14px 22px 20px; }}
      .class-chart-row {{ display: grid; grid-template-columns: 120px minmax(0, 1fr); gap: 16px; align-items: start; padding: 10px 0; border-bottom: 1px solid #edf0f4; }}
      .class-chart-label {{ display: grid; gap: 2px; color: #657385; font-size: 12px; }}
      .class-chart-label strong {{ color: #202a35; }}
      .class-chart-models {{ display: grid; gap: 7px; }}
      .class-chart-model {{ display: grid; grid-template-columns: 44px minmax(50px, 1fr) 48px minmax(90px, auto); gap: 9px; align-items: center; color: #657385; font-size: 11px; }}
      .class-chart-model strong {{ color: #202a35; font-variant-numeric: tabular-nums; }}
      .bar-track {{ height: 9px; overflow: hidden; border-radius: 6px; background: #edf1f6; }}
      .bar-fill {{ display: block; height: 100%; width: var(--bar-width); border-radius: inherit; background: #356ac3; }}
      .class-chart-model[data-model="ptq"] .bar-fill {{ background: #d18432; }}
      .class-chart-model[data-model="qat"] .bar-fill {{ background: #357b54; }}
      .class-chart-delta {{ color: #748091; font-size: 10px; }}
      .class-chart-note {{ padding: 0 22px; color: #748091; font-size: 11px; }}
      label {{ display: grid; gap: 5px; color: #657385; font-size: 12px; }}
      input, select {{ min-height: 38px; padding: 7px 10px; border: 1px solid #cbd4df; border-radius: 7px; background: #fff; color: #202a35; font: inherit; }}
      input {{ width: min(270px, 75vw); }}
      input:focus, select:focus, a:focus {{ outline: 3px solid #9cc0f4; outline-offset: 2px; }}
      .results-count {{ margin: 0 0 8px auto; color: #748091; font-size: 12px; }}
      .table-wrap {{ overflow-x: auto; overscroll-behavior-x: contain; -webkit-overflow-scrolling: touch; }}
      .table-wrap[tabindex="0"]:focus-visible {{ outline: 3px solid #9cc0f4; outline-offset: 2px; }}
      .visually-hidden {{ position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }}
      table {{ width: 100%; border-collapse: collapse; text-align: left; white-space: nowrap; }}
      th, td {{ padding: 12px 18px; border-bottom: 1px solid #edf0f4; font-size: 12px; }}
      th {{ background: #fafbfd; color: #768293; font-weight: 600; }}
      .sort-button {{ padding: 0; border: 0; background: none; color: inherit; font: inherit; font-weight: inherit; text-align: left; cursor: pointer; }}
      .sort-button:hover, .sort-button:focus {{ color: #244f91; }}
      th[aria-sort="ascending"] .sort-button::after {{ content: " ↑"; }}
      th[aria-sort="descending"] .sort-button::after {{ content: " ↓"; }}
      tbody th {{ color: #202a35; }}
      #saved-reports a[aria-current="page"] {{ color: #244f91; font-weight: 650; }}
      #class-table tbody th {{ position: sticky; left: 0; z-index: 1; background: #fff; }}
      #class-table thead th:first-child {{ position: sticky; left: 0; z-index: 2; }}
      .note {{ padding: 0 22px 18px; color: #748091; font-size: 12px; }}
      details {{ margin: 20px 0; padding: 16px 20px; border: 1px solid #e4e9ef; border-radius: 12px; background: #fff; }}
      summary {{ cursor: pointer; font-weight: 600; }}
      pre {{ overflow: auto; padding: 14px; border-radius: 8px; background: #f3f6fa; font-size: 12px; }}
      [hidden] {{ display: none !important; }}
      @media (max-width: 640px) {{ .topbar {{ align-items: flex-start; flex-direction: column; gap: 8px; padding-top: 12px; }} nav {{ min-height: 42px; gap: 20px; }} main {{ width: 92vw; margin-top: 30px; }} .intro {{ align-items: flex-start; flex-direction: column; }} .summary {{ grid-template-columns: 1fr; gap: 10px; }} .summary-card {{ padding: 15px 17px; }} .panel-heading {{ align-items: flex-start; flex-direction: column; }} .results-count {{ width: 100%; margin-left: 0; }} }}
      @media (max-width: 480px) {{ .class-chart-row {{ grid-template-columns: 1fr; gap: 8px; }} .class-chart-model {{ grid-template-columns: 38px minmax(35px, 1fr) 42px; }} .class-chart-delta {{ grid-column: 2 / -1; }} }}
    </style>
  </head>
  <body>
    <header class="topbar">
      <a class="brand" href="#overview" aria-label="Traffic Sign Analytics home">
        <span class="brand-mark" aria-hidden="true">TS</span>
        <span>Traffic Sign Analytics</span>
      </a>
      <nav aria-label="Main navigation">
        <a href="#overview">Overview</a>
        <a href="#models">Models</a>
        {benchmark_nav}
        {report_nav}
        <a href="#evaluations">Evaluations</a>
      </nav>
    </header>
    <main id="overview">
      <section class="intro" aria-labelledby="page-title">
        <div>
          <p class="eyebrow">MODEL PERFORMANCE</p>
          <h1 id="page-title">Traffic sign analytics</h1>
          <p class="subheading">{dataset} {split} · {sample_count:,} real images · {represented_count} represented classes</p>
        </div>
        <span class="status"><span class="status-dot" aria-hidden="true"></span>Saved report loaded</span>
      </section>
      {report_picker}
      {report_summary}
      <section class="summary" aria-label="Evaluation summary">
        <article class="summary-card"><p>Images evaluated</p><strong>{sample_count:,}</strong><span>{dataset} {split}</span></article>
        <article class="summary-card"><p>Classes represented</p><strong>{represented_count}</strong><span>of {len(report["classes"])} listed classes</span></article>
        <article class="summary-card"><p>Saved reports</p><strong>{report_count}</strong><span>available for selection</span></article>
      </section>
      <section class="panel" id="models" aria-labelledby="models-title">
        <div class="panel-heading"><div><h2 id="models-title">Model comparison</h2><p>Accuracy bars use the same 0–100% scale; changes are relative to FP32.</p></div></div>
        <div class="model-grid" aria-label="Model accuracy">{model_cards}</div>
        <ul class="agreement" aria-label="Prediction agreement">{agreements}</ul>
      </section>
      {benchmark_panel}
      <section class="panel" id="classes" aria-labelledby="classes-title">
        <div class="panel-heading"><div><h2 id="classes-title">Per-class accuracy</h2><p>Compare recorded model scores for each class in this {sample_count:,}-image report.</p></div></div>
        <div class="table-tools">
          <label for="class-search">Search classes<input id="class-search" type="search" placeholder="For example, 17" autocomplete="off"></label>
          <label for="model-filter">Show accuracy for<select id="model-filter"><option value="all">All formats</option>{''.join(f'<option value="{escape(name)}">{escape(label)}</option>' for name, label in model_names.items())}</select></label>
          <label for="class-sort">Sort classes<select id="class-sort"><option value="class">Class number</option><option value="samples">Most samples</option><option value="ptq-drop">Largest PTQ drop vs FP32</option></select></label>
          <p class="results-count" id="class-count" role="status" aria-live="polite">Showing {len(report["classes"])} classes</p>
        </div>
        <div class="class-chart" id="class-chart" role="list" aria-label="Per-class model accuracy comparison">{class_chart}</div>
        <p class="class-chart-note">Bars use the same 0–100% scale. Changes are percentage points relative to FP32; absent values are not treated as zero.</p>
        <div class="table-wrap" role="region" tabindex="0" aria-label="Per-class accuracy table"><table id="class-table"><thead><tr><th scope="col">Class</th><th scope="col">Samples</th>{model_headers}</tr></thead><tbody>{class_rows}{empty_class_row}</tbody></table></div>
      </section>
      <section class="panel" id="evaluations" aria-labelledby="evaluations-title">
        <div class="panel-heading"><div><h2 id="evaluations-title">Evaluation report</h2><p>Fields not recorded in the source report are identified below.</p></div></div>
        <div class="table-wrap" role="region" tabindex="0" aria-label="Evaluation report table"><table><thead><tr><th scope="col">Run</th><th scope="col">Model</th><th scope="col">Format</th><th scope="col">Accuracy</th><th scope="col">Robustness</th><th scope="col">Date</th></tr></thead><tbody>{evaluation_rows}</tbody></table></div>
        <p class="note">Robustness results and run dates are not present in this validation report.</p>
      </section>
      <details><summary>Report provenance</summary><pre>{provenance}</pre></details>
    </main>
    <script>
      const search = document.querySelector("#class-search");
      const modelFilter = document.querySelector("#model-filter");
      const sortSelect = document.querySelector("#class-sort");
      const reportSelect = document.querySelector("#report-select");
      const overview = document.querySelector("#report-overview");
      const sortStatus = document.querySelector("#report-sort-status");
      const rows = [...document.querySelectorAll("#class-table tbody tr[data-search]")];
      const chartRows = [...document.querySelectorAll("#class-chart .class-chart-row")];
      const chart = document.querySelector("#class-chart");
      const tableBody = document.querySelector("#class-table tbody");
      const tableRows = new Map(rows.map(row => [row.dataset.classId, row]));
      const emptyRow = document.querySelector("#class-empty");
      const count = document.querySelector("#class-count");
      function score(row, name) {{
        const value = row.dataset[name];
        if (value === "") return null;
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
      }}
      function sortClasses() {{
        const mode = sortSelect.value;
        const ordered = [...chartRows].sort((a, b) => {{
          if (mode === "samples") {{
            return Number(b.dataset.samples) - Number(a.dataset.samples)
              || Number(a.dataset.classId) - Number(b.dataset.classId);
          }}
          if (mode === "ptq-drop") {{
            const drop = row => {{
              const fp32 = score(row, "fp32");
              const ptq = score(row, "ptq");
              return fp32 === null || ptq === null ? null : fp32 - ptq;
            }};
            const aDrop = drop(a);
            const bDrop = drop(b);
            if (aDrop === null) return bDrop === null ? 0 : 1;
            if (bDrop === null) return -1;
            return bDrop - aDrop || Number(a.dataset.classId) - Number(b.dataset.classId);
          }}
          return Number(a.dataset.classId) - Number(b.dataset.classId);
        }});
        for (const row of ordered) {{
          chart.append(row);
          tableBody.insertBefore(tableRows.get(row.dataset.classId), emptyRow);
        }}
      }}
      function updateClasses() {{
        sortClasses();
        const query = search.value.trim().toLowerCase().replace(/^class\\s+/, "");
        let visible = 0;
        for (const row of rows) {{
          row.hidden = Boolean(query) && row.dataset.search !== `class ${{query}}`;
          if (!row.hidden) visible += 1;
        }}
        for (const row of chartRows) {{
          row.hidden = Boolean(query) && row.dataset.search !== `class ${{query}}`;
        }}
        emptyRow.hidden = visible !== 0;
        for (const model of document.querySelectorAll("#class-table [data-model], #class-chart [data-model]")) {{
          model.hidden = modelFilter.value !== "all" && model.dataset.model !== modelFilter.value;
        }}
        count.textContent = `Showing ${{visible}} of ${{rows.length}} classes`;
      }}
      search.addEventListener("input", updateClasses);
      modelFilter.addEventListener("change", updateClasses);
      sortSelect.addEventListener("change", updateClasses);
      if (reportSelect) {{
        reportSelect.addEventListener("change", () => {{
          window.location.href = reportSelect.value;
        }});
      }}
      if (overview) {{
        const overviewBody = overview.tBodies[0];
        let sortColumn = -1;
        let sortDirection = 1;
        for (const button of overview.querySelectorAll("[data-sort-column]")) {{
          button.addEventListener("click", () => {{
            const column = Number(button.dataset.sortColumn);
            sortDirection = column === sortColumn ? -sortDirection : 1;
            sortColumn = column;
            const rows = [...overviewBody.rows];
            rows.sort((a, b) => {{
              const left = a.cells[column].dataset.sortValue ?? "";
              const right = b.cells[column].dataset.sortValue ?? "";
              if (left === "" || right === "") return left === right ? 0 : left === "" ? 1 : -1;
              const result = column < 4
                ? left.localeCompare(right)
                : Number(left) - Number(right);
              return result * sortDirection;
            }});
            for (const row of rows) overviewBody.append(row);
            for (const header of overview.querySelectorAll("thead th")) {{
              header.setAttribute("aria-sort", header === button.closest("th")
                ? sortDirection === 1 ? "ascending" : "descending"
                : "none");
            }}
            sortStatus.textContent = "Sorted by " + button.textContent.trim()
              + (sortDirection === 1 ? ", ascending." : ", descending.");
          }});
        }}
      }}
    </script>
  </body>
</html>
"""


def create_dashboard(report_path: str | Path, output_path: str | Path) -> Path:
    """Build a dashboard from one report or a directory of saved comparisons."""
    source = Path(report_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        output.write_text(dashboard_html(load_report(source)), encoding="utf-8")
        return output

    reports = discover_reports(source)
    if not reports:
        raise ValueError(f"no valid comparison reports found in {source}")

    report_dir = output.with_name(f"{output.stem}_reports")
    root_options = []
    detail_options = []
    for index, (path, report) in enumerate(reports):
        slug = f"{quote(path.stem, safe='._-')}.html"
        root_href = output.name if index == 0 else f"{report_dir.name}/{slug}"
        detail_href = f"../{output.name}" if index == 0 else slug
        dataset = report.get("dataset") or "Dataset"
        split = report.get("split") or "unspecified split"
        label = f"{dataset} {split} · {report['samples']:,} images — {path.stem}"
        root_options.append((root_href, label))
        detail_options.append((detail_href, label))

    for index, (_, report) in enumerate(reports):
        if index == 0:
            target = output
            options = root_options
            selected = root_options[0][0]
        else:
            report_dir.mkdir(parents=True, exist_ok=True)
            target = report_dir / f"{quote(reports[index][0].stem, safe='._-')}.html"
            options = detail_options
            selected = detail_options[index][0]
        target.write_text(
            dashboard_html(
                report,
                options,
                selected,
                [
                    (root_options[i][0], reports[i][0].stem, reports[i][1])
                    for i in range(len(reports))
                ]
                if index == 0
                else None,
            ),
            encoding="utf-8",
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a local HTML view of a saved model comparison"
    )
    parser.add_argument("report", type=Path, help="saved comparison JSON or directory")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/dashboard.html")
    )
    args = parser.parse_args()
    try:
        output = create_dashboard(args.report, args.output)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"saved dashboard to {output}")


def _percent(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}"


def _meter(label: str, value: float | None) -> str:
    if value is None:
        return '<span class="metric-unavailable">Accuracy unavailable</span>'
    return (
        f'<meter min="0" max="1" value="{value:.6f}" '
        f'aria-label="{escape(label)} accuracy">{_percent(value)}</meter>'
    )


def _delta(value: float | None, reference: float | None, name: str) -> str:
    if value is None:
        return "score unavailable"
    if name == "fp32":
        return "reference"
    if reference is None:
        return "FP32 score unavailable"
    return f"{(value - reference) * 100:+.1f} pp vs FP32"


def _saved_report_section(
    summaries: list[tuple[str, str, Mapping[str, Any]]] | None,
    selected_report: str | None = None,
) -> str:
    if not summaries:
        return ""

    rows = []
    for href, filename, report in summaries:
        current = ' aria-current="page"' if href == selected_report else ""
        provenance = report.get("provenance", {})
        selection = provenance.get("selection", {}) if isinstance(provenance, Mapping) else {}
        cohort = selection.get("selection") if isinstance(selection, Mapping) else None
        cohort = str(cohort).replace("_", " ").title() if cohort else "Not recorded"
        models = report.get("models", {})
        config = provenance.get("config", {}) if isinstance(provenance, Mapping) else {}
        fp32_timing = provenance.get("fp32_timing", {}) if isinstance(provenance, Mapping) else {}
        ptq_timing = provenance.get("quantized_timing", {}) if isinstance(provenance, Mapping) else {}
        if not isinstance(config, Mapping):
            config = {}
        if not isinstance(fp32_timing, Mapping):
            fp32_timing = {}
        if not isinstance(ptq_timing, Mapping):
            ptq_timing = {}
        scores = []
        score_values = []
        for name in ("fp32", "ptq", "qat"):
            metrics = models.get(name, {}) if isinstance(models, Mapping) else {}
            score = metrics.get("accuracy") if isinstance(metrics, Mapping) else None
            scores.append(_percent(score) if score is not None else "Not recorded")
            score_values.append(_sort_number(score))
        size_reduction = config.get("checkpoint_size_reduction")
        size_reduction = (
            size_reduction * 100 if _valid_measure(size_reduction) else None
        )
        report_name = escape(filename.casefold(), quote=True)
        dataset = escape(str(report.get("dataset") or "Not recorded"), quote=True)
        split = escape(str(report.get("split") or "Not recorded"), quote=True)
        cohort_sort = escape(cohort.casefold(), quote=True)
        rows.append(
            "<tr>"
            f'<th scope="row" data-sort-value="{report_name}"><a href="{escape(href, quote=True)}"{current}>'
            f'{escape(filename)}</a></th>'
            f'<td data-sort-value="{dataset}">{escape(str(report.get("dataset") or "Not recorded"))}</td>'
            f'<td data-sort-value="{split}">{escape(str(report.get("split") or "Not recorded"))}</td>'
            f'<td data-sort-value="{cohort_sort}">{escape(cohort)}</td>'
            f'<td data-sort-value="{report["samples"]}">{report["samples"]:,}</td>'
            + "".join(
                f'<td data-sort-value="{value}">{score}</td>'
                for score, value in zip(scores, score_values)
            )
            + f'<td data-sort-value="{_sort_number(fp32_timing.get("mean_ms"))}">{_measure(fp32_timing.get("mean_ms"), " ms")}</td>'
            + f'<td data-sort-value="{_sort_number(ptq_timing.get("mean_ms"))}">{_measure(ptq_timing.get("mean_ms"), " ms")}</td>'
            + f'<td data-sort-value="{_sort_number(provenance.get("speedup"))}">{_measure(provenance.get("speedup"), "×")}</td>'
            + f'<td data-sort-value="{_sort_number(size_reduction)}">{_measure(size_reduction, "%", 1)}</td>'
            + "</tr>"
        )

    return f"""
      <section class="panel" id="saved-reports" aria-labelledby="saved-reports-title">
        <div class="panel-heading"><div><h2 id="saved-reports-title">Saved report overview</h2><p>Recorded scores and sample counts from each saved evaluation.</p></div></div>
        <div class="table-wrap" role="region" tabindex="0" aria-label="Saved report comparison"><table id="report-overview"><thead><tr>{''.join(f'<th scope="col" aria-sort="none"><button class="sort-button" type="button" data-sort-column="{index}">{label}</button></th>' for index, label in enumerate(("Report", "Dataset", "Split", "Selection", "Images", "FP32", "PTQ", "QAT", "FP32 mean", "PTQ mean", "PTQ speedup", "Size reduction")))}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
        <p class="visually-hidden" id="report-sort-status" role="status" aria-live="polite"></p>
        <p class="note">The selected report is marked. Benchmark values are copied from each saved run. Rows retain their own split and selection; results are not pooled or presented as matched-sample comparisons. Missing fields are marked “Not recorded.”</p>
      </section>
    """


def _benchmark_section(report: Mapping[str, Any]) -> str:
    provenance = report.get("provenance", {})
    if not isinstance(provenance, Mapping):
        return ""
    config = provenance.get("config", {})
    runtime = provenance.get("runtime", {})
    fp32 = provenance.get("fp32_timing")
    ptq = provenance.get("quantized_timing")
    if not isinstance(config, Mapping):
        config = {}
    if not isinstance(runtime, Mapping):
        runtime = {}

    rows = []
    has_timing = isinstance(fp32, Mapping) or isinstance(ptq, Mapping)
    if has_timing:
        for name, label, timing, size_key in (
            ("fp32", "FP32", fp32, "fp32_checkpoint_size_bytes"),
            ("ptq", "PTQ", ptq, "quantized_checkpoint_size_bytes"),
        ):
            if not isinstance(timing, Mapping):
                timing = {}
            size = config.get(size_key)
            size_text = (
                f"{size / 1024:,.1f} KiB"
                if isinstance(size, int) and not isinstance(size, bool) and size >= 0
                else "Not recorded"
            )
            rows.append(
                "<tr>"
                f"<th scope=\"row\">{label}</th>"
                f"<td>{_percent(report['models'].get(name, {}).get('accuracy'))}</td>"
                f"<td>{_measure(timing.get('mean_ms'), ' ms')}</td>"
                f"<td>{_measure(timing.get('p95_ms'), ' ms')}</td>"
                f"<td>{_measure(timing.get('images_per_second'), ' images/s', 0)}</td>"
                f"<td>{size_text}</td>"
                "</tr>"
            )

    reduction = config.get("checkpoint_size_reduction")
    reduction = reduction * 100 if _valid_measure(reduction) else None
    speedup = provenance.get("speedup")
    batch_size = config.get("batch_size")
    device = escape(str(runtime.get("device") or "Not recorded"))
    benchmark_note = "No timing values in this report"
    if has_timing:
        batch_note = (
            f" · batch size {batch_size}"
            if isinstance(batch_size, int) and not isinstance(batch_size, bool)
            else ""
        )
        benchmark_note = f"Saved measurements · {device}{batch_note}"
    if has_timing:
        body = (
            '<div class="table-wrap" role="region" tabindex="0" aria-label="Inference benchmark results"><table><thead><tr>'
            '<th scope="col">Format</th><th scope="col">Accuracy</th>'
            '<th scope="col">Mean latency</th><th scope="col">P95 latency</th>'
            '<th scope="col">Throughput</th><th scope="col">Checkpoint size</th>'
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            '<ul class="agreement" aria-label="Saved benchmark summary">'
            f'<li><span>PTQ speedup</span><strong>{_measure(speedup, "×")}</strong></li>'
            '<li><span>Checkpoint size reduction</span>'
            f'<strong>{_measure(reduction, "%", 1)}</strong></li></ul>'
            '<p class="note">Unrecorded benchmark fields are marked above. '
            "This page does not rerun inference.</p>"
        )
    else:
        body = (
            '<p class="note">Timing data was not recorded in this report. '
            "No inference measurements are displayed.</p>"
        )
    return f"""
      <section class="panel" id="benchmark" aria-labelledby="benchmark-title">
        <div class="panel-heading"><div><h2 id="benchmark-title">Inference benchmark</h2><p>{benchmark_note}</p></div></div>
        {body}
      </section>
    """


def _class_comparison(
    item: Mapping[str, Any],
    models: Mapping[str, Any],
    model_names: Mapping[str, str],
) -> str:
    class_id = item["class_id"]
    sample_count = item["samples"]
    scores = item["accuracy"]
    reference = scores.get("fp32")
    model_rows = []
    for name in models:
        score = scores.get(name)
        if score is None:
            note = "No samples" if sample_count == 0 else "Not recorded"
            bar = f'<span class="class-chart-note">{note}</span>'
        else:
            width = score * 100
            bar = (
                f'<span class="bar-track" aria-hidden="true"><span '
                f'class="bar-fill" style="--bar-width:{width:.2f}%"></span></span>'
            )
        model_rows.append(
            f'<div class="class-chart-model" data-model="{escape(name)}">'
            f'<span>{escape(model_names[name])}</span>{bar}'
            f'<strong>{_percent(score)}</strong>'
            f'<span class="class-chart-delta">'
            f'{escape(_delta(score, reference, name))}</span></div>'
        )
    return (
        f'<article class="class-chart-row" data-search="class {class_id}" '
        f'{_class_data_attributes(item)} '
        f'role="listitem"><div class="class-chart-label">'
        f'<strong>Class {class_id}</strong><span>{sample_count:,} images</span>'
        f'</div><div class="class-chart-models">{"".join(model_rows)}</div></article>'
    )


def _class_data_attributes(item: Mapping[str, Any]) -> str:
    scores = item["accuracy"]
    fp32 = scores.get("fp32")
    ptq = scores.get("ptq")
    fp32_value = "" if fp32 is None else format(fp32, ".12g")
    ptq_value = "" if ptq is None else format(ptq, ".12g")
    return (
        f'data-class-id="{item["class_id"]}" '
        f'data-samples="{item["samples"]}" '
        f'data-fp32="{escape(fp32_value, quote=True)}" '
        f'data-ptq="{escape(ptq_value, quote=True)}"'
    )


def _valid_measure(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _sort_number(value: Any) -> str:
    return f"{value:.12g}" if _valid_measure(value) else ""


def _measure(value: Any, unit: str, digits: int = 2) -> str:
    if not _valid_measure(value):
        return "Not recorded"
    return f"{value:,.{digits}f}{unit}"


if __name__ == "__main__":
    main()
