import argparse
from collections.abc import Mapping
from html import escape
import json
from pathlib import Path
from typing import Any

from traffic_sign_quantization.dashboard_data import load_report


def dashboard_html(report: Mapping[str, Any]) -> str:
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
    model_name = escape(str(report["provenance"].get("model") or "Not recorded"))
    reference = models.get("fp32", {}).get("accuracy")
    model_cards = "".join(
        f'<article class="model-card"><span>{escape(label)}</span>'
        f'<strong>{_percent(metrics.get("accuracy"))}</strong>'
        f'<meter min="0" max="1" value="{metrics["accuracy"]:.6f}" '
        f'aria-label="{escape(label)} accuracy">{_percent(metrics["accuracy"])}</meter>'
        f'<span class="card-note">Overall accuracy · '
        f'{_delta(metrics["accuracy"], reference, name)}</span></article>'
        for name, metrics in models.items()
        for label in (model_names[name],)
    )
    agreements = "".join(
        f'<li><span>{escape(name.replace("_", " → ").upper())}</span>'
        f'<strong>{_percent(value)}</strong></li>'
        for name, value in report["agreement"].items()
    )
    model_headers = "".join(
        f'<th data-model="{escape(name)}" scope="col">{escape(label)}</th>'
        for name, label in model_names.items()
    )
    class_rows = "".join(
        '<tr data-search="class '
        f'{item["class_id"]}"><th scope="row">{item["class_id"]}</th>'
        f'<td>{item["samples"]:,}</td>'
        + "".join(
            f'<td data-model="{escape(name)}">'
            f'{_percent(item["accuracy"].get(name))}</td>'
            for name in models
        )
        + "</tr>"
        for item in report["classes"]
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
      label {{ display: grid; gap: 5px; color: #657385; font-size: 12px; }}
      input, select {{ min-height: 38px; padding: 7px 10px; border: 1px solid #cbd4df; border-radius: 7px; background: #fff; color: #202a35; font: inherit; }}
      input {{ width: min(270px, 75vw); }}
      input:focus, select:focus, a:focus {{ outline: 3px solid #9cc0f4; outline-offset: 2px; }}
      .results-count {{ margin: 0 0 8px auto; color: #748091; font-size: 12px; }}
      .table-wrap {{ overflow-x: auto; }}
      table {{ width: 100%; border-collapse: collapse; text-align: left; white-space: nowrap; }}
      th, td {{ padding: 12px 18px; border-bottom: 1px solid #edf0f4; font-size: 12px; }}
      th {{ background: #fafbfd; color: #768293; font-weight: 600; }}
      tbody th {{ color: #202a35; }}
      .note {{ padding: 0 22px 18px; color: #748091; font-size: 12px; }}
      details {{ margin: 20px 0; padding: 16px 20px; border: 1px solid #e4e9ef; border-radius: 12px; background: #fff; }}
      summary {{ cursor: pointer; font-weight: 600; }}
      pre {{ overflow: auto; padding: 14px; border-radius: 8px; background: #f3f6fa; font-size: 12px; }}
      [hidden] {{ display: none !important; }}
      @media (max-width: 640px) {{ .topbar {{ align-items: flex-start; flex-direction: column; gap: 8px; padding-top: 12px; }} nav {{ min-height: 42px; gap: 20px; }} main {{ width: 92vw; margin-top: 30px; }} .intro {{ align-items: flex-start; flex-direction: column; }} .summary {{ grid-template-columns: 1fr; gap: 10px; }} .summary-card {{ padding: 15px 17px; }} .panel-heading {{ align-items: flex-start; flex-direction: column; }} .results-count {{ width: 100%; margin-left: 0; }} }}
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
      <section class="summary" aria-label="Evaluation summary">
        <article class="summary-card"><p>Saved comparisons</p><strong>1</strong><span>One report shown</span></article>
        <article class="summary-card"><p>Model formats</p><strong>{len(models)}</strong><span>{escape(", ".join(model_names.values()))}</span></article>
        <article class="summary-card"><p>Corruption results</p><strong>Not measured</strong><span>This is a clean validation report</span></article>
      </section>
      <section class="panel" id="models" aria-labelledby="models-title">
        <div class="panel-heading"><div><h2 id="models-title">Model comparison</h2><p>Accuracy bars use the same 0–100% scale; changes are relative to FP32.</p></div></div>
        <div class="model-grid" aria-label="Model accuracy">{model_cards}</div>
        <ul class="agreement" aria-label="Prediction agreement">{agreements}</ul>
      </section>
      <section class="panel" id="classes" aria-labelledby="classes-title">
        <div class="panel-heading"><div><h2 id="classes-title">Per-class accuracy</h2><p>Class counts and scores from the same {sample_count:,}-image report.</p></div></div>
        <div class="table-tools">
          <label for="class-search">Search classes<input id="class-search" type="search" placeholder="For example, 17" autocomplete="off"></label>
          <label for="model-filter">Show accuracy for<select id="model-filter"><option value="all">All formats</option>{''.join(f'<option value="{escape(name)}">{escape(label)}</option>' for name, label in model_names.items())}</select></label>
          <p class="results-count" id="class-count" role="status" aria-live="polite">Showing {len(report["classes"])} classes</p>
        </div>
        <div class="table-wrap"><table id="class-table"><thead><tr><th scope="col">Class</th><th scope="col">Samples</th>{model_headers}</tr></thead><tbody>{class_rows}</tbody></table></div>
      </section>
      <section class="panel" id="evaluations" aria-labelledby="evaluations-title">
        <div class="panel-heading"><div><h2 id="evaluations-title">Evaluation report</h2><p>Fields not recorded in the source report are identified below.</p></div></div>
        <div class="table-wrap"><table><thead><tr><th scope="col">Run</th><th scope="col">Model</th><th scope="col">Format</th><th scope="col">Accuracy</th><th scope="col">Robustness</th><th scope="col">Date</th></tr></thead><tbody>{evaluation_rows}</tbody></table></div>
        <p class="note">Robustness results and run dates are not present in this validation report.</p>
      </section>
      <details><summary>Report provenance</summary><pre>{provenance}</pre></details>
    </main>
    <script>
      const search = document.querySelector("#class-search");
      const modelFilter = document.querySelector("#model-filter");
      const rows = [...document.querySelectorAll("#class-table tbody tr")];
      const count = document.querySelector("#class-count");
      function updateClasses() {{
        const query = search.value.trim().toLowerCase().replace(/^class\\s+/, "");
        let visible = 0;
        for (const row of rows) {{
          row.hidden = Boolean(query) && row.dataset.search !== `class ${{query}}`;
          if (!row.hidden) visible += 1;
        }}
        for (const cell of document.querySelectorAll("#class-table [data-model]")) {{
          cell.hidden = modelFilter.value !== "all" && cell.dataset.model !== modelFilter.value;
        }}
        count.textContent = `Showing ${{visible}} of ${{rows.length}} classes`;
      }}
      search.addEventListener("input", updateClasses);
      modelFilter.addEventListener("change", updateClasses);
    </script>
  </body>
</html>
"""


def create_dashboard(report_path: str | Path, output_path: str | Path) -> Path:
    """Build a dashboard from an existing comparison report, without changing it."""
    page = dashboard_html(load_report(report_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a local HTML view of a saved model comparison"
    )
    parser.add_argument("report", type=Path, help="saved comparison JSON")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/dashboard.html")
    )
    args = parser.parse_args()
    output = create_dashboard(args.report, args.output)
    print(f"saved dashboard to {output}")


def _percent(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}"


def _delta(value: float, reference: float | None, name: str) -> str:
    if name == "fp32":
        return "reference"
    if reference is None:
        return "FP32 score unavailable"
    return f"{(value - reference) * 100:+.1f} pp vs FP32"


if __name__ == "__main__":
    main()
