import argparse
from collections.abc import Mapping
from html import escape
import json
from pathlib import Path
from typing import Any

from traffic_sign_quantization.dashboard_data import load_report


def dashboard_html(report: Mapping[str, Any]) -> str:
    """Render a normalized comparison report as a self-contained HTML dashboard."""
    models = report["models"]
    cards = "".join(
        f'<article class="card"><span>{escape(name.upper())}</span>'
        f'<strong>{_percent(metrics.get("accuracy"))}</strong></article>'
        for name, metrics in models.items()
    )
    agreements = "".join(
        f'<li>{escape(name.replace("_", " "))}: {_percent(value)}</li>'
        for name, value in report["agreement"].items()
    )
    headings = "".join(f"<th>{escape(name.upper())}</th>" for name in models)
    rows = "".join(
        "<tr>"
        f"<th scope=\"row\">{item['class_id']}</th>"
        f"<td>{item['samples']:,}</td>"
        + "".join(
            f"<td>{_percent(item['accuracy'].get(name))}</td>"
            for name in models
        )
        + "</tr>"
        for item in report["classes"]
    )
    provenance = escape(
        json.dumps(report["provenance"], indent=2, sort_keys=True, default=str)
    )
    dataset = escape(str(report.get("dataset") or "Dataset"))
    split = escape(str(report.get("split") or "unspecified split"))
    sample_count = report["samples"]
    coverage = report["coverage"]
    represented = coverage.get("represented_classes")
    represented_count = len(represented) if represented is not None else sum(
        item["samples"] > 0 for item in report["classes"]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{dataset} model comparison</title>
  <style>
    :root {{ color-scheme: light; font: 16px/1.5 system-ui, sans-serif; color: #18212f; background: #f3f6fa; }}
    body {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px; }}
    h1 {{ margin-bottom: 4px; }}
    .muted {{ color: #58677a; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin: 24px 0; }}
    .card, section {{ background: white; border: 1px solid #dbe2ea; border-radius: 10px; padding: 16px; }}
    .card span {{ display: block; color: #58677a; font-size: .8rem; font-weight: 650; letter-spacing: .06em; }}
    .card strong {{ display: block; font-size: 1.8rem; margin-top: 8px; }}
    section {{ margin: 18px 0; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ padding: 9px 12px; border-bottom: 1px solid #e5eaf0; text-align: right; white-space: nowrap; }}
    thead th {{ color: #58677a; font-size: .8rem; }}
    th:first-child, td:first-child {{ text-align: left; }}
    details {{ margin-top: 20px; }}
    pre {{ overflow: auto; padding: 14px; background: #f3f6fa; border-radius: 8px; }}
  </style>
</head>
<body>
  <header>
    <h1>{dataset} model comparison</h1>
    <div class="muted">{escape(split)} · {sample_count:,} real samples · {represented_count} represented classes</div>
  </header>
  <main>
    <div class="cards" aria-label="Model accuracy">{cards}</div>
    <section aria-labelledby="agreement-title">
      <h2 id="agreement-title">Prediction agreement</h2>
      <ul>{agreements or '<li>No agreement values in this report.</li>'}</ul>
    </section>
    <section aria-labelledby="classes-title">
      <h2 id="classes-title">Per-class accuracy</h2>
      <table>
        <thead><tr><th scope="col">Class</th><th scope="col">Samples</th>{headings}</tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    <details>
      <summary>Report provenance</summary>
      <pre>{provenance}</pre>
    </details>
  </main>
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


if __name__ == "__main__":
    main()
