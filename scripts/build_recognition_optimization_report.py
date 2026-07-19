from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
DEBUG_DIR = PROJECT_ROOT / "data" / "debug" / "recognition-optimization-v67"
REPORT_PATH = PROJECT_ROOT / "docs" / "recognition_optimization_report_v67.html"
SUMMARY_PATH = PROCESSED / "recognition_optimization_current_best_v67_summary.json"
SOURCE_NOTES_PATH = PROCESSED / "recognition_optimization_report_v67_sources.json"
PLUGIN_ROOT = Path(
    r"C:\Users\Administrator\.codex\plugins\cache\openai-curated-remote"
    r"\data-analytics\0.2.6-d37358633e00"
)
HTML_SHELL = PLUGIN_ROOT / "assets" / "html-report-shell.html"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pooled(pattern: str, folds: tuple[int, ...] = (1, 2, 3)) -> dict[str, Any]:
    summaries = [
        load_json(PROCESSED / pattern.format(fold=fold))["summary"]
        for fold in folds
    ]
    eligible = sum(int(row["eligible_case_count"]) for row in summaries)
    hits = {
        cutoff: sum(int(row["hits"][cutoff]) for row in summaries)
        for cutoff in ("1", "10", "100", "1000")
    }
    return {
        "eligible": eligible,
        "hits": hits,
        "recall": {key: value / eligible for key, value in hits.items()},
    }


def tooltip(value: str, tooltip_id: str, datasets: list[str]) -> str:
    source = "Source: local frozen evaluation artifacts<br>Dataset: " + ", ".join(
        html.escape(name) for name in datasets
    )
    return (
        f'<span class="source-tooltip" tabindex="0" '
        f'aria-describedby="{tooltip_id}">{html.escape(value)}'
        f'<span class="source-tooltip-content" id="{tooltip_id}" role="tooltip">'
        f"{source}</span></span>"
    )


def goal_tooltip(value: str, tooltip_id: str) -> str:
    return (
        f'<span class="source-tooltip" tabindex="0" '
        f'aria-describedby="{tooltip_id}">{html.escape(value)}'
        f'<span class="source-tooltip-content" id="{tooltip_id}" role="tooltip">'
        "Source: user-defined project objective<br>"
        "Document: current recognition optimization goal"
        "</span></span>"
    )


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_records() -> list[dict[str, Any]]:
    stage = load_json(PROCESSED / "non_body_optimization_current_best_v63_summary.json")
    stage_by_slot = {row["slot"]: row for row in stage["records"]}

    head = pooled(
        "retrieval_query_only_v64_head_fold{fold:02d}_"
        "dino_siglip2_upper_to_face_rrf1.json"
    )
    hands = {
        "eligible": 945,
        "hits": {"1": 20, "10": 83, "100": 337, "1000": 863},
    }
    hands["recall"] = {
        key: value / hands["eligible"] for key, value in hands["hits"].items()
    }
    legs = pooled(
        "retrieval_query_only_v64_legs_v6_fold{fold:02d}_qv72_rrf_k1.json"
    )
    feet = pooled(
        "retrieval_query_only_v65_feet_fold{fold:02d}_"
        "dino_siglip2_rrf1_crop_rrf1.json"
    )

    body = stage_by_slot["BODY"]
    rows = [
        {
            "slot": "BODY",
            "eligible": int(body["eligible"]),
            "baseline_top1": float(body["r1"]),
            "baseline_top10": float(body["r10"]),
            "top1": float(body["r1"]),
            "top10": float(body["r10"]),
            "top1000": 1.0,
            "method": "DINOv2 query encoder + pair reranker",
            "scope": "strict 3-fold pooled",
            "sources": ["retrieval_funnel_v53_huiji_closed_v3_reranker_strict.json"],
        },
        {
            "slot": "HEAD",
            "eligible": head["eligible"],
            "baseline_top1": float(stage_by_slot["HEAD"]["r1"]),
            "baseline_top10": float(stage_by_slot["HEAD"]["r10"]),
            "top1": head["recall"]["1"],
            "top10": head["recall"]["10"],
            "top1000": head["recall"]["1000"],
            "method": "DINOv2 + SigLIP2 rank ensemble",
            "scope": "strict 3-fold pooled",
            "sources": [
                "retrieval_funnel_v58_head_upper_query_rrf_folds01-03.json",
                "retrieval_query_only_v64_head_fold01_dino_siglip2_upper_to_face_rrf1.json",
                "retrieval_query_only_v64_head_fold02_dino_siglip2_upper_to_face_rrf1.json",
                "retrieval_query_only_v64_head_fold03_dino_siglip2_upper_to_face_rrf1.json",
            ],
        },
        {
            "slot": "HANDS",
            "eligible": hands["eligible"],
            "baseline_top1": 12 / 945,
            "baseline_top10": 57 / 945,
            "top1": hands["recall"]["1"],
            "top10": hands["recall"]["10"],
            "top1000": hands["recall"]["1000"],
            "method": "DINOv2 + paired crop/gender routing + RRF",
            "scope": "strict 3-fold pooled; route selected on fold 1",
            "sources": [
                "retrieval_query_only_v63_hands_fold01_qv71_v5_consensus.json",
                "retrieval_query_only_v64_hands_fold02_qv71_v5_rrf_k1.json",
                "retrieval_query_only_v64_hands_fold03_qv71_v5_rrf_k1.json",
            ],
        },
        {
            "slot": "LEGS",
            "eligible": legs["eligible"],
            "baseline_top1": float(stage_by_slot["LEGS"]["r1"]),
            "baseline_top10": float(stage_by_slot["LEGS"]["r10"]),
            "top1": legs["recall"]["1"],
            "top10": legs["recall"]["10"],
            "top1000": legs["recall"]["1000"],
            "method": "DINOv2 query encoder + query-crop RRF",
            "scope": "strict 3-fold pooled",
            "sources": [
                "retrieval_query_only_v64_legs_v6_fold01_qv72_rrf_k1.json",
                "retrieval_query_only_v64_legs_v6_fold02_qv72_rrf_k1.json",
                "retrieval_query_only_v64_legs_v6_fold03_qv72_rrf_k1.json",
            ],
        },
        {
            "slot": "FEET",
            "eligible": feet["eligible"],
            "baseline_top1": float(stage_by_slot["FEET"]["r1"]),
            "baseline_top10": float(stage_by_slot["FEET"]["r10"]),
            "top1": feet["recall"]["1"],
            "top10": feet["recall"]["10"],
            "top1000": feet["recall"]["1000"],
            "method": "DINOv2 + SigLIP2 rank ensemble + query-crop RRF",
            "scope": "strict 3-fold pooled",
            "sources": [
                "retrieval_query_only_v65_feet_fold01_dino_siglip2_rrf1_crop_rrf1.json",
                "retrieval_query_only_v65_feet_fold02_dino_siglip2_rrf1_crop_rrf1.json",
                "retrieval_query_only_v65_feet_fold03_dino_siglip2_rrf1_crop_rrf1.json",
            ],
        },
    ]
    for row in rows:
        row["delta_top1"] = row["top1"] - row["baseline_top1"]
        row["delta_top10"] = row["top10"] - row["baseline_top10"]
        row["target_gap"] = 0.8 - row["top1"]
    return rows


def build_fallback_svg(rows: list[dict[str, Any]]) -> str:
    left, width = 150, 730
    max_value = 0.8
    scale = width / max_value
    svg = [
        '<svg viewBox="0 0 960 460" role="img" '
        'aria-label="Baseline and current Top-1 accuracy by equipment slot">',
        '<style>.axis{fill:#737373;font:12px system-ui}.label{fill:currentColor;'
        'font:600 13px system-ui}.value{fill:currentColor;font:12px ui-monospace,'
        'monospace}.grid{stroke:#d8d8d8;stroke-width:1}.base{fill:#a3a3a3}'
        '.current{fill:#0169cc}</style>',
    ]
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8):
        x = left + tick * scale
        svg.append(
            f'<line class="grid" x1="{x:.1f}" y1="45" x2="{x:.1f}" y2="410"/>'
        )
        svg.append(
            f'<text class="axis" x="{x:.1f}" y="430" text-anchor="middle">'
            f"{tick:.0%}</text>"
        )
    svg.extend(
        [
            '<rect class="base" x="590" y="12" width="18" height="10" rx="2"/>',
            '<text class="axis" x="614" y="22">Comparable baseline</text>',
            '<rect class="current" x="760" y="12" width="18" height="10" rx="2"/>',
            '<text class="axis" x="784" y="22">Current</text>',
        ]
    )
    for index, row in enumerate(rows):
        y = 68 + index * 69
        baseline_width = row["baseline_top1"] * scale
        current_width = row["top1"] * scale
        svg.append(
            f'<text class="label" x="20" y="{y + 21}">{row["slot"]}</text>'
        )
        svg.append(
            f'<rect class="base" x="{left}" y="{y}" width="{baseline_width:.1f}" '
            'height="18" rx="3"/>'
        )
        svg.append(
            f'<text class="value" x="{left + baseline_width + 7:.1f}" '
            f'y="{y + 14}">{pct(row["baseline_top1"])}</text>'
        )
        svg.append(
            f'<rect class="current" x="{left}" y="{y + 25}" '
            f'width="{current_width:.1f}" height="18" rx="3"/>'
        )
        svg.append(
            f'<text class="value" x="{left + current_width + 7:.1f}" '
            f'y="{y + 39}">{pct(row["top1"])}</text>'
        )
    svg.append("</svg>")
    return "".join(svg)


def build_report(rows: list[dict[str, Any]]) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    title = "FFXIV Prism Recognition Optimization"
    all_sources = sorted({source for row in rows for source in row["sources"]})
    goal_deck = goal_tooltip("80%", "goal-deck")
    goal_card = goal_tooltip("80%", "goal-card")
    goal_chart = goal_tooltip("0?80%", "goal-chart")
    fold_one = tooltip("fold 1", "method-fold-one", all_sources)
    crop_rrf = tooltip("RRF(k=1)", "method-crop-rrf", all_sources)
    branch_rrf = tooltip("RRF(k=1)", "method-branch-rrf", all_sources)
    source_text = (
        "Source: local frozen evaluation artifacts<br>Dataset: "
        + ", ".join(html.escape(name) for name in all_sources)
    )

    head = next(row for row in rows if row["slot"] == "HEAD")
    feet = next(row for row in rows if row["slot"] == "FEET")
    hands = next(row for row in rows if row["slot"] == "HANDS")
    metrics = "".join(
        [
            '<div class="metric"><div class="metric-label">HEAD Top-1</div>'
            f'<div class="metric-value">{tooltip(pct(head["top1"]), "m-head", head["sources"])}</div>'
            f'<span class="pill">+{head["delta_top1"] * 100:.2f} pp</span>'
            '<div class="metric-note">strict 3-fold pooled</div></div>',
            '<div class="metric"><div class="metric-label">FEET Top-1</div>'
            f'<div class="metric-value">{tooltip(pct(feet["top1"]), "m-feet", feet["sources"])}</div>'
            f'<span class="pill">+{feet["delta_top1"] * 100:.2f} pp</span>'
            '<div class="metric-note">strict 3-fold pooled</div></div>',
            '<div class="metric"><div class="metric-label">HANDS Top-1</div>'
            f'<div class="metric-value">{tooltip(pct(hands["top1"]), "m-hands", hands["sources"])}</div>'
            '<span class="pill warn">weakest slot</span>'
            '<div class="metric-note">strict 3-fold pooled</div></div>',
            '<div class="metric"><div class="metric-label">Target status</div>'
            f'<div class="metric-value">Not met</div><span class="pill warn">{goal_card} per slot</span>'
            '<div class="metric-note">candidate retrieval is usable; automatic identification is not</div></div>',
        ]
    )

    table_rows = []
    for index, row in enumerate(rows):
        sources = row["sources"]
        table_rows.append(
            "<tr>"
            f'<td>{row["slot"]}</td>'
            f'<td>{tooltip(str(row["eligible"]), f"n-{index}", sources)}</td>'
            f'<td>{tooltip(pct(row["baseline_top1"]), f"b1-{index}", sources)}</td>'
            f'<td>{tooltip(pct(row["top1"]), f"c1-{index}", sources)}</td>'
            f'<td class="positive">{tooltip(f"{row["delta_top1"] * 100:+.2f} pp", f"d1-{index}", sources)}</td>'
            f'<td>{tooltip(pct(row["top10"]), f"c10-{index}", sources)}</td>'
            f'<td>{tooltip(pct(row["top1000"]), f"c1000-{index}", sources)}</td>'
            f"<td>{html.escape(row['method'])}</td>"
            "</tr>"
        )
    semantic_table = (
        "<table><thead><tr><th>Slot</th><th>Eligible</th><th>Baseline Top-1</th>"
        "<th>Current Top-1</th><th>? Top-1</th><th>Current Top-10</th>"
        "<th>Current Top-1000</th><th>Promoted method</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table>"
    )

    main = f"""
    <main data-report-audience="technical">
      <article class="reading">
        <div class="kicker">Recognition optimization ? 2026-07-10</div>
        <header data-contract-section="title"><h1>{title}</h1></header>
        <p class="deck">Frozen, appearance-disjoint three-fold validation shows
        material ranking gains, especially for HEAD and FEET, but no slot-level
        result supports the requested {goal_deck} automatic-identification target.</p>

        <section class="summary" data-contract-section="technical-summary">
          <div class="summary-label">Technical Summary</div>
          <div class="summary-body">
            <p><strong>The project is now usable as a local candidate-retrieval tool.</strong>
            HEAD and FEET use a DINOv2 + SigLIP2 reciprocal-rank ensemble; LEGS,
            FEET, and HANDS use query-crop consensus. The CLI validates assets,
            crops one screenshot, retrieves candidates, and emits JSON plus HTML.</p>
            <p><strong>Accuracy remains the release constraint.</strong> BODY is still
            the strongest strict result, while HANDS remains far below a safe
            automatic decision threshold. Results must be treated as ranked
            suggestions for human confirmation.</p>
          </div>
        </section>
        <section class="metrics">{metrics}</section>

        <section class="narrative" data-contract-section="key-findings">
          <h2>Cross-backbone consensus moved HEAD and FEET most</h2>
          <p>The shared pattern is a ranking bottleneck: the correct appearance is
          usually present deep in the candidate pool, while a single visual
          backbone orders it poorly. Reciprocal-rank fusion captures complementary
          DINOv2 and SigLIP2 errors. The same method was rejected for LEGS because
          candidate coverage regressed, and for HANDS because Top-1 fell.</p>
        </section>
      </article>

      <div class="wide">
        <figure class="card source-figure">
          <div class="card-head"><h3>Appearance Top-1 by equipment slot</h3>
          <p>Comparable frozen baseline versus current promoted configuration;
          common {goal_chart} scale.</p></div>
          <div class="chart-wrap">
            <div data-recharts-chart="slot-top1">
              <div class="chart-fallback" data-recharts-fallback>
                {build_fallback_svg(rows)}
              </div>
              <div data-recharts-live aria-hidden="true"></div>
            </div>
          </div>
          <figcaption class="chart-note">BODY did not change in this iteration.
          HEAD and FEET gain from cross-backbone rank consensus; HANDS, LEGS,
          and FEET use multi-crop reciprocal-rank fusion.</figcaption>
          <button type="button" class="source-tooltip"
          aria-describedby="chart-source-tooltip-1">Source<span
          class="source-tooltip-content" id="chart-source-tooltip-1"
          role="tooltip">{source_text}</span></button>
        </figure>
      </div>

      <article class="reading">
        <section class="card table-card">
          <div class="card-head"><h3>Frozen evaluation results</h3>
          <p>Exact pooled results and promoted methods; percentages use eligible
          appearance-labeled screenshots as the denominator.</p></div>
          <div class="table-scroll">{semantic_table}</div>
        </section>

        <section class="narrative"
        data-contract-section="scope-data-and-metric-definitions">
          <h2>Scope, data, and metric definitions</h2>
          <p>Evaluation uses front/full-body player screenshots whose equipment
          appearances are disjoint across folds. ?Top-1? means the expected
          appearance is the first ranked item; ?Top-10? and ?Top-1000? mean it
          appears within those candidate cutoffs. Appearance equivalence follows
          official model mappings rather than exact item ID alone. The BODY,
          HEAD, LEGS, and FEET results are strict pooled estimates. HANDS uses the
          same folds, but its crop route was originally selected on {fold_one}, so the
          later folds are the stronger confirmation evidence.</p>
        </section>

        <section class="narrative" data-contract-section="methodology">
          <h2>Methodology and experimental design</h2>
          <p>Each experiment freezes the canonical candidate index and
          appearance-disjoint fold, changes one ranking component, and retains a
          route only after confirmation. Query-crop {crop_rrf} was promoted for
          HANDS, LEGS, and FEET. A public SigLIP2 image encoder was evaluated
          zero-shot against the same candidate renders; equal branch-level
          {branch_rrf} was promoted for HEAD and FEET. Candidate degradation,
          query sharpening, job filters, gender-only candidate routing, and
          unconfirmed role-weight changes were rejected.</p>
        </section>

        <section class="narrative"
        data-contract-section="limitations-uncertainty-and-robustness-checks">
          <h2>Limitations, uncertainty, and robustness checks</h2>
          <p>The source screenshots are low resolution and provide only a few
          images per appearance identity; HEAD, HANDS, and FEET crops therefore
          contain limited discriminative texture. Fold-level direction was checked
          before promotion, and gender/race cuts were reviewed. Both genders
          improve for HEAD and FEET, but small race cohorts remain noisy: FEET
          Lalafell Top-1 regressed while its deeper recall improved, and HEAD
          Roegadyn Top-10 lost a small number of hits. LEGS cross-backbone fusion was rejected
          because Top-1000 coverage declined. These checks support candidate
          ranking, not calibrated probability or autonomous item selection.</p>
        </section>

        <section class="narrative" data-contract-section="recommended-next-steps">
          <h2>Recommended next steps</h2>
          <ul>
            <li>Prioritize higher-resolution or multi-view player crops, especially
            for HANDS; ranking changes alone have reached diminishing returns.</li>
            <li>Train a shared render-to-screenshot student using the successful
            DINOv2/SigLIP2 teacher ranks, then evaluate on a newly held-out fold.</li>
            <li>Add race-aware monitoring and a conservative fallback for small
            domains before claiming domain parity.</li>
            <li>Keep the product contract as Top-K candidate assistance until every
            slot clears a predeclared automatic-identification threshold.</li>
          </ul>
        </section>

        <section class="narrative" data-contract-section="further-questions">
          <h2>Further questions</h2>
          <ul>
            <li>Can higher-resolution originals be recovered for the existing
            labeled screenshots without changing the identity split?</li>
            <li>Would a dedicated hand/foot part detector improve crop purity more
            than another global visual backbone?</li>
            <li>What latency and storage budget is acceptable for retaining two
            candidate indexes in production?</li>
          </ul>
        </section>
      </article>
    </main>
    """

    shell = HTML_SHELL.read_text(encoding="utf-8")
    shell = re.sub(
        r"\s*<!-- Deliver this HTML file itself\..*?-->",
        "",
        shell,
        flags=re.DOTALL,
    )
    shell = shell.replace("{{TITLE}}", title)
    shell = shell.replace("{{SOURCE_AND_DATE}}", "Frozen local evaluation ? 10 July 2026")
    shell = shell.replace("Data Analytics", "FFXIV Prism")
    shell = re.sub(r"<main\b[^>]*>.*?</main>", main, shell, flags=re.DOTALL)
    shell = shell.replace(
        "</style>",
        """
    ul { margin: 10px 0 0; padding-left: 22px; color: var(--secondary); }
    li + li { margin-top: 8px; }
    code { padding: 1px 5px; border-radius: 5px; background: var(--surface-tertiary); }
    @media (max-width: 800px) {
      .table-card { width: calc(100vw - 36px); }
    }
  </style>""",
    )
    shell_path = DEBUG_DIR / "report-shell.html"
    payload_path = DEBUG_DIR / "report-payload.json"
    shell_path.write_text(shell, encoding="utf-8")

    chart_rows = []
    for row in rows:
        for series, value in (
            ("Comparable baseline", row["baseline_top1"]),
            ("Current", row["top1"]),
        ):
            chart_rows.append(
                {
                    "slot": row["slot"],
                    "series": series,
                    "top1": value,
                    "eligible": row["eligible"],
                    "top10": row["top10"],
                    "method": row["method"],
                }
            )
    payload = {
        "charts": [
            {
                "id": "slot-top1",
                "height": 380,
                "type": "bar",
                "dataset": {
                    "id": "slot-top1",
                    "title": "Appearance Top-1 by equipment slot",
                    "data": chart_rows,
                    "chart_spec": {
                        "id": "slot-top1",
                        "dataset": "slot-top1",
                        "title": "Appearance Top-1 by equipment slot",
                        "type": "bar",
                        "encodings": {
                            "x": {"field": "slot", "type": "nominal"},
                            "y": {
                                "field": "top1",
                                "label": "Appearance Top-1",
                                "type": "quantitative",
                            },
                            "color": {"field": "series", "type": "nominal"},
                            "tooltip": [
                                {"field": "eligible", "label": "Eligible"},
                                {"field": "top10", "label": "Current Top-10"},
                                {"field": "method", "label": "Method"},
                            ],
                        },
                        "xAxisTitle": "",
                        "yAxisTitle": "Appearance Top-1",
                        "valueFormat": "percent",
                    },
                    "settings": {
                        "orientation": "horizontal",
                        "groupMode": "grouped",
                        "domain": [0, 0.8],
                    },
                },
            }
        ]
    }
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "version": 67,
        "generated_at": "2026-07-10",
        "goal_top1": 0.8,
        "goal_achieved": False,
        "product_status": "usable_candidate_retrieval_not_automatic_identification",
        "records": rows,
        "promoted_changes": [
            "LEGS and FEET query-crop RRF(k=1)",
            "HANDS paired crop/gender routing with RRF(k=1), confirmed directionally",
            "HEAD DINOv2+SigLIP2 branch RRF(k=1), upper query to face candidate route",
            "FEET DINOv2+SigLIP2 branch RRF(k=1) plus query-crop RRF(k=1)",
            "CLI support for multi-backbone retrieval and asset validation",
        ],
        "rejected_changes": [
            "LEGS cross-backbone ensemble: Top-1000 regression",
            "HANDS cross-backbone ensemble: Top-1 regression",
            "candidate degradation, query sharpening, job filters, and role-weight changes",
        ],
        "report": "docs/recognition_optimization_report_v67.html",
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    source_notes = {
        "audience": "technical",
        "delivery_mode": "html",
        "section_map": [
            "title",
            "technical-summary",
            "key-findings",
            "scope-data-and-metric-definitions",
            "methodology",
            "limitations-uncertainty-and-robustness-checks",
            "recommended-next-steps",
            "further-questions",
        ],
        "chart_map": [
            {
                "section": "key-findings",
                "question": "How did comparable Top-1 change by slot?",
                "family": "comparison",
                "type": "grouped horizontal bar",
                "fields": ["slot", "series", "top1", "eligible", "top10", "method"],
                "takeaway": "HEAD and FEET gain most; every slot remains below target.",
                "palette": "hard two-root cap: neutral baseline and blue current",
                "fallback": "same-data inline SVG",
            }
        ],
        "source_files": all_sources,
        "omissions": {
            "second_chart": "One grouped bar plus the exact table fully covers the discrete five-slot comparison.",
            "visible_sources_appendix": "Source provenance is attached to values and the chart via tooltips.",
        },
        "qa": [
            "zero-based common 0-80% visual scale",
            "same-data static SVG fallback",
            "system light/dark appearance preserved",
            "source tooltips attached to metric cards, table values, and chart",
        ],
        "qa_limitations": [
            "Interactive browser rendering QA was blocked by the in-app browser file URL policy; static HTML and embedded-runtime checks were completed instead."
        ],
    }
    SOURCE_NOTES_PATH.write_text(
        json.dumps(source_notes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "shell": str(shell_path),
                "payload": str(payload_path),
                "report": str(REPORT_PATH),
                "summary": str(SUMMARY_PATH),
                "source_notes": str(SOURCE_NOTES_PATH),
            },
            indent=2,
        )
    )


def main() -> int:
    build_report(build_records())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
