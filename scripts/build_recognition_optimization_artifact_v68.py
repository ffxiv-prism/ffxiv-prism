from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pooled(paths: list[Path], key: str = "summary") -> dict:
    summaries = [read_json(path)[key] for path in paths]
    eligible = sum(int(value["eligible_case_count"]) for value in summaries)
    cutoffs = (1, 10, 100, 1000)
    hits = {
        str(cutoff): sum(int(value["hits"][str(cutoff)]) for value in summaries)
        for cutoff in cutoffs
    }
    return {
        "eligible": eligible,
        "hits": hits,
        "recall": {key: value / eligible for key, value in hits.items()},
    }


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    previous = read_json(
        PROCESSED / "recognition_optimization_current_best_v67_summary.json"
    )
    head_files = [
        PROCESSED
        / f"retrieval_query_only_v68_head_fold{fold:02d}_dino_siglip2_upper_to_face_rrf1_fixed.json"
        for fold in range(1, 4)
    ]
    legs_files = [
        PROCESSED
        / f"retrieval_query_only_v68_legs_fold{fold:02d}_spatial_k10_b035.json"
        for fold in range(1, 4)
    ]
    head = pooled(head_files)
    legs = pooled(legs_files, key="reranked_summary")
    legs_baseline = pooled(legs_files, key="baseline_summary")

    records = [dict(row) for row in previous["records"]]
    by_slot = {row["slot"]: row for row in records}
    by_slot["HEAD"].update(
        {
            "top1": head["recall"]["1"],
            "top10": head["recall"]["10"],
            "top1000": head["recall"]["1000"],
            "method": "DINOv2 + SigLIP2 rank ensemble; upper query to face candidates",
            "sources": [path.name for path in head_files],
            "delta_top1": head["recall"]["1"]
            - by_slot["HEAD"]["baseline_top1"],
            "delta_top10": head["recall"]["10"]
            - by_slot["HEAD"]["baseline_top10"],
            "target_gap": 0.8 - head["recall"]["1"],
            "note": "v68 corrects an evaluator bug that ignored query/candidate crop arguments; this is a benchmark correction, not a new-model gain.",
        }
    )
    by_slot["LEGS"].update(
        {
            "top1": legs["recall"]["1"],
            "top10": legs["recall"]["10"],
            "top1000": legs["recall"]["1000"],
            "method": "DINOv2 query-crop RRF + grayscale spatial row-profile reranker",
            "sources": [path.name for path in legs_files],
            "delta_top1": legs["recall"]["1"]
            - by_slot["LEGS"]["baseline_top1"],
            "delta_top10": legs["recall"]["10"]
            - by_slot["LEGS"]["baseline_top10"],
            "target_gap": 0.8 - legs["recall"]["1"],
        }
    )

    summary = {
        **previous,
        "version": 68,
        "generated_at": generated_at,
        "records": records,
        "promoted_changes": previous["promoted_changes"]
        + [
            "LEGS Top-10 spatial row-profile reranking (blend=0.35)",
            "HEAD low-confidence warning when Top-1/Top-2 margin is below 0.01",
            "HEAD rank-ensemble evaluator now applies query/candidate crop arguments",
        ],
        "rejected_changes": previous["rejected_changes"]
        + [
            "HEAD global coverage reranking: strict-fold Top-1/Top-10 regression",
            "HEAD global DINOv2 spatial Top-10 reranking: fold-1 Top-1 regression",
            "HEAD guarded coverage reranking: folds 2/3 regression",
        ],
        "report": "docs/recognition_optimization_report_v68.html",
    }
    summary_path = PROCESSED / "recognition_optimization_current_best_v68_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    slot_rows = [
        {
            "slot": row["slot"],
            "eligible": row["eligible"],
            "top1": row["top1"],
            "top10": row["top10"],
            "top1000": row["top1000"],
            "method": row["method"],
        }
        for row in records
    ]
    slot_metric_rows = [
        {"slot": row["slot"], "metric": "Top-1", "value": row["top1"]}
        for row in records
    ] + [
        {"slot": row["slot"], "metric": "Top-10", "value": row["top10"]}
        for row in records
    ]
    legs_fold_rows = []
    for fold, path in enumerate(legs_files, start=1):
        payload = read_json(path)
        baseline = payload["baseline_summary"]
        reranked = payload["reranked_summary"]
        eligible = int(reranked["eligible_case_count"])
        legs_fold_rows.append(
            {
                "fold": fold,
                "eligible": eligible,
                "baseline_top1": baseline["hits"]["1"] / eligible,
                "spatial_top1": reranked["hits"]["1"] / eligible,
                "top10": reranked["hits"]["10"] / eligible,
            }
        )
    head_fold_rows = []
    for fold, path in enumerate(head_files, start=1):
        payload = read_json(path)
        current = payload["summary"]
        eligible = int(current["eligible_case_count"])
        head_fold_rows.append(
            {
                "fold": fold,
                "eligible": eligible,
                "top1": current["hits"]["1"] / eligible,
                "top10": current["hits"]["10"] / eligible,
                "top1000": current["hits"]["1000"] / eligible,
            }
        )
    head_experiments = [
        {
            "experiment": "Promoted retrieval (corrected evaluator)",
            "scope": "3 folds / 899 cases",
            "top1": head["recall"]["1"],
            "top10": head["recall"]["10"],
            "decision": "Keep",
        },
        {
            "experiment": "Coverage-only rerank",
            "scope": "Fold 1 / 305 cases",
            "top1": 47 / 305,
            "top10": 128 / 305,
            "decision": "Reject: Top-1 fell",
        },
        {
            "experiment": "Guarded coverage rerank",
            "scope": "3 folds / 899 cases",
            "top1": (67 + 80 + 60) / 899,
            "top10": (128 + 151 + 116) / 899,
            "decision": "Reject: folds 2/3 regressed",
        },
        {
            "experiment": "DINOv2 spatial Top-10 rerank",
            "scope": "Fold 1 / 305 cases",
            "top1": 62 / 305,
            "top10": 128 / 305,
            "decision": "Reject: Top-1 fell",
        },
    ]
    e2e = read_json(
        ROOT / "data" / "debug" / "glamour-match-e2e-v68-five-slots" / "results.json"
    )
    e2e_rows = [
        {
            "slot": row["slot"],
            "top1_candidate": row["candidates"][0]["name_en"],
            "quality": row.get("ranking_quality", {}).get("level", "normal"),
            "top_margin": row.get("top_score_margin"),
        }
        for row in e2e["slots"]
    ]

    source_inventory = {
        "generated_at": generated_at,
        "summary": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
        "head_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in head_files],
        "legs_reports": [str(path.relative_to(ROOT)).replace("\\", "/") for path in legs_files],
        "head_negative_reports": [
            "data/processed/retrieval_query_only_v68_head_fold01_coverage_k10_b035.json",
            "data/processed/retrieval_query_only_v68_head_fold01_spatial_k10_b035.json",
            "data/processed/retrieval_query_only_v68_head_fold01_guard_k100_d030_e040_t030_m001.json",
            "data/processed/retrieval_query_only_v68_head_fold02_guard_k100_d030_e040_t030_m001.json",
            "data/processed/retrieval_query_only_v68_head_fold03_guard_k100_d030_e040_t030_m001.json",
        ],
        "e2e_report": "data/debug/glamour-match-e2e-v68-five-slots/results.json",
        "test_command": "D:/Anaconda/envs/ffxiv-prism/python.exe -m unittest discover -s tests -v",
        "test_result": "144 tests passed",
    }
    source_path = PROCESSED / "recognition_optimization_report_v68_sources.json"
    source_path.write_text(
        json.dumps(source_inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sources = [
        {
            "id": "strict_eval",
            "label": "Strict three-fold evaluation summary",
            "path": "data/processed/recognition_optimization_current_best_v68_summary.json",
            "query": {
                "engine": "duckdb",
                "sql": "SELECT r.slot, r.eligible, r.top1, r.top10, r.top1000, r.method FROM read_json_auto('data/processed/recognition_optimization_current_best_v68_summary.json') AS s, UNNEST(s.records) AS t(r)",
                "description": "Reads the registered v68 strict pooled metrics.",
                "executed_at": generated_at,
            },
        },
        {
            "id": "experiment_inventory",
            "label": "v68 experiment and source inventory",
            "path": "data/processed/recognition_optimization_report_v68_sources.json",
            "query": {
                "engine": "duckdb",
                "sql": "SELECT * FROM read_json_auto('data/processed/recognition_optimization_report_v68_sources.json')",
                "description": "Reads the v68 experiment inventory; report datasets are derived by the checked-in Python builder.",
                "executed_at": generated_at,
            },
        },
        {
            "id": "e2e",
            "label": "Terra five-slot end-to-end result",
            "path": "data/debug/glamour-match-e2e-v68-five-slots/results.json",
            "query": {
                "engine": "duckdb",
                "sql": "SELECT s.slot, s.candidates[1].name_en AS top1_candidate, s.ranking_quality.level AS quality, s.top_score_margin FROM read_json_auto('data/debug/glamour-match-e2e-v68-five-slots/results.json') AS r, UNNEST(r.slots) AS t(s)",
                "description": "Reads the reviewed Terra v68 five-slot output.",
                "executed_at": generated_at,
            },
        },
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "FFXIV Prism Recognition Optimization v68",
            "description": "Strict retrieval metrics, visual failure diagnostics, and promoted v68 changes.",
            "generatedAt": generated_at,
            "filters": [],
            "cards": [
                {
                    "id": "legs_card",
                    "description": "LEGS strict pooled Top-1 with prior RRF-only comparison.",
                    "dataset": "headline_metrics",
                    "sourceId": "strict_eval",
                    "filter": {"metric": "LEGS"},
                    "metrics": [
                        {"label": "LEGS Top-1", "field": "value", "format": "percent"},
                        {"label": "RRF-only", "field": "comparison", "format": "percent"},
                    ],
                },
                {
                    "id": "head_card",
                    "description": "HEAD strict pooled Top-1 after evaluator correction.",
                    "dataset": "headline_metrics",
                    "sourceId": "strict_eval",
                    "filter": {"metric": "HEAD"},
                    "metrics": [
                        {"label": "HEAD Top-1", "field": "value", "format": "percent"},
                        {"label": "Old registered value", "field": "comparison", "format": "percent"},
                    ],
                },
                {
                    "id": "best_card",
                    "description": "Highest current strict pooled Top-1 across equipment slots.",
                    "dataset": "headline_metrics",
                    "sourceId": "strict_eval",
                    "filter": {"metric": "BEST"},
                    "metrics": [
                        {"label": "Best strict Top-1", "field": "value", "format": "percent"},
                        {"label": "Target", "field": "comparison", "format": "percent"},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "slot_metrics_chart",
                    "title": "Current strict retrieval by slot",
                    "subtitle": "Pooled three-fold Top-1 and Top-10; denominators vary from 899 to 987 eligible cases.",
                    "type": "bar",
                    "dataset": "slot_metric_rows",
                    "sourceId": "strict_eval",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "slot", "type": "nominal", "label": "Equipment slot"},
                        "y": {"field": "value", "type": "quantitative", "label": "Recall"},
                        "color": {"field": "metric", "type": "nominal", "label": "Metric"},
                        "tooltip": [
                            {"field": "value", "type": "quantitative", "label": "Recall", "format": "percent"}
                        ],
                    },
                }
            ],
            "tables": [
                {
                    "id": "slot_table",
                    "title": "Strict pooled metrics and promoted method",
                    "subtitle": "Exact denominators and current registered retrieval configuration.",
                    "dataset": "slot_rows",
                    "sourceId": "strict_eval",
                    "columns": [
                        {"field": "slot", "label": "Slot", "type": "text"},
                        {"field": "eligible", "label": "Eligible", "format": "number"},
                        {"field": "top1", "label": "Top-1", "format": "percent"},
                        {"field": "top10", "label": "Top-10", "format": "percent"},
                        {"field": "top1000", "label": "Top-1000", "format": "percent"},
                        {"field": "method", "label": "Method", "type": "text"},
                    ],
                },
                {
                    "id": "legs_fold_table",
                    "title": "LEGS spatial reranker validation by fold",
                    "subtitle": "Frozen blend=0.35 and rerank-k=10; Top-10 candidate membership is unchanged.",
                    "dataset": "legs_folds",
                    "sourceId": "experiment_inventory",
                    "columns": [
                        {"field": "fold", "label": "Fold", "format": "number"},
                        {"field": "eligible", "label": "Eligible", "format": "number"},
                        {"field": "baseline_top1", "label": "RRF Top-1", "format": "percent"},
                        {"field": "spatial_top1", "label": "Spatial Top-1", "format": "percent"},
                        {"field": "top10", "label": "Top-10", "format": "percent"},
                    ],
                },
                {
                    "id": "head_fold_table",
                    "title": "HEAD corrected evaluation by fold",
                    "subtitle": "The evaluator now applies head_upper_accessory to head_face_accessory routing.",
                    "dataset": "head_folds",
                    "sourceId": "experiment_inventory",
                    "columns": [
                        {"field": "fold", "label": "Fold", "format": "number"},
                        {"field": "eligible", "label": "Eligible", "format": "number"},
                        {"field": "top1", "label": "Top-1", "format": "percent"},
                        {"field": "top10", "label": "Top-10", "format": "percent"},
                        {"field": "top1000", "label": "Top-1000", "format": "percent"},
                    ],
                },
                {
                    "id": "head_experiment_table",
                    "title": "HEAD auxiliary reranker decisions",
                    "subtitle": "Negative results are retained because global visual heuristics reduced strict retrieval.",
                    "dataset": "head_experiments",
                    "sourceId": "experiment_inventory",
                    "columns": [
                        {"field": "experiment", "label": "Experiment", "type": "text"},
                        {"field": "scope", "label": "Scope", "type": "text"},
                        {"field": "top1", "label": "Top-1", "format": "percent"},
                        {"field": "top10", "label": "Top-10", "format": "percent"},
                        {"field": "decision", "label": "Decision", "type": "text"},
                    ],
                },
                {
                    "id": "e2e_table",
                    "title": "Terra v68 five-slot end-to-end output",
                    "subtitle": "Single reviewed image; this table is qualitative evidence, not an accuracy estimate.",
                    "dataset": "e2e_rows",
                    "sourceId": "e2e",
                    "columns": [
                        {"field": "slot", "label": "Slot", "type": "text"},
                        {"field": "top1_candidate", "label": "Top-1 candidate", "type": "text"},
                        {"field": "quality", "label": "Ranking quality", "type": "text"},
                        {"field": "top_margin", "label": "Top margin", "format": "number"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": "## Technical summary\n\n- **LEGS is the only promoted ranking change in v68.** A grayscale DINOv2 row-profile reranker lifts strict pooled Top-1 from 9.11% to 10.58% while preserving the 26.39% Top-10 set.\n- **The Terra length error is fixed.** Faire Kohakama now outranks Orison Skirt because the spatial signal captures where the stocking boundary occurs.\n- **HEAD remains the largest qualitative risk.** Coverage and DINOv2 spatial rerankers improved the reviewed Terra shortlist but regressed strict folds, so they were rejected; low-margin HEAD outputs now carry a visible warning.\n- **HEAD's registered benchmark was corrected, not model-improved.** The previous evaluator ignored query/candidate crop arguments; the matching configuration actually scores 23.25% Top-1 and 44.16% Top-10 over 899 strict cases."
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["legs_card", "head_card", "best_card"]},
                {
                    "id": "current_result",
                    "type": "markdown",
                    "body": "## Strict performance remains far below automatic-identification quality\n\nThe current system is usable as a candidate-retrieval assistant, not as a reliable automatic identifier. BODY remains strongest at 45.90% Top-1; all slots are still well below the 80% target. Top-10 is the more decision-useful product metric today because the interface exposes candidate lists for human selection."
                },
                {"id": "slot_chart", "type": "chart", "chartId": "slot_metrics_chart"},
                {"id": "slot_metrics", "type": "table", "tableId": "slot_table"},
                {
                    "id": "legs_result",
                    "type": "markdown",
                    "sourceId": "experiment_inventory",
                    "body": "## Vertical topology fixes the LEGS stocking-length failure\n\nRRF treated five overlapping crops as independent votes and ignored cosine-score magnitude, so Orison Skirt accumulated two rank-1 votes even though its calf-high socks mismatched the image. The promoted reranker averages grayscale DINOv2 patch tokens by vertical row, then blends spatial and retrieval percentiles only within the existing Top-10. This keeps recall membership fixed while allowing garment-boundary location to correct order."
                },
                {"id": "legs_folds", "type": "table", "tableId": "legs_fold_table"},
                {
                    "id": "head_result",
                    "type": "markdown",
                    "sourceId": "experiment_inventory",
                    "body": "## HEAD failure is a visibility and domain-alignment problem\n\nOn Terra, the 66×62 upper-head crop is dominated by green hair and character ears. DINOv2 promotes large hoods; SigLIP2 promotes horns. The true Thavnairian Headdress is a small forehead ornament partly hidden by hair. Human-parsing coverage cannot distinguish all fantasy hoods, and DINOv2 spatial matching moves the correct item into a visual shortlist but still ranks several hoods above it. Because these heuristics regress strict folds, v68 only adds a low-confidence warning when the top two HEAD scores differ by less than 0.01."
                },
                {"id": "head_folds", "type": "table", "tableId": "head_fold_table"},
                {"id": "head_experiments", "type": "table", "tableId": "head_experiment_table"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "body": "## Scope and metric definitions\n\n**Top-k recall** counts an eligible slot case when any accepted piece ID or registered same-appearance name is present in the first k ranked candidates. **Strict three-fold pooled** means held-out equipment identities are evaluated once across three disjoint folds, then hits and denominators are summed before computing recall. Denominators differ by slot because labels, candidate coverage, and usable crops differ. The fixed candidate image library is unchanged."
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": "## Validation and implementation\n\nThe LEGS blend and rerank window were selected on fold 1, frozen, then applied unchanged to folds 2 and 3. All three folds improved Top-1 (33→40, 25→30, 29→31). The HEAD evaluator correction was independently rerun on all three folds with explicit `head_upper_accessory` query filtering and `head_face_accessory` candidate routing. The production CLI now validates the spatial index in `--check`, exposes retrieval and spatial evidence in JSON, and passes 144 automated tests."
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## Limitations and robustness checks\n\n- Strict label recall is necessary but does not fully measure visual similarity when the true item is occluded.\n- The Terra end-to-end example is one reviewed image and cannot estimate population accuracy.\n- HANDS remains extremely weak at 2.12% Top-1; no v68 experiment addressed that representation gap.\n- The HEAD DINOv3 experiment is blocked until the gated Hugging Face model license is accepted and the local environment is authenticated.\n- No rejected HEAD reranker is active in production; only the low-margin warning was promoted."
                },
                {"id": "e2e", "type": "table", "tableId": "e2e_table"},
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": "## Recommended next steps\n\n1. After Hugging Face authorization, build a DINOv3 high-resolution patch index for HEAD and compare it against the corrected 23.25%/44.16% baseline.\n2. Add a pairwise visual-preference benchmark alongside identity recall, starting with stocking length, hood coverage, horns, circlets, and small forehead ornaments.\n3. Extend the spatial-profile approach to FEET only after a frozen-fold ablation; do not assume the LEGS gain transfers.\n4. Prioritize HANDS representation work because candidate ordering cannot compensate for its current recall ceiling."
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": "## Further questions\n\n- Can DINOv3 distinguish character ears/hair from equipped horns and hoods at the patch level?\n- What proportion of user images contain equipment that is too occluded for identity retrieval but still supports a useful visual-similarity recommendation?\n- Should the product expose identity candidates and visual-style candidates as separate lists when the two objectives disagree?"
                }
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline_metrics": [
                    {"metric": "LEGS", "value": legs["recall"]["1"], "comparison": legs_baseline["recall"]["1"]},
                    {"metric": "HEAD", "value": head["recall"]["1"], "comparison": 0.1757508342602892},
                    {"metric": "BEST", "value": by_slot["BODY"]["top1"], "comparison": 0.8},
                ],
                "slot_rows": slot_rows,
                "slot_metric_rows": slot_metric_rows,
                "legs_folds": legs_fold_rows,
                "head_folds": head_fold_rows,
                "head_experiments": head_experiments,
                "e2e_rows": e2e_rows,
            },
        },
        "sources": sources,
    }
    artifact_path = DOCS / "recognition_optimization_report_v68_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "sources": str(source_path),
                "artifact": str(artifact_path),
                "head": head,
                "legs": legs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
