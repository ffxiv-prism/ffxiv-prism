from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restrict a quality manifest to an explicit model metadata scope."
    )
    parser.add_argument("--examples", type=Path, default=Path("data/processed/glamour_examples.jsonl"))
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--model-races", nargs="+")
    parser.add_argument("--model-genders", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    examples = read_jsonl(args.examples)
    quality_rows = read_jsonl(args.quality_manifest)
    metadata = {
        str(row["example_id"]): {
            "model_race": str(row.get("model_race") or "").strip(),
            "model_gender": str(row.get("model_gender") or "").strip().lower(),
        }
        for row in examples
    }
    allowed_races = set(args.model_races or [])
    allowed_genders = {value.lower() for value in (args.model_genders or [])}

    scoped: list[dict] = []
    excluded = Counter()
    for source in quality_rows:
        row = dict(source)
        if row.get("passed"):
            values = metadata.get(str(row["example_id"]), {})
            race = values.get("model_race", "")
            gender = values.get("model_gender", "")
            if allowed_races and race not in allowed_races:
                excluded["model_race"] += 1
                row["passed"] = False
            if allowed_genders and gender not in allowed_genders:
                excluded["model_gender"] += 1
                row["passed"] = False
            if not row["passed"]:
                reasons = list(row.get("rejection_reasons", []))
                if "outside_model_metadata_scope" not in reasons:
                    reasons.append("outside_model_metadata_scope")
                row["rejection_reasons"] = reasons
        scoped.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in scoped:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    accepted = sum(bool(row.get("passed")) for row in scoped)
    summary = {
        "accepted": accepted,
        "examples": len(scoped),
        "excluded_reason_counts": dict(sorted(excluded.items())),
        "model_genders": sorted(allowed_genders),
        "model_races": sorted(allowed_races),
        "output": args.output.as_posix(),
        "source_quality_manifest": args.quality_manifest.as_posix(),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"examples={len(scoped)} accepted={accepted} output={args.output}")


if __name__ == "__main__":
    main()
