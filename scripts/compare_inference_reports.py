"""Compare complete candidate order, not just a known target's rank."""
import argparse
import json
from pathlib import Path


def compare(reports):
    first = reports[0]
    baseline = {s["slot"]: [c["piece_id"] for c in s["candidates"]] for s in first["slots"]}
    comparisons = []
    for other in reports[1:]:
        slots = {s["slot"]: [c["piece_id"] for c in s["candidates"]] for s in other["slots"]}
        differences = {slot: {"reference": baseline.get(slot), "actual": slots.get(slot)}
                       for slot in sorted(baseline.keys() | slots.keys())
                       if baseline.get(slot) != slots.get(slot)}
        settings = {k: [first.get(k), other.get(k)] for k in ("top_k", "candidate_k", "profile", "gender_filter")
                    if first.get(k) != other.get(k)}
        comparisons.append({"reference_device": first["device"], "device": other["device"],
                            "different_slots": differences, "different_settings": settings,
                            "ok": not differences and not settings})
    return {"ok": all(c["ok"] for c in comparisons), "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.reports) < 2:
        parser.error("Provide at least two reports")
    result = compare([json.loads(p.read_text("utf-8")) for p in args.reports])
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", "utf-8")
    print(output)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
