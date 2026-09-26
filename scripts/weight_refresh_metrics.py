"""Pure metrics and conservative promotion decisions for weight refreshes."""
from collections import defaultdict

CUTOFFS = (1, 5, 10, 20, 100, 1000)


def metrics(cases, rank_key):
    count = len(cases)
    hits = {str(k): sum(c[rank_key] is not None and c[rank_key] <= k for c in cases) for k in CUTOFFS}
    return dict(eligible_case_count=count, hits=hits, recall={k: v / count if count else 0 for k, v in hits.items()},
                mrr=sum(1 / c[rank_key] if c[rank_key] else 0 for c in cases) / count if count else 0)


def compare_variants(cases, variants):
    groups = defaultdict(list)
    for case in cases:
        for name in ("all", f"fold:{case['fold']}", f"gender:{case['gender']}", f"race:{case['race']}"):
            groups[name].append(case)
    table = {name: {v: metrics(rows, v) for v in variants} for name, rows in groups.items()}
    decisions = {}
    for variant in variants:
        if variant == "baseline":
            continue
        regressions = [
            dict(group=g, cutoff=k, baseline=t["baseline"]["hits"][str(k)], candidate=t[variant]["hits"][str(k)])
            for g, t in table.items() for k in (1, 10)
            if t[variant]["hits"][str(k)] < t["baseline"]["hits"][str(k)]
        ]
        base, new = table["all"]["baseline"], table["all"][variant]
        improves = any(new["hits"][str(k)] > base["hits"][str(k)] for k in (1, 10))
        decisions[variant] = dict(passes=not regressions and improves, improves=improves, regressions=regressions)
    eligible = [v for v in decisions if decisions[v]["passes"]]
    selected = max(eligible, key=lambda v: (table["all"][v]["hits"]["10"], table["all"][v]["hits"]["1"], table["all"][v]["mrr"])) if eligible else "baseline"
    return dict(selected=selected, decisions=decisions, groups=table)
