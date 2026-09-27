"""Export an existing original report without running model inference again."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ffxiv_prism.matching.report_export import render_report_png  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to report.html")
    args = parser.parse_args()
    print(render_report_png(args.report))
