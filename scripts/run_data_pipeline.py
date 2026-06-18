from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


STAGES: dict[str, list[list[str]]] = {
    "discover": [["scripts/discover_eorzea_gearsets.py"]],
    "collect-details": [["scripts/collect_eorzea_gearset_details.py", "--resume", "--workers", "12"]],
    "build-image-manifest": [["scripts/build_eorzea_image_url_manifest_from_detail_cache.py"]],
    "cafemaker-zh": [
        ["scripts/download_cafemaker_item_names.py"],
        ["scripts/filter_cafemaker_item_names_for_gear_pieces.py"],
        ["scripts/enrich_gear_pieces_with_zh_names.py"],
    ],
    "huiji-missing-zh": [["scripts/search_huiji_item_zh_name.py", "--batch-missing", "--retry-failed"]],
    "build-index": [["scripts/build_eorzea_dataset_index.py"]],
}


DEFAULT_STAGES = [
    "collect-details",
    "build-image-manifest",
    "cafemaker-zh",
    "huiji-missing-zh",
    "build-index",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local FFXIV Prism dataset pipeline.")
    parser.add_argument(
        "--stage",
        action="append",
        choices=sorted(STAGES),
        help="Stage to run. May be repeated. Defaults to the refresh pipeline.",
    )
    parser.add_argument("--all", action="store_true", help="Run discovery plus the default refresh pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--huiji-proxy",
        default=None,
        help="Optional proxy passed only to the Huiji Selenium stage.",
    )
    parser.add_argument(
        "--huiji-user-data-dir",
        default=None,
        help="Optional persistent Chrome profile directory for the Huiji Selenium stage.",
    )
    args = parser.parse_args()

    stage_names = ["discover", *DEFAULT_STAGES] if args.all else (args.stage or DEFAULT_STAGES)
    for stage_name in stage_names:
        for command in commands_for_stage(stage_name, args):
            print(format_command(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


def commands_for_stage(stage_name: str, args: argparse.Namespace) -> list[list[str]]:
    commands = [[PYTHON, *command] for command in STAGES[stage_name]]
    if stage_name == "huiji-missing-zh":
        for command in commands:
            if args.huiji_proxy:
                command.extend(["--proxy", args.huiji_proxy])
            if args.huiji_user_data_dir:
                command.extend(["--user-data-dir", args.huiji_user_data_dir])
    return commands


def format_command(command: list[str]) -> str:
    return " ".join(quote_arg(part) for part in command)


def quote_arg(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return f'"{value}"'
    return value


if __name__ == "__main__":
    raise SystemExit(main())
