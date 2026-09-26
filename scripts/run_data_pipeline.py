from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


STAGES: dict[str, list[list[str]]] = {
    "discover": [["scripts/discover_eorzea_gearsets_by_model.py"]],
    "collect-details": [["scripts/collect_eorzea_gearset_details.py", "--resume", "--workers", "12"]],
    "build-image-manifest": [
        [
            "scripts/build_eorzea_image_url_manifest_from_list_cache.py",
            "--resume",
        ]
    ],
    "cafemaker-zh": [
        ["scripts/download_cafemaker_item_names.py"],
        ["scripts/download_cafemaker_equip_slot_categories.py"],
        ["scripts/filter_cafemaker_item_names_for_gear_pieces.py"],
        ["scripts/enrich_gear_pieces_with_zh_names.py"],
        ["scripts/enrich_gear_pieces_with_lodestone_equip_notes.py"],
    ],
    "cafemaker-enrich-local": [
        ["scripts/filter_cafemaker_item_names_for_gear_pieces.py"],
        ["scripts/enrich_gear_pieces_with_zh_names.py"],
        ["scripts/enrich_gear_pieces_with_lodestone_equip_notes.py"],
    ],
    "huiji-missing-zh": [["scripts/search_huiji_item_zh_name.py", "--batch-missing", "--retry-failed"]],
    "build-index": [["scripts/build_eorzea_dataset_index.py"]],
    "validate-images": [["scripts/validate_eorzea_images.py"]],
    "build-sqlite": [["scripts/build_eorzea_sqlite.py"]],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local FFXIV Prism dataset pipeline.")
    parser.add_argument(
        "--stage",
        action="append",
        choices=sorted(STAGES),
        help="Internal maintenance stage. Without --stage, use the unified image update workflow.",
    )
    parser.add_argument("--all", action="store_true", help="Compatibility alias for the unified image update workflow.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--refresh-source",
        action="store_true",
        help=(
            "Bypass cached Eorzea list pages during discovery. Without this "
            "flag, discovery remains reproducible from the existing cache."
        ),
    )
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
    parser.add_argument(
        "--browser-verification",
        action="store_true",
        help=(
            "Use a visible persistent Chrome for Eorzea discovery/details and "
            "wait for manual Cloudflare verification."
        ),
    )
    parser.add_argument(
        "--browser-profile-dir",
        default=None,
        help="Persistent Chrome profile for Eorzea browser verification.",
    )
    parser.add_argument(
        "--browser-wait-seconds",
        type=float,
        default=300.0,
        help="Maximum manual-verification wait per Eorzea page.",
    )
    parser.add_argument("--discovery-mode", choices=("auto", "incremental", "full"), default="auto")
    parser.add_argument("--discovery-head-pages", type=int, default=3)
    parser.add_argument("--discovery-max-head-pages", type=int, default=8)
    parser.add_argument("--discovery-history-pages", type=int, default=3)
    args = parser.parse_args()

    if args.all or not args.stage:
        command = [PYTHON, "scripts/run_lifecycle_pipeline.py", "--workflow", "sample-library",
                   "--source-mode", "refresh", "--candidate-mode", "extend"]
        command.extend(discovery_arguments(args))
        if args.dry_run:
            command.append("--dry-run")
        if args.browser_verification:
            command.extend(["--browser-verification", "--browser-wait-seconds", str(args.browser_wait_seconds)])
            if args.browser_profile_dir:
                command.extend(["--browser-profile-dir", args.browser_profile_dir])
        for flag, value in (("--huiji-proxy", args.huiji_proxy), ("--huiji-user-data-dir", args.huiji_user_data_dir)):
            if value:
                command.extend([flag, value])
        print("Using the unified image-library update workflow.", flush=True)
        return subprocess.run(command, cwd=PROJECT_ROOT).returncode
    stage_names = args.stage
    for stage_name in stage_names:
        for command in commands_for_stage(stage_name, args):
            print(format_command(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


def commands_for_stage(stage_name: str, args: argparse.Namespace) -> list[list[str]]:
    commands = [[PYTHON, *command] for command in STAGES[stage_name]]
    if stage_name == "build-image-manifest" and args.refresh_source:
        for command in commands:
            command.append("--retry-unavailable")
    if stage_name == "discover":
        for command in commands:
            command.extend(discovery_arguments(args, collector=True))
    if stage_name == "discover" and args.refresh_source:
        for command in commands:
            command.append("--refresh")
    if stage_name in {"discover", "collect-details", "build-image-manifest"} and args.browser_verification:
        for command in commands:
            command.extend(
                [
                    "--browser-verification",
                    "--browser-wait-seconds",
                    str(args.browser_wait_seconds),
                ]
            )
            if args.browser_profile_dir:
                command.extend(["--browser-profile-dir", args.browser_profile_dir])
    if stage_name == "huiji-missing-zh":
        for command in commands:
            if args.huiji_proxy:
                command.extend(["--proxy", args.huiji_proxy])
            if args.huiji_user_data_dir:
                command.extend(["--user-data-dir", args.huiji_user_data_dir])
    return commands


def discovery_arguments(args: argparse.Namespace, *, collector: bool = False) -> list[str]:
    result = ["--discovery-mode", getattr(args, "discovery_mode", "auto")]
    for name, default in (("head-pages", 3), ("max-head-pages", 8), ("history-pages", 3)):
        flag = "--" + ("" if collector else "discovery-") + name
        result.extend([flag, str(getattr(args, "discovery_" + name.replace("-", "_"), default))])
    return result


def format_command(command: list[str]) -> str:
    return " ".join(quote_arg(part) for part in command)


def quote_arg(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return f'"{value}"'
    return value


if __name__ == "__main__":
    raise SystemExit(main())

