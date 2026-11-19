#!/usr/bin/env python3
"""Create a Humble-compatible sqlite3 rosbag directory from newer metadata."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


HUMBLE_SETUP = Path("/opt/ros/humble/setup.bash")
METADATA_NAME = "metadata.yaml"

QOS_ENUM_REPLACEMENTS = {
    "history": {
        "system_default": "0",
        "keep_last": "1",
        "keep_all": "2",
        "unknown": "3",
    },
    "reliability": {
        "system_default": "0",
        "reliable": "1",
        "best_effort": "2",
        "unknown": "3",
    },
    "durability": {
        "system_default": "0",
        "transient_local": "1",
        "volatile": "2",
        "unknown": "3",
    },
    "liveliness": {
        "system_default": "0",
        "automatic": "1",
        "manual_by_topic": "3",
        "unknown": "4",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a sqlite3 rosbag recorded/converted on a newer ROS 2 distro "
            "so it can be opened by ros2 bag on Humble."
        )
    )
    parser.add_argument(
        "bag_dir",
        type=Path,
        help="Path to the source rosbag directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help=(
            "Output directory for the Humble-compatible bag. "
            "Defaults to <bag_dir>_humble."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite metadata.yaml inside the source bag directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing output directory before regenerating it.",
    )
    return parser.parse_args()


def ensure_humble_available() -> None:
    if not HUMBLE_SETUP.is_file():
        raise SystemExit(
            f"Humble environment not found: {HUMBLE_SETUP}\n"
            "Install ROS 2 Humble or adjust the script."
        )


def validate_source_dir(bag_dir: Path) -> None:
    if not bag_dir.is_dir():
        raise SystemExit(f"Bag directory does not exist: {bag_dir}")

    db3_files = sorted(bag_dir.glob("*.db3"))
    if not db3_files:
        raise SystemExit(f"No .db3 files found in: {bag_dir}")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    bag_dir = args.bag_dir.resolve()
    if args.in_place:
        if args.output_dir is not None:
            raise SystemExit("--output-dir cannot be used together with --in-place")
        return bag_dir

    if args.output_dir is not None:
        return args.output_dir.resolve()
    return bag_dir.with_name(f"{bag_dir.name}_humble")


def prepare_output_dir(source_dir: Path, output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise SystemExit(
                f"Output directory already exists: {output_dir}\n"
                "Use --force to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the original bag files via symlinks to avoid copying large db3 files.
    for entry in source_dir.iterdir():
        if entry.name == METADATA_NAME:
            continue
        dest = output_dir / entry.name
        os.symlink(entry.resolve(), dest)


def run_reindex(target_dir: Path) -> None:
    metadata_path = target_dir / METADATA_NAME
    if metadata_path.exists():
        metadata_path.unlink()

    command = (
        f"source {HUMBLE_SETUP} && "
        f"ros2 bag reindex -s sqlite3 {shlex.quote(str(target_dir))}"
    )
    subprocess.run(
        ["/bin/bash", "-lc", command],
        check=True,
    )


def normalize_qos_enums(metadata_path: Path) -> None:
    text = metadata_path.read_text(encoding="utf-8")
    for key, mapping in QOS_ENUM_REPLACEMENTS.items():
        for label, number in mapping.items():
            text = text.replace(f"{key}: {label}", f"{key}: {number}")
    metadata_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_dir = args.bag_dir.resolve()
    output_dir = resolve_output_dir(args)

    ensure_humble_available()
    validate_source_dir(source_dir)

    if args.in_place:
        metadata_path = output_dir / METADATA_NAME
        if metadata_path.exists():
            backup_path = output_dir / f"{METADATA_NAME}.pre_humble_compat.bak"
            shutil.copy2(metadata_path, backup_path)
        run_reindex(output_dir)
    else:
        prepare_output_dir(source_dir, output_dir, args.force)
        run_reindex(output_dir)

    metadata_path = output_dir / METADATA_NAME
    normalize_qos_enums(metadata_path)

    print(f"Prepared Humble-compatible bag: {output_dir}")
    print(f"Play it with: ros2 bag play {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
