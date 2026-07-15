#!/usr/bin/env python3
"""Export image frames from a rosbag2 sqlite3 bag."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - depends on ROS environment
    raise SystemExit(
        "ROS 2 Python modules are unavailable. Run this script in a sourced ROS 2 "
        f"environment. Original error: {exc}"
    ) from exc


DEFAULT_TOPIC = "/camera/camera/color/image_raw"
SUPPORTED_8BIT_ENCODINGS = {
    "bgr8",
    "rgb8",
    "bgra8",
    "rgba8",
    "mono8",
    "8uc1",
    "8uc3",
    "8uc4",
}
SUPPORTED_16BIT_ENCODINGS = {
    "mono16",
    "16uc1",
    "16sc1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export image frames from a rosbag2 bag into image files."
    )
    parser.add_argument(
        "bag_dir",
        type=Path,
        help="Path to the rosbag2 directory, for example data_humble.",
    )
    parser.add_argument(
        "-t",
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"Image topic to export. Default: {DEFAULT_TOPIC}",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory. Default: <bag_dir>/rgb_frames",
    )
    parser.add_argument(
        "--format",
        choices=("png", "jpg"),
        default="png",
        help="Image file format. Use png for lossless export.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Only save every Nth frame. Default: 1",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of frames to save. Default: 0 means no limit.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N matching messages before exporting.",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="Filename prefix. Default: frame",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.bag_dir.is_dir():
        raise SystemExit(f"Bag directory does not exist: {args.bag_dir}")
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.start_index < 0:
        raise SystemExit("--start-index must be >= 0")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (args.bag_dir / "rgb_frames").resolve()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise SystemExit(f"Output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not overwrite:
            raise SystemExit(
                f"Output directory is not empty: {output_dir}\n"
                "Use --overwrite or choose a different --output-dir."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def open_reader(bag_dir: Path, topic: str) -> tuple[rosbag2_py.SequentialReader, str]:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    topic_types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    if topic not in topic_types:
        available = "\n".join(f"- {name} ({msg_type})" for name, msg_type in topic_types.items())
        raise SystemExit(
            f"Topic not found in bag: {topic}\n"
            f"Available topics:\n{available}"
        )

    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    return reader, topic_types[topic]


def decode_8bit_image(msg, encoding: str) -> np.ndarray:
    channels_by_encoding = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
        "8uc1": 1,
        "8uc3": 3,
        "8uc4": 4,
    }
    channels = channels_by_encoding[encoding]
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    trimmed = row[:, : msg.width * channels]
    image = trimmed.reshape(msg.height, msg.width, channels) if channels > 1 else trimmed.reshape(msg.height, msg.width)

    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return image.copy()


def decode_16bit_image(msg, encoding: str) -> np.ndarray:
    channels = 1
    row = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)
    trimmed = row[:, : msg.width * channels]
    image = trimmed.reshape(msg.height, msg.width)
    return image.copy()


def decode_image(msg) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding in SUPPORTED_8BIT_ENCODINGS:
        return decode_8bit_image(msg, encoding)
    if encoding in SUPPORTED_16BIT_ENCODINGS:
        return decode_16bit_image(msg, encoding)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def make_filename(prefix: str, index: int, msg, extension: str) -> str:
    stamp = msg.header.stamp
    return (
        f"{prefix}_{index:06d}_{stamp.sec}_{stamp.nanosec:09d}.{extension}"
    )


def save_image(image: np.ndarray, output_path: Path, image_format: str) -> None:
    if image.dtype == np.uint16 and image_format != "png":
        raise ValueError("16-bit images must be exported as PNG.")
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write image: {output_path}")


def export_frames(args: argparse.Namespace) -> int:
    output_dir = resolve_output_dir(args)
    prepare_output_dir(output_dir, args.overwrite)

    reader, topic_type = open_reader(args.bag_dir.resolve(), args.topic)
    message_cls = get_message(topic_type)

    matched = 0
    saved = 0

    while reader.has_next():
        topic_name, raw_data, _timestamp = reader.read_next()
        if topic_name != args.topic:
            continue

        matched += 1
        if matched <= args.start_index:
            continue

        relative_index = matched - args.start_index - 1
        if relative_index % args.stride != 0:
            continue

        msg = deserialize_message(raw_data, message_cls)
        try:
            image = decode_image(msg)
        except ValueError as exc:
            raise SystemExit(f"Failed to decode frame {matched}: {exc}") from exc

        output_path = output_dir / make_filename(args.prefix, saved, msg, args.format)
        save_image(image, output_path, args.format)
        saved += 1

        if args.limit and saved >= args.limit:
            break

    if matched == 0:
        raise SystemExit(f"No messages found on topic: {args.topic}")

    print(f"Exported {saved} frames from {args.topic}")
    print(f"Output directory: {output_dir}")
    return saved


def main() -> int:
    args = parse_args()
    validate_args(args)
    export_frames(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
