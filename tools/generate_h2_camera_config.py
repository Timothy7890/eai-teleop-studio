#!/usr/bin/env python3
"""Discover Orbbec V4L2 nodes and generate teleimager camera config.

Typical workflow on the camera host:

    python tools/generate_h2_camera_config.py --list

    python tools/generate_h2_camera_config.py \
      --head-serial CP0T263000BE \
      --torso-serial CPCBC530002E \
      --left-serial CP0F463000HS \
      --right-serial CP06563000E6 \
      --depth all \
      --color-size 640x480 \
      --head-depth-size 848x480 \
      -o teleop/teleimager/cam_config_server.yaml

Use --depth none for RGB-only config, --depth head for head RGB-D only, and
--depth all for head/torso/left-wrist/right-wrist RGB-D templates. Generated
RGB-D streams are disabled by default; set the desired *_rgbd_camera.enable_zmq
to true when you want to publish one.

The generated config uses stable selectors:
serial_number + usb_interface + video_index.
It never relies on /dev/videoX, because those numbers drift after reboot or USB
re-enumeration.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CAMERA_ORDER = ("head_camera", "torso_camera", "left_wrist_camera", "right_wrist_camera")
CAMERA_LABELS = {
    "head_camera": "head",
    "torso_camera": "torso",
    "left_wrist_camera": "left_wrist",
    "right_wrist_camera": "right_wrist",
}

COLOR_PORTS = {
    "head_camera": (55555, 60001),
    "torso_camera": (55556, 60002),
    "left_wrist_camera": (55557, 60003),
    "right_wrist_camera": (55558, 60004),
}

RGBD_PORTS = {
    "head_rgbd_camera": 55560,
    "left_wrist_rgbd_camera": 55564,
    "right_wrist_rgbd_camera": 55565,
    "torso_rgbd_camera": 55566,
}

COLOR_FORMATS = ("YUYV", "MJPG")
DEPTH_FORMATS = ("Z16",)
BAD_COLOR_FORMATS = ("GREY", "BA81", "Z16")


@dataclass(frozen=True)
class VideoNode:
    path: str
    number: int
    index: int | None
    usb_interface: str | None
    serial_number: str | None
    name: str
    formats: tuple[str, ...]
    sizes_by_format: dict[str, tuple[tuple[int, int], ...]]
    usb_speed: str | None


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


def walk_parents(path: Path, limit: int = 12) -> Iterable[Path]:
    current = path
    for _ in range(limit):
        yield current
        if current.parent == current:
            break
        current = current.parent


def usb_serial_from_device_path(device_path: Path) -> str | None:
    for current in walk_parents(device_path):
        serial = read_text(current / "serial")
        if serial:
            return serial
    return None


def usb_speed_from_device_path(device_path: Path) -> str | None:
    for current in walk_parents(device_path):
        speed = read_text(current / "speed")
        if speed:
            return f"{speed}Mbps"
    return None


def run_v4l2(video_path: str) -> str:
    try:
        return subprocess.check_output(
            ["v4l2-ctl", "-d", video_path, "--list-formats-ext"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ""


def parse_formats(output: str) -> tuple[tuple[str, ...], dict[str, tuple[tuple[int, int], ...]]]:
    formats: list[str] = []
    sizes_by_format: dict[str, list[tuple[int, int]]] = {}
    current_format: str | None = None

    for line in output.splitlines():
        fmt_match = re.search(r"\[\d+\]: '([^']+)'", line)
        if fmt_match:
            current_format = fmt_match.group(1).strip()
            if current_format not in formats:
                formats.append(current_format)
                sizes_by_format[current_format] = []
            continue

        size_match = re.search(r"Size:\s+Discrete\s+(\d+)x(\d+)", line)
        if size_match and current_format:
            size = (int(size_match.group(1)), int(size_match.group(2)))
            if size not in sizes_by_format[current_format]:
                sizes_by_format[current_format].append(size)

    frozen_sizes = {fmt: tuple(sizes) for fmt, sizes in sizes_by_format.items()}
    return tuple(formats), frozen_sizes


def discover_video_nodes() -> list[VideoNode]:
    root = Path("/sys/class/video4linux")
    if not root.exists():
        return []

    nodes: list[VideoNode] = []
    for sysfs_path in sorted(root.glob("video*"), key=lambda p: int(p.name[5:])):
        number = int(sysfs_path.name[5:])
        device_path = (sysfs_path / "device").resolve()
        device_name = device_path.name
        usb_interface = device_name.split(":")[-1] if ":" in device_name else None
        index_text = read_text(sysfs_path / "index")
        try:
            index = int(index_text)
        except ValueError:
            index = None
        video_path = f"/dev/{sysfs_path.name}"
        formats, sizes_by_format = parse_formats(run_v4l2(video_path))
        nodes.append(
            VideoNode(
                path=video_path,
                number=number,
                index=index,
                usb_interface=usb_interface,
                serial_number=usb_serial_from_device_path(device_path),
                name=read_text(sysfs_path / "name"),
                formats=formats,
                sizes_by_format=sizes_by_format,
                usb_speed=usb_speed_from_device_path(device_path),
            )
        )
    return nodes


def has_format(node: VideoNode, expected: tuple[str, ...]) -> bool:
    return any(fmt.strip() in expected or fmt.strip().startswith(expected) for fmt in node.formats for expected in expected)


def is_color_node(node: VideoNode) -> bool:
    return any(fmt in node.formats for fmt in COLOR_FORMATS) and not any(fmt in node.formats for fmt in BAD_COLOR_FORMATS)


def is_depth_node(node: VideoNode) -> bool:
    return any(fmt.startswith("Z16") for fmt in node.formats)


def parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", value.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError("size must look like WIDTHxHEIGHT, for example 640x480")
    return int(match.group(1)), int(match.group(2))


def size_to_shape(size: tuple[int, int]) -> list[int]:
    width, height = size
    return [height, width]


def node_supports_size(node: VideoNode, size: tuple[int, int], preferred_formats: tuple[str, ...]) -> bool:
    for fmt in preferred_formats:
        if size in node.sizes_by_format.get(fmt, ()):
            return True
    return False


def pick_node(
    nodes: list[VideoNode],
    serial_number: str,
    *,
    stream: str,
    preferred_interface: str,
    preferred_index: int,
    preferred_size: tuple[int, int],
) -> VideoNode | None:
    candidates = [node for node in nodes if node.serial_number == serial_number]
    if stream == "color":
        candidates = [node for node in candidates if is_color_node(node)]
        preferred_formats = COLOR_FORMATS
    elif stream == "depth":
        candidates = [node for node in candidates if is_depth_node(node)]
        preferred_formats = DEPTH_FORMATS
    else:
        raise ValueError(stream)

    if not candidates:
        return None

    def score(node: VideoNode) -> tuple[int, int, int, int]:
        return (
            0 if node.usb_interface == preferred_interface else 1,
            0 if node.index == preferred_index else 1,
            0 if node_supports_size(node, preferred_size, preferred_formats) else 1,
            node.number,
        )

    return sorted(candidates, key=score)[0]


def yaml_scalar(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    text = str(value)
    if text in {"YUYV", "h264", "opencv", "opencv_depth", "depth_z16", "gray16le", "uint16", "rgbd"}:
        return text
    return '"' + text.replace('"', '\\"') + '"'


def render_mapping(values: dict[str, object], indent: int = 2) -> list[str]:
    lines = []
    prefix = " " * indent
    for key, value in values.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(render_mapping(value, indent + 2))
        else:
            lines.append(f"{prefix}{key}: {yaml_scalar(value)}")
    return lines


def render_block(name: str, values: dict[str, object]) -> str:
    lines = [f"{name}:"]
    lines.extend(render_mapping(values))
    return "\n".join(lines)


def color_config(
    *,
    zmq_port: int,
    webrtc_port: int,
    serial_number: str,
    usb_interface: str,
    video_index: int,
    image_shape: list[int],
    fps: int,
) -> dict[str, object]:
    return {
        "enable_zmq": True,
        "zmq_port": zmq_port,
        "enable_webrtc": True,
        "webrtc_port": webrtc_port,
        "webrtc_codec": "h264",
        "type": "opencv",
        "fourcc": "YUYV",
        "direct_video_id": True,
        "usb_interface": usb_interface,
        "video_index": video_index,
        "image_shape": image_shape,
        "binocular": False,
        "fps": fps,
        "video_id": None,
        "serial_number": serial_number,
        "physical_path": None,
    }


def depth_config(
    *,
    serial_number: str,
    usb_interface: str,
    video_index: int,
    image_shape: list[int],
    fps: int,
) -> dict[str, object]:
    return {
        "type": "opencv_depth",
        "data_format": "depth_z16",
        "fourcc": "Z16 ",
        "ffmpeg_input_format": "gray16le",
        "depth_dtype": "uint16",
        "direct_video_id": True,
        "usb_interface": usb_interface,
        "video_index": video_index,
        "image_shape": image_shape,
        "binocular": False,
        "fps": fps,
        "video_id": None,
        "serial_number": serial_number,
        "physical_path": None,
    }


def rgbd_config(*, zmq_port: int, color_camera: str, depth: dict[str, object], fps: int) -> dict[str, object]:
    return {
        "enable_zmq": False,
        "zmq_port": zmq_port,
        "enable_webrtc": False,
        "webrtc_port": None,
        "type": "rgbd",
        "data_format": "rgbd",
        "color_camera": color_camera,
        "depth_dtype": "uint16",
        "fps": fps,
        "depth": depth,
    }


def rgbd_name_for_color(camera_name: str) -> str:
    label = CAMERA_LABELS[camera_name]
    return f"{label}_rgbd_camera"


def render_config(blocks: list[tuple[str, dict[str, object]]], inspected_date: str) -> str:
    header = [
        "# Capture configuration generated from V4L2 discovery.",
        f"# Device mapping was inspected on {inspected_date}.",
        "# Stable selector = serial_number + usb_interface + video_index.",
        "",
    ]
    body = "\n\n".join(render_block(name, values) for name, values in blocks)
    return "\n".join(header) + body + "\n"


def print_node_list(nodes: list[VideoNode]) -> None:
    if not nodes:
        print("No /sys/class/video4linux/video* nodes found.")
        return
    for node in nodes:
        kind = "depth" if is_depth_node(node) else ("rgb" if is_color_node(node) else "skip")
        sizes = []
        for fmt, fmt_sizes in node.sizes_by_format.items():
            advertised = ",".join(f"{width}x{height}" for width, height in fmt_sizes)
            if advertised:
                sizes.append(f"{fmt}:{advertised}")
        print(
            f"{node.path:12} kind={kind:5} serial={node.serial_number or '-':14} "
            f"interface={node.usb_interface or '-':4} index={node.index if node.index is not None else '-':2} "
            f"speed={node.usb_speed or '-':9} formats={','.join(node.formats) or '-'}"
        )
        if sizes:
            print(f"  sizes(all): {' | '.join(sizes)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print detected video nodes and exit.")
    parser.add_argument("-o", "--output", type=Path, help="Write generated YAML to this path. Defaults to stdout.")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="Date string for config header.")
    parser.add_argument("--allow-missing", action="store_true", help="Generate preferred selectors even if a role is missing.")

    parser.add_argument("--head-serial", help="Serial number for head_camera.")
    parser.add_argument("--torso-serial", help="Serial number for torso_camera.")
    parser.add_argument("--left-serial", help="Serial number for left_wrist_camera.")
    parser.add_argument("--right-serial", help="Serial number for right_wrist_camera.")

    parser.add_argument("--color-size", type=parse_size, default=(640, 480), help="RGB size WIDTHxHEIGHT. Default: 640x480.")
    parser.add_argument("--head-color-size", type=parse_size, help="Override RGB size for head_camera.")
    parser.add_argument("--depth-size", type=parse_size, default=(640, 480), help="Depth size WIDTHxHEIGHT. Default: 640x480.")
    parser.add_argument("--head-depth-size", type=parse_size, help="Override depth size for head_rgbd_camera.depth.")
    parser.add_argument("--color-fps", type=int, default=30)
    parser.add_argument("--depth-fps", type=int, default=30)

    parser.add_argument("--prefer-color-interface", default="1.4")
    parser.add_argument("--prefer-color-index", type=int, default=0)
    parser.add_argument("--prefer-depth-interface", default="1.0")
    parser.add_argument("--prefer-depth-index", type=int, default=0)
    parser.add_argument(
        "--depth",
        choices=("none", "head", "all"),
        default="head",
        help="Which RGB-D configs to generate: none, head, or all four cameras. Default: head.",
    )
    return parser.parse_args()


def required_serials(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "head_camera": args.head_serial,
        "torso_camera": args.torso_serial,
        "left_wrist_camera": args.left_serial,
        "right_wrist_camera": args.right_serial,
    }


def build_config(args: argparse.Namespace, nodes: list[VideoNode]) -> list[tuple[str, dict[str, object]]]:
    serials = required_serials(args)
    missing_roles = [name for name, serial in serials.items() if not serial]
    if missing_roles and not args.allow_missing:
        raise SystemExit(
            "Missing role serials: "
            + ", ".join(missing_roles)
            + ". Pass --head-serial/--torso-serial/--left-serial/--right-serial, or use --allow-missing."
        )

    blocks: list[tuple[str, dict[str, object]]] = []
    for camera_name in CAMERA_ORDER:
        serial = serials[camera_name] or "UNKNOWN"
        color_size = args.head_color_size if camera_name == "head_camera" and args.head_color_size else args.color_size
        color_node = pick_node(
            nodes,
            serial,
            stream="color",
            preferred_interface=args.prefer_color_interface,
            preferred_index=args.prefer_color_index,
            preferred_size=color_size,
        )
        if color_node is None:
            if not args.allow_missing:
                raise SystemExit(f"Cannot find RGB node for {camera_name} serial={serial}. Run --list to inspect devices.")
            usb_interface = args.prefer_color_interface
            video_index = args.prefer_color_index
        else:
            usb_interface = color_node.usb_interface or args.prefer_color_interface
            video_index = color_node.index if color_node.index is not None else args.prefer_color_index
            print(
                f"[pick] {camera_name}: {serial} -> {color_node.path} "
                f"interface={usb_interface} index={video_index} speed={color_node.usb_speed or '-'} "
                f"formats={','.join(color_node.formats) or '-'}",
                file=sys.stderr,
            )
            if not node_supports_size(color_node, color_size, COLOR_FORMATS):
                print(
                    f"[warn] {camera_name}: selected node does not advertise {color_size[0]}x{color_size[1]} "
                    "for YUYV/MJPG; check v4l2-ctl before starting service.",
                    file=sys.stderr,
                )

        zmq_port, webrtc_port = COLOR_PORTS[camera_name]
        blocks.append(
            (
                camera_name,
                color_config(
                    zmq_port=zmq_port,
                    webrtc_port=webrtc_port,
                    serial_number=serial,
                    usb_interface=usb_interface,
                    video_index=video_index,
                    image_shape=size_to_shape(color_size),
                    fps=args.color_fps,
                ),
            )
        )

        if args.depth == "all" or (args.depth == "head" and camera_name == "head_camera"):
            rgbd_name = rgbd_name_for_color(camera_name)
            depth_size = args.head_depth_size if rgbd_name == "head_rgbd_camera" and args.head_depth_size else args.depth_size
            depth_node = pick_node(
                nodes,
                serial,
                stream="depth",
                preferred_interface=args.prefer_depth_interface,
                preferred_index=args.prefer_depth_index,
                preferred_size=depth_size,
            )
            if depth_node is None:
                if not args.allow_missing:
                    raise SystemExit(f"Cannot find Z16 depth node for {rgbd_name}.depth serial={serial}. Run --list to inspect devices.")
                depth_interface = args.prefer_depth_interface
                depth_index = args.prefer_depth_index
            else:
                depth_interface = depth_node.usb_interface or args.prefer_depth_interface
                depth_index = depth_node.index if depth_node.index is not None else args.prefer_depth_index
                print(
                    f"[pick] {rgbd_name}.depth: {serial} -> {depth_node.path} "
                    f"interface={depth_interface} index={depth_index} speed={depth_node.usb_speed or '-'} "
                    f"formats={','.join(depth_node.formats) or '-'}",
                    file=sys.stderr,
                )
                if not node_supports_size(depth_node, depth_size, DEPTH_FORMATS):
                    print(
                        f"[warn] {rgbd_name}.depth: selected node does not advertise {depth_size[0]}x{depth_size[1]} "
                        "for Z16; check v4l2-ctl before starting service.",
                        file=sys.stderr,
                    )

            blocks.append(
                (
                    rgbd_name,
                    rgbd_config(
                        zmq_port=RGBD_PORTS[rgbd_name],
                        color_camera=camera_name,
                        depth=depth_config(
                            serial_number=serial,
                            usb_interface=depth_interface,
                            video_index=depth_index,
                            image_shape=size_to_shape(depth_size),
                            fps=args.depth_fps,
                        ),
                        fps=min(args.color_fps, args.depth_fps),
                    ),
                )
            )

    return blocks


def main() -> int:
    args = parse_args()
    nodes = discover_video_nodes()
    if args.list:
        print_node_list(nodes)
        return 0

    blocks = build_config(args, nodes)
    yaml_text = render_config(blocks, args.date)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml_text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(yaml_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
