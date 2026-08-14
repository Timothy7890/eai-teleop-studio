#!/usr/bin/env python3
"""Read the head depth stream from a teleimager server.

Example:
    python teleop/teleimager/examples/head_depth_client_demo.py --host 192.168.123.5

Optional:
    python teleop/teleimager/examples/head_depth_client_demo.py \
        --host 192.168.123.5 \
        --frames 100 \
        --save-npy head_depth.npy \
        --save-png head_depth_preview.png
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _ensure_repo_teleimager_on_path():
    """Allow running this demo directly from a repository checkout."""
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.exists():
        sys.path.insert(0, str(repo_src))


_ensure_repo_teleimager_on_path()

from teleimager.image_client import ImageClient  # noqa: E402


def _valid_depth_pixels(depth: np.ndarray) -> np.ndarray:
    return depth[depth > 0]


def _save_depth_preview_png(depth: np.ndarray, output_path: Path) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Saving PNG requires opencv-python. Install cv2 or omit --save-png.") from exc

    valid = _valid_depth_pixels(depth)
    if valid.size == 0:
        preview = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(valid, [1, 99])
        if hi <= lo:
            hi = lo + 1
        preview = np.clip((depth.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)

    colored = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    if not cv2.imwrite(str(output_path), colored):
        raise RuntimeError(f"Failed to save PNG preview to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read head depth frames from a teleimager server.")
    parser.add_argument("--host", required=True, help="Teleimager server IP address, for example 192.168.123.5")
    parser.add_argument("--request-port", type=int, default=60000, help="Teleimager config request port.")
    parser.add_argument("--camera-name", default="head_depth_camera", help="Depth stream name in cam_config_server.yaml.")
    parser.add_argument("--frames", type=int, default=30, help="How many depth frames to wait for before exiting.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for the first valid depth frame.")
    parser.add_argument("--save-npy", type=Path, help="Optional path to save the latest raw uint16 depth frame as .npy.")
    parser.add_argument("--save-png", type=Path, help="Optional path to save a colorized depth preview PNG.")
    args = parser.parse_args()

    client = ImageClient(host=args.host, request_port=args.request_port, request_bgr=False)
    cam_config = client.get_cam_config()
    depth_cfg = cam_config.get(args.camera_name)
    if not isinstance(depth_cfg, dict):
        raise RuntimeError(f"Depth camera '{args.camera_name}' is not present in server config.")
    if not depth_cfg.get("enable_zmq"):
        raise RuntimeError(f"Depth camera '{args.camera_name}' exists but enable_zmq is false.")
    if depth_cfg.get("data_format") != "depth_z16":
        raise RuntimeError(f"Camera '{args.camera_name}' is not a depth_z16 stream: {depth_cfg}")

    latest_depth = None
    start = time.monotonic()
    for idx in range(max(1, args.frames)):
        depth_frame = client.get_depth_frame(args.camera_name)
        depth = depth_frame.depth
        if depth is None:
            if time.monotonic() - start > args.timeout:
                raise TimeoutError(f"No valid depth frame received from '{args.camera_name}' within {args.timeout:.1f}s.")
            time.sleep(0.02)
            continue

        latest_depth = depth.copy()
        valid = _valid_depth_pixels(latest_depth)
        if valid.size:
            valid_stats = f"valid_min={int(valid.min())} valid_max={int(valid.max())} valid_mean={float(valid.mean()):.1f}"
        else:
            valid_stats = "no non-zero depth pixels"
        print(
            f"[{idx + 1}/{args.frames}] fps={depth_frame.fps:.1f} "
            f"shape={latest_depth.shape} dtype={latest_depth.dtype} {valid_stats}"
        )
        time.sleep(0.02)

    if latest_depth is None:
        raise RuntimeError(f"No valid depth frame received from '{args.camera_name}'.")

    if args.save_npy:
        np.save(args.save_npy, latest_depth)
        print(f"Saved raw depth frame: {args.save_npy}")

    if args.save_png:
        _save_depth_preview_png(latest_depth, args.save_png)
        print(f"Saved colorized depth preview: {args.save_png}")

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
