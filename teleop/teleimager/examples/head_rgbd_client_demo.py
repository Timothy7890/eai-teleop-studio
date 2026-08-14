#!/usr/bin/env python3
"""Read the combined head RGB-D stream from a teleimager server.

Example:
    python teleop/teleimager/examples/head_rgbd_client_demo.py --host 192.168.61.142
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _ensure_repo_teleimager_on_path():
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.exists():
        sys.path.insert(0, str(repo_src))


_ensure_repo_teleimager_on_path()

from teleimager.image_client import ImageClient  # noqa: E402


def _save_depth_preview_png(depth: np.ndarray, output_path: Path) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Saving PNG requires opencv-python. Install cv2 or omit --save-depth-png.") from exc

    valid = depth[depth > 0]
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
    parser = argparse.ArgumentParser(description="Read combined head RGB-D frames from a teleimager server.")
    parser.add_argument("--host", required=True, help="Teleimager server IP address, for example 192.168.61.142")
    parser.add_argument("--request-port", type=int, default=60000, help="Teleimager config request port.")
    parser.add_argument("--camera-name", default="head_rgbd_camera", help="RGB-D stream name in cam_config_server.yaml.")
    parser.add_argument("--frames", type=int, default=30, help="How many RGB-D frames to wait for before exiting.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for the first valid RGB-D frame.")
    parser.add_argument("--save-rgb-jpg", type=Path, help="Optional path to save the latest RGB JPEG frame.")
    parser.add_argument("--save-depth-npy", type=Path, help="Optional path to save the latest raw uint16 depth frame.")
    parser.add_argument("--save-depth-png", type=Path, help="Optional path to save a colorized depth preview PNG.")
    args = parser.parse_args()

    client = ImageClient(host=args.host, request_port=args.request_port, request_bgr=True)
    cam_config = client.get_cam_config()
    rgbd_cfg = cam_config.get(args.camera_name)
    if not isinstance(rgbd_cfg, dict):
        raise RuntimeError(f"RGB-D stream '{args.camera_name}' is not present in server config.")
    if not rgbd_cfg.get("enable_zmq"):
        raise RuntimeError(f"RGB-D stream '{args.camera_name}' exists but enable_zmq is false.")
    if rgbd_cfg.get("data_format") != "rgbd":
        raise RuntimeError(f"Camera '{args.camera_name}' is not an rgbd stream: {rgbd_cfg}")

    latest_frame = None
    start = time.monotonic()
    for idx in range(max(1, args.frames)):
        frame = client.get_rgbd_frame(args.camera_name)
        if not frame or frame.bgr is None or frame.depth is None:
            if time.monotonic() - start > args.timeout:
                raise TimeoutError(f"No valid RGB-D frame received from '{args.camera_name}' within {args.timeout:.1f}s.")
            time.sleep(0.02)
            continue

        latest_frame = frame
        valid = frame.depth[frame.depth > 0]
        if valid.size:
            depth_stats = f"depth_min={int(valid.min())} depth_max={int(valid.max())} depth_mean={float(valid.mean()):.1f}"
        else:
            depth_stats = "no non-zero depth pixels"
        metadata = frame.metadata or {}
        print(
            f"[{idx + 1}/{args.frames}] fps={frame.fps:.1f} "
            f"frame_id={metadata.get('frame_id')} timestamp_ns={metadata.get('timestamp_ns')} "
            f"rgb_shape={frame.bgr.shape} depth_shape={frame.depth.shape} {depth_stats}"
        )
        time.sleep(0.02)

    if latest_frame is None:
        raise RuntimeError(f"No valid RGB-D frame received from '{args.camera_name}'.")

    if args.save_rgb_jpg:
        args.save_rgb_jpg.write_bytes(latest_frame.rgb_jpg)
        print(f"Saved RGB JPEG frame: {args.save_rgb_jpg}")

    if args.save_depth_npy:
        np.save(args.save_depth_npy, latest_frame.depth)
        print(f"Saved raw depth frame: {args.save_depth_npy}")

    if args.save_depth_png:
        _save_depth_preview_png(latest_frame.depth, args.save_depth_png)
        print(f"Saved colorized depth preview: {args.save_depth_png}")

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
