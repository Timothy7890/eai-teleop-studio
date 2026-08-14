from __future__ import annotations

import json
import os
import queue
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np


class HandEyeRecorder:
    """Asynchronously persist one short RGB-D burst as one completed episode."""

    def __init__(self, task_dir: str | os.PathLike[str]):
        self.task_dir = Path(task_dir).expanduser()
        self.task_dir.mkdir(parents=True, exist_ok=True)
        episode_ids = []
        for child in self.task_dir.iterdir():
            match = re.fullmatch(r"episode_(\d+)", child.name) if child.is_dir() else None
            if match:
                episode_ids.append(int(match.group(1)))
        self._next_episode_id = max(episode_ids, default=0) + 1
        self._frames: list[dict[str, Any]] = []
        self._frame_ids: set[Any] = set()
        self._saving = False
        self._result: Optional[dict[str, Any]] = None
        self._lock = threading.Lock()
        self._queue: queue.Queue[Optional[tuple[int, list[dict[str, Any]], dict[str, Any]]]] = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def saving(self) -> bool:
        with self._lock:
            return self._saving

    def begin_sample(self) -> None:
        with self._lock:
            if self._saving:
                raise RuntimeError("previous hand-eye sample is still being saved")
            self._frames = []
            self._frame_ids = set()
            self._result = None

    def add_frame(
        self,
        *,
        rgb_jpg: bytes,
        depth: np.ndarray,
        rgbd_metadata: dict[str, Any],
        right_arm_q: np.ndarray,
        joint_timestamp_ns: int,
        sample_timestamp_ns: int,
    ) -> int:
        metadata = dict(rgbd_metadata or {})
        frame_id = metadata.get("frame_id")
        q = np.asarray(right_arm_q, dtype=float)
        depth_array = np.asarray(depth)
        if not rgb_jpg:
            raise ValueError("RGB JPEG payload is empty")
        if depth_array.dtype != np.uint16 or depth_array.ndim != 2:
            raise ValueError(f"depth must be a 2D uint16 array, got shape={depth_array.shape} dtype={depth_array.dtype}")
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("right_arm_q must contain 7 finite values")
        with self._lock:
            if self._saving:
                raise RuntimeError("cannot add frames while a sample is being saved")
            if frame_id is not None and frame_id in self._frame_ids:
                return len(self._frames)
            if frame_id is not None:
                self._frame_ids.add(frame_id)
            self._frames.append(
                {
                    "rgb_jpg": bytes(rgb_jpg),
                    "depth": depth_array.copy(),
                    "rgbd_metadata": metadata,
                    "right_arm_q": q.copy(),
                    "joint_timestamp_ns": int(joint_timestamp_ns),
                    "sample_timestamp_ns": int(sample_timestamp_ns),
                }
            )
            return len(self._frames)

    def submit_sample(self, sample_info: Optional[dict[str, Any]] = None) -> int:
        with self._lock:
            if self._saving:
                raise RuntimeError("a hand-eye sample is already being saved")
            if not self._frames:
                raise RuntimeError("cannot save an empty hand-eye sample")
            episode_id = self._next_episode_id
            self._next_episode_id += 1
            frames = self._frames
            self._frames = []
            self._frame_ids = set()
            self._saving = True
            self._result = None
        self._queue.put((episode_id, frames, dict(sample_info or {})))
        return episode_id

    def poll_result(self) -> Optional[dict[str, Any]]:
        with self._lock:
            if self._result is None:
                return None
            result = dict(self._result)
            self._result = None
            return result

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            episode_id, frames, sample_info = item
            try:
                result = self._write_episode(episode_id, frames, sample_info)
            except Exception as exc:
                result = {"ok": False, "error": str(exc), "episode_id": episode_id}
            with self._lock:
                self._saving = False
                self._result = result
            self._queue.task_done()

    def _write_episode(
        self,
        episode_id: int,
        frames: list[dict[str, Any]],
        sample_info: dict[str, Any],
    ) -> dict[str, Any]:
        episode_name = f"episode_{episode_id:04d}"
        final_dir = self.task_dir / episode_name
        temp_dir = self.task_dir / f".{episode_name}.tmp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if final_dir.exists():
            raise FileExistsError(f"episode already exists: {final_dir}")
        rgb_dir = temp_dir / "rgb"
        depth_dir = temp_dir / "depth"
        rgb_dir.mkdir(parents=True)
        depth_dir.mkdir(parents=True)

        data = []
        for idx, frame in enumerate(frames):
            rgb_name = f"{idx:06d}_head_rgb.jpg"
            depth_name = f"{idx:06d}_head_depth.npy"
            (rgb_dir / rgb_name).write_bytes(frame["rgb_jpg"])
            np.save(depth_dir / depth_name, frame["depth"], allow_pickle=False)
            metadata = dict(frame["rgbd_metadata"])
            data.append(
                {
                    "idx": idx,
                    "colors": {"head_rgb": f"rgb/{rgb_name}"},
                    "depths": {"head_depth": f"depth/{depth_name}"},
                    "states": {
                        "right_arm": {
                            "qpos": frame["right_arm_q"].tolist(),
                        }
                    },
                    "timestamps": {
                        "rgbd_timestamp_ns": metadata.get("timestamp_ns"),
                        "joint_timestamp_ns": frame["joint_timestamp_ns"],
                        "sample_timestamp_ns": frame["sample_timestamp_ns"],
                    },
                    "rgbd": {
                        "frame_id": metadata.get("frame_id"),
                        "stream": metadata.get("stream"),
                        "color_camera": metadata.get("color_camera"),
                        "depth_camera": metadata.get("depth_camera"),
                        "color_shape": metadata.get("color_shape"),
                        "depth_shape": metadata.get("depth_shape"),
                        "color_format": metadata.get("color_format", "jpeg"),
                        "depth_format": metadata.get("depth_format", "depth_z16"),
                        "depth_dtype": metadata.get("depth_dtype", "uint16"),
                    },
                }
            )

        payload = {
            "info": {
                "version": "1.0.0",
                "kind": "hand_eye_calibration",
                "frame_count": len(data),
                **sample_info,
            },
            "data": data,
        }
        (temp_dir / "data.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp_dir, final_dir)
        return {
            "ok": True,
            "episode_id": episode_id,
            "path": str(final_dir),
            "frame_count": len(data),
        }

    def close(self) -> None:
        self._queue.join()
        self._queue.put(None)
        self._worker.join(timeout=5.0)
