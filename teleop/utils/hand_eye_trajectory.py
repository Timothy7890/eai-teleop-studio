from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


class HandEyeTrajectoryRecorder:
    """Record the exact right-arm command stream and calibration capture events."""

    def __init__(
        self,
        task_dir: str | os.PathLike[str],
        *,
        left_fixed_q: np.ndarray,
        frequency: float,
    ):
        task_path = Path(task_dir).expanduser()
        trajectories_dir = task_path / "trajectories"
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        base_name = time.strftime("trajectory_%Y%m%d_%H%M%S")
        self.directory = trajectories_dir / base_name
        suffix = 1
        while self.directory.exists():
            self.directory = trajectories_dir / f"{base_name}_{suffix:02d}"
            suffix += 1
        self.directory.mkdir(parents=True)

        self.left_fixed_q = self._q(left_fixed_q, "left_fixed_q")
        self.frequency = float(frequency)
        if not np.isfinite(self.frequency) or self.frequency <= 0:
            raise ValueError("frequency must be positive")
        self._start_monotonic_ns = time.monotonic_ns()
        self._elapsed_ns: list[int] = []
        self._right_command_q: list[np.ndarray] = []
        self._right_measured_q: list[np.ndarray] = []
        self._events: list[dict[str, Any]] = []
        self._open_event: Optional[dict[str, Any]] = None
        self._closed = False
        self._jsonl = (self.directory / "trajectory.jsonl").open("a", encoding="utf-8")
        self._write_metadata(completed=False)

    @staticmethod
    def _q(value: np.ndarray, name: str) -> np.ndarray:
        q = np.asarray(value, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError(f"{name} must contain 7 finite joint values")
        return q.copy()

    @property
    def frame_count(self) -> int:
        return len(self._elapsed_ns)

    @property
    def last_frame_index(self) -> Optional[int]:
        return self.frame_count - 1 if self.frame_count else None

    def add_frame(
        self,
        *,
        right_command_q: np.ndarray,
        right_measured_q: np.ndarray,
        monotonic_ns: Optional[int] = None,
    ) -> int:
        if self._closed:
            raise RuntimeError("trajectory recorder is closed")
        command_q = self._q(right_command_q, "right_command_q")
        measured_q = self._q(right_measured_q, "right_measured_q")
        now_ns = time.monotonic_ns() if monotonic_ns is None else int(monotonic_ns)
        elapsed_ns = max(0, now_ns - self._start_monotonic_ns)
        index = self.frame_count
        self._elapsed_ns.append(elapsed_ns)
        self._right_command_q.append(command_q)
        self._right_measured_q.append(measured_q)
        self._jsonl.write(
            json.dumps(
                {
                    "frame_index": index,
                    "elapsed_ns": elapsed_ns,
                    "right_command_q": command_q.tolist(),
                    "right_measured_q": measured_q.tolist(),
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        if index % max(1, int(round(self.frequency))) == 0:
            self._jsonl.flush()
        return index

    def begin_capture_event(self, frame_index: Optional[int] = None) -> int:
        if self._open_event is not None:
            raise RuntimeError("a trajectory capture event is already open")
        index = self.last_frame_index if frame_index is None else int(frame_index)
        if index is None or index < 0:
            index = 0
        event = {
            "event_id": len(self._events),
            "hold_start_frame_index": index,
            "capture_frame_index": None,
            "resume_frame_index": None,
            "episode": None,
            "error": None,
        }
        self._events.append(event)
        self._open_event = event
        self._flush_events()
        return event["event_id"]

    def mark_capture_saved(
        self,
        episode_path: str,
        frame_index: Optional[int] = None,
    ) -> None:
        if self._open_event is None:
            raise RuntimeError("no open trajectory capture event")
        index = self.last_frame_index if frame_index is None else int(frame_index)
        self._open_event["capture_frame_index"] = index
        self._open_event["episode"] = Path(episode_path).name
        self._flush_events()

    def mark_capture_error(self, message: str) -> None:
        if self._open_event is not None:
            self._open_event["error"] = str(message)
            self._flush_events()

    def finish_capture_event(self, frame_index: Optional[int] = None) -> None:
        if self._open_event is None:
            raise RuntimeError("no open trajectory capture event")
        index = self.last_frame_index if frame_index is None else int(frame_index)
        self._open_event["resume_frame_index"] = index
        self._open_event = None
        self._flush_events()

    def _events_payload(self, *, completed: bool) -> dict[str, Any]:
        return {
            "version": "1.0.0",
            "kind": "hand_eye_right_arm_trajectory",
            "completed": bool(completed),
            "frequency": self.frequency,
            "frame_count": self.frame_count,
            "left_fixed_q": self.left_fixed_q.tolist(),
            "events": self._events,
        }

    def _write_metadata(self, *, completed: bool) -> None:
        target = self.directory / "events.json"
        temp = self.directory / ".events.json.tmp"
        temp.write_text(
            json.dumps(self._events_payload(completed=completed), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp, target)

    def _flush_events(self) -> None:
        self._jsonl.flush()
        self._write_metadata(completed=False)

    def close(self) -> Path:
        if self._closed:
            return self.directory
        if self._open_event is not None:
            self._open_event["error"] = self._open_event.get("error") or "trajectory stopped before event resumed"
            self._open_event["resume_frame_index"] = self.last_frame_index
            self._open_event = None
        for event in self._events:
            capture_index = event.get("capture_frame_index")
            resume_index = event.get("resume_frame_index")
            if (
                capture_index is not None
                and resume_index is not None
                and (
                    int(capture_index) < 0
                    or int(resume_index) < int(capture_index)
                    or int(resume_index) >= self.frame_count
                )
            ):
                event["error"] = event.get("error") or "capture event indices are incomplete"
        self._jsonl.flush()
        self._jsonl.close()
        elapsed = np.asarray(self._elapsed_ns, dtype=np.int64)
        commands = np.asarray(self._right_command_q, dtype=np.float64).reshape((-1, 7))
        measured = np.asarray(self._right_measured_q, dtype=np.float64).reshape((-1, 7))
        np.savez(
            self.directory / "trajectory.npz",
            elapsed_ns=elapsed,
            right_command_q=commands,
            right_measured_q=measured,
            left_fixed_q=self.left_fixed_q,
        )
        self._closed = True
        self._write_metadata(completed=True)
        return self.directory


class HandEyeTrajectoryReplay:
    """Sequentially replay every recorded right-arm command without replanning."""

    WAITING = "WAITING"
    PLAYING = "PLAYING"
    EVENT_HOLD = "EVENT_HOLD"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"

    def __init__(
        self,
        trajectory_path: str | os.PathLike[str],
        *,
        start_tolerance: float = 0.05,
        tracking_error_limit: float = 0.08,
        tracking_error_frames: int = 15,
        time_scale: float = 1.0,
    ):
        path = Path(trajectory_path).expanduser()
        self.directory = path if path.is_dir() else path.parent
        npz_path = path if path.suffix == ".npz" else self.directory / "trajectory.npz"
        events_path = self.directory / "events.json"
        if not npz_path.is_file() or not events_path.is_file():
            raise FileNotFoundError(f"incomplete trajectory: {self.directory}")
        with np.load(npz_path, allow_pickle=False) as data:
            self.elapsed_ns = np.asarray(data["elapsed_ns"], dtype=np.int64)
            self.right_command_q = np.asarray(data["right_command_q"], dtype=float)
            self.right_measured_q = np.asarray(data["right_measured_q"], dtype=float)
            self.left_fixed_q = np.asarray(data["left_fixed_q"], dtype=float)
        metadata = json.loads(events_path.read_text(encoding="utf-8"))
        if not metadata.get("completed"):
            raise ValueError("trajectory was not finalized")
        if self.right_command_q.ndim != 2 or self.right_command_q.shape[1:] != (7,):
            raise ValueError("right_command_q must have shape [N, 7]")
        if self.right_measured_q.shape != self.right_command_q.shape:
            raise ValueError("right_measured_q shape does not match commands")
        if self.elapsed_ns.shape != (self.right_command_q.shape[0],):
            raise ValueError("elapsed_ns shape does not match commands")
        if self.right_command_q.shape[0] == 0:
            raise ValueError("trajectory contains no frames")
        if np.any(np.diff(self.elapsed_ns) < 0):
            raise ValueError("trajectory elapsed_ns must be monotonic")
        if self.left_fixed_q.shape != (7,):
            raise ValueError("left_fixed_q must contain 7 values")
        if not (
            np.all(np.isfinite(self.right_command_q))
            and np.all(np.isfinite(self.right_measured_q))
            and np.all(np.isfinite(self.left_fixed_q))
        ):
            raise ValueError("trajectory contains non-finite joint values")
        self.events = sorted(
            [
                event
                for event in metadata.get("events", [])
                if event.get("capture_frame_index") is not None
                and event.get("resume_frame_index") is not None
                and not event.get("error")
            ],
            key=lambda event: int(event["capture_frame_index"]),
        )
        for event in self.events:
            capture_index = int(event["capture_frame_index"])
            resume_index = int(event["resume_frame_index"])
            if not 0 <= capture_index <= resume_index < self.right_command_q.shape[0]:
                raise ValueError(f"invalid capture event indices: {event}")
        self.start_tolerance = float(start_tolerance)
        self.tracking_error_limit = float(tracking_error_limit)
        self.tracking_error_frames = int(tracking_error_frames)
        if self.start_tolerance < 0:
            raise ValueError("start_tolerance cannot be negative")
        if self.tracking_error_limit <= 0 or self.tracking_error_frames <= 0:
            raise ValueError("tracking error limit and frame count must be positive")
        self.time_scale = float(time_scale)
        if not 0 < self.time_scale <= 1:
            raise ValueError("time_scale must be in (0, 1] to prevent faster-than-recorded replay")
        self.state = self.WAITING
        self.index = 0
        self._event_cursor = 0
        self._active_event: Optional[dict[str, Any]] = None
        self._tracking_error_count = 0
        self.error: Optional[str] = None
        self.last_interval_seconds = 0.0

    @property
    def active_event(self) -> Optional[dict[str, Any]]:
        return None if self._active_event is None else dict(self._active_event)

    @property
    def target_right_q(self) -> np.ndarray:
        return self.right_command_q[self.index].copy()

    @property
    def progress(self) -> dict[str, Any]:
        return {
            "TRAJECTORY_REPLAY_STATE": self.state,
            "TRAJECTORY_REPLAY_INDEX": self.index,
            "TRAJECTORY_REPLAY_FRAMES": int(self.right_command_q.shape[0]),
            "TRAJECTORY_REPLAY_EVENT": self.active_event,
            "TRAJECTORY_REPLAY_ERROR": self.error,
            "TRAJECTORY_REPLAY_PATH": str(self.directory),
            "TRAJECTORY_REPLAY_TIME_SCALE": self.time_scale,
        }

    def start(self, current_dual_arm_q: np.ndarray) -> None:
        q = np.asarray(current_dual_arm_q, dtype=float)
        if q.shape != (14,) or not np.all(np.isfinite(q)):
            raise ValueError("current_dual_arm_q must contain 14 finite values")
        left_error = float(np.max(np.abs(q[:7] - self.left_fixed_q)))
        right_error = float(np.max(np.abs(q[-7:] - self.right_measured_q[0])))
        error = max(left_error, right_error)
        if error > self.start_tolerance:
            self.state = self.ERROR
            self.error = (
                f"trajectory start mismatch {error:.5f} rad exceeds "
                f"{self.start_tolerance:.5f} rad"
            )
            raise RuntimeError(self.error)
        self.state = self.PLAYING
        self.index = 0
        self.error = None

    def check_tracking(self, measured_right_q: np.ndarray) -> None:
        if self.state != self.PLAYING:
            return
        measured = np.asarray(measured_right_q, dtype=float)
        if measured.shape != (7,) or not np.all(np.isfinite(measured)):
            self.abort("measured_right_q must contain 7 finite values")
            raise RuntimeError(self.error)
        error = float(np.max(np.abs(measured - self.right_measured_q[self.index])))
        if error > self.tracking_error_limit:
            self._tracking_error_count += 1
        else:
            self._tracking_error_count = 0
        if self._tracking_error_count >= self.tracking_error_frames:
            self.state = self.ERROR
            self.error = (
                f"trajectory tracking error persisted for {self._tracking_error_count} frames; "
                f"latest={error:.5f} rad limit={self.tracking_error_limit:.5f} rad"
            )
            raise RuntimeError(self.error)

    def advance(self) -> Optional[dict[str, Any]]:
        if self.state != self.PLAYING:
            return None
        if self._event_cursor < len(self.events):
            event = self.events[self._event_cursor]
            if self.index >= int(event["capture_frame_index"]):
                self._active_event = event
                self.state = self.EVENT_HOLD
                return dict(event)
        if self.index + 1 >= self.right_command_q.shape[0]:
            self.state = self.COMPLETED
            return None
        previous_index = self.index
        self.index += 1
        recorded_dt = float(self.elapsed_ns[self.index] - self.elapsed_ns[previous_index]) / 1e9
        self.last_interval_seconds = max(0.001, recorded_dt / self.time_scale)
        return None

    def resume_after_event(self) -> None:
        if self.state != self.EVENT_HOLD or self._active_event is None:
            raise RuntimeError(f"cannot resume replay from {self.state}")
        resume_index = int(self._active_event["resume_frame_index"])
        self.index = min(max(resume_index, self.index), self.right_command_q.shape[0] - 1)
        self._event_cursor += 1
        self._active_event = None
        self.state = self.PLAYING
        self._tracking_error_count = 0

    def abort(self, message: str) -> None:
        self.state = self.ERROR
        self.error = str(message)
