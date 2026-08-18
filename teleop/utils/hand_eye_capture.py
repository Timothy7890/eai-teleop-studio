from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


FOLLOW = "FOLLOW"
SETTLING = "SETTLING"
CAPTURING = "CAPTURING"
SAVING = "SAVING"
HOLD = "HOLD"
RESUMING = "RESUMING"


def build_hand_eye_hud_status(
    snapshot: dict,
    *,
    started: bool,
    motion_ready: bool = True,
) -> tuple[str, str, str]:
    """Build a concise Chinese VR HUD message from capture state."""
    if not started:
        return "等待开始遥操", "确认追踪后按 A，或在电脑点击“开始遥操”", "info"

    state = snapshot.get("HAND_EYE_STATE", FOLLOW)
    follow_enabled = bool(snapshot.get("HAND_EYE_FOLLOW_ENABLED", True))
    saved = int(snapshot.get("HAND_EYE_SAVED_SAMPLES", 0) or 0)
    captured = int(snapshot.get("HAND_EYE_CAPTURED_FRAMES", 0) or 0)
    burst = int(snapshot.get("HAND_EYE_BURST_FRAMES", 1) or 1)
    error = snapshot.get("HAND_EYE_ERROR")

    if error:
        return "采集错误 · 机器人保持中", str(error)[:80], "error"
    if not follow_enabled:
        if not motion_ready:
            return "等待右手柄追踪", "确认控制器权限和手柄连接", "warning"
        return "固定姿态已就绪", "B：开始绝对位姿跟随　A：结束", "info"
    if state == SETTLING:
        progress = float(snapshot.get("HAND_EYE_SETTLING_PROGRESS", 0.0) or 0.0)
        speed = float(snapshot.get("HAND_EYE_SETTLING_WINDOW_SPEED", 0.0) or 0.0)
        span = float(snapshot.get("HAND_EYE_SETTLING_WINDOW_SPAN", 0.0) or 0.0)
        speed_limit = float(snapshot.get("HAND_EYE_MAX_JOINT_SPEED", 0.0) or 0.0)
        span_limit = float(snapshot.get("HAND_EYE_MAX_JOINT_SPAN", 0.0) or 0.0)
        return (
            f"关节判稳中… {progress * 100:.0f}%",
            f"编码器窗口速度 {speed:.4f}/{speed_limit:.4f} · "
            f"波动 {span:.4f}/{span_limit:.4f} rad",
            "warning",
        )
    if state == CAPTURING:
        return f"RGB-D 采集中 {captured} / {burst}", "请等待采集完成", "warning"
    if state == SAVING:
        return "数据保存中…", "请勿退出，等待写盘完成", "warning"
    if state == RESUMING:
        return "正在追赶手柄位置…", "机械臂平滑移动中，请保持手柄稳定", "warning"
    if state == HOLD:
        return f"第 {saved} 条已保存", "B：恢复绝对位姿跟随　A：结束", "success"
    if not motion_ready:
        return "等待右手柄追踪", "确认 VR 控制器权限和手柄连接", "warning"
    return f"遥操中 · 已保存 {saved} 条", "B：锁定并采样　A：结束遥操", "success"


def _pose(value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("pose must be a finite 4x4 matrix")
    return pose.copy()


@dataclass(frozen=True)
class RebaseAnchors:
    left_robot: np.ndarray
    right_robot: np.ndarray
    left_xr: np.ndarray
    right_xr: np.ndarray

    def apply(self, left_xr: np.ndarray, right_xr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left_delta = np.linalg.inv(self.left_xr) @ _pose(left_xr)
        right_delta = np.linalg.inv(self.right_xr) @ _pose(right_xr)
        return self.left_robot @ left_delta, self.right_robot @ right_delta


class HandEyeCaptureState:
    """Thread-safe capture/hold state shared by XR input and IPC callbacks."""

    def __init__(
        self,
        *,
        settle_seconds: float = 0.5,
        max_joint_speed: float = 0.05,
        max_joint_span: float = 0.003,
        max_hold_error: float = 0.30,
        burst_frames: int = 5,
    ):
        if settle_seconds <= 0:
            raise ValueError("settle_seconds must be positive")
        if max_joint_speed <= 0 or max_joint_span <= 0 or max_hold_error <= 0:
            raise ValueError("joint thresholds must be positive")
        if burst_frames <= 0:
            raise ValueError("burst_frames must be positive")
        self.settle_seconds = float(settle_seconds)
        self.max_joint_speed = float(max_joint_speed)
        self.max_joint_span = float(max_joint_span)
        self.max_hold_error = float(max_hold_error)
        self.burst_frames = int(burst_frames)
        self._lock = threading.RLock()
        self._state = FOLLOW
        self._toggle_requested = False
        self._button_was_pressed = False
        self._hold_q: Optional[np.ndarray] = None
        self._stable_since: Optional[float] = None
        self._q_history: deque[tuple[float, np.ndarray]] = deque()
        self._anchors: Optional[RebaseAnchors] = None
        self._captured_frames = 0
        self._saved_samples = 0
        self._last_sample: Optional[str] = None
        self._error: Optional[str] = None
        self._fatal_error = False
        self._follow_enabled = False
        self._settling_window_speed = 0.0
        self._settling_window_span = 0.0
        self._settling_raw_dq = 0.0
        self._settling_progress = 0.0
        self._settling_blocker = "collecting_window"

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def hold_q(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._hold_q is None else self._hold_q.copy()

    @property
    def is_holding(self) -> bool:
        return self.state != FOLLOW

    @property
    def fatal_error(self) -> bool:
        with self._lock:
            return self._fatal_error

    @property
    def follow_enabled(self) -> bool:
        with self._lock:
            return self._follow_enabled

    def enable_follow(self) -> None:
        with self._lock:
            if self._state != FOLLOW:
                raise RuntimeError(f"cannot enable follow from {self._state}")
            self._follow_enabled = True

    def enable_follow_with_resume(self) -> None:
        """Enable follow but enter RESUMING first.

        Used on the first B press so the arm glides from its current pose to
        the controller's absolute pose instead of snapping there at full speed.
        """
        with self._lock:
            if self._state != FOLLOW:
                raise RuntimeError(f"cannot enable follow from {self._state}")
            self._follow_enabled = True
            self._state = RESUMING
            self._error = None

    def observe_button(self, pressed: bool) -> bool:
        """Queue one toggle on a rising edge and return whether it fired."""
        with self._lock:
            fired = bool(pressed) and not self._button_was_pressed
            self._button_was_pressed = bool(pressed)
            if fired:
                self._toggle_requested = True
            return fired

    def request_toggle(self) -> None:
        with self._lock:
            self._toggle_requested = True

    def consume_toggle(self) -> bool:
        with self._lock:
            requested = self._toggle_requested
            self._toggle_requested = False
            return requested

    def begin_hold(self, current_q: np.ndarray, now: Optional[float] = None) -> None:
        q = np.asarray(current_q, dtype=float)
        if q.shape != (14,) or not np.all(np.isfinite(q)):
            raise ValueError("hold target must contain 14 finite joint values")
        with self._lock:
            if self._state != FOLLOW:
                raise RuntimeError(f"cannot begin hold from {self._state}")
            self._hold_q = q.copy()
            self._state = SETTLING
            self._stable_since = None
            self._q_history.clear()
            self._captured_frames = 0
            self._error = None
            self._fatal_error = False
            self._settling_window_speed = 0.0
            self._settling_window_span = 0.0
            self._settling_raw_dq = 0.0
            self._settling_progress = 0.0
            self._settling_blocker = "collecting_window"

    def hold_for_safety(self, current_q: np.ndarray, message: str) -> bool:
        """Latch HOLD at the given joints when XR input is lost mid-follow.

        Unlike ``begin_hold`` this does not enter SETTLING (no capture is
        triggered).  The operator must press B after XR reconnects, which goes
        through the smooth RESUMING transition instead of letting the IK
        target jump to wherever the controller ended up.
        """
        q = np.asarray(current_q, dtype=float)
        if q.shape != (14,) or not np.all(np.isfinite(q)):
            return False
        with self._lock:
            if self._state != FOLLOW or not self._follow_enabled:
                return False
            self._hold_q = q.copy()
            self._state = HOLD
            self._error = str(message)
            self._fatal_error = False
            self._stable_since = None
            self._q_history.clear()
            self._captured_frames = 0
            return True

    def check_hold_drift(self, current_q: np.ndarray) -> bool:
        """Latch a fatal HOLD error if measured joints leave the commanded hold pose."""
        q = np.asarray(current_q, dtype=float)
        with self._lock:
            if self._state in {FOLLOW, RESUMING} or self._hold_q is None:
                return False
            if self._fatal_error:
                return False
            if q.shape != (14,) or not np.all(np.isfinite(q)):
                self._state = HOLD
                self._error = "Hold safety failed: measured joints are invalid."
                self._fatal_error = True
                return True
            error = float(np.max(np.abs(q[-7:] - self._hold_q[-7:])))
            if error <= self.max_hold_error:
                return False
            self._state = HOLD
            self._error = (
                f"Hold drift {error:.5f} rad exceeds {self.max_hold_error:.5f} rad; "
                "capture blocked, stop teleoperation."
            )
            self._fatal_error = True
            self._stable_since = None
            self._q_history.clear()
            return True

    def _append_q(self, now: float, q: np.ndarray) -> None:
        self._q_history.append((now, q.copy()))
        cutoff = now - self.settle_seconds
        while self._q_history and self._q_history[0][0] < cutoff:
            self._q_history.popleft()

    def update_settling(
        self,
        current_q: np.ndarray,
        current_dq: np.ndarray,
        now: Optional[float] = None,
    ) -> bool:
        """Advance after a stable measured-position window.

        H2 ``motor_state.dq`` occasionally contains isolated spikes while the
        encoder position is stationary.  Using every raw dq sample as a hard
        gate can therefore reset settling forever.  The capture gate derives
        speed from the right-arm encoder positions across the full window and
        also checks their peak-to-peak span.  ``current_dq`` is retained only
        for diagnostics and API compatibility.
        """
        q = np.asarray(current_q, dtype=float)
        dq = np.asarray(current_dq, dtype=float)
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            if self._state != SETTLING:
                return False
            if q.shape != (14,) or not np.all(np.isfinite(q)):
                self._stable_since = None
                self._q_history.clear()
                self._settling_progress = 0.0
                self._settling_blocker = "invalid_position"
                return False
            self._settling_raw_dq = (
                float(np.max(np.abs(dq[-7:])))
                if dq.shape == (14,) and np.all(np.isfinite(dq))
                else float("nan")
            )
            if self._stable_since is None:
                self._stable_since = now
            self._append_q(now, q)
            elapsed = now - self._stable_since
            self._settling_progress = min(1.0, max(0.0, elapsed / self.settle_seconds))
            if elapsed + 1e-9 < self.settle_seconds or len(self._q_history) < 2:
                self._settling_blocker = "collecting_window"
                return False
            times = np.asarray([item[0] for item in self._q_history], dtype=float)
            q_values = np.stack([item[1] for item in self._q_history])
            window_seconds = max(float(times[-1] - times[0]), 1e-9)
            speed = float(
                np.max(np.abs(q_values[-1, -7:] - q_values[0, -7:]))
                / window_seconds
            )
            span = float(np.max(np.ptp(q_values[:, -7:], axis=0)))
            self._settling_window_speed = speed
            self._settling_window_span = span
            if speed > self.max_joint_speed or span > self.max_joint_span:
                self._stable_since = now
                self._q_history.clear()
                self._append_q(now, q)
                self._settling_progress = 0.0
                self._settling_blocker = (
                    "window_speed" if speed > self.max_joint_speed else "window_span"
                )
                return False
            self._state = CAPTURING
            self._captured_frames = 0
            self._settling_blocker = None
            return True

    def set_capture_progress(self, captured_frames: int) -> None:
        with self._lock:
            self._captured_frames = max(0, int(captured_frames))

    def begin_saving(self) -> None:
        with self._lock:
            if self._state != CAPTURING:
                raise RuntimeError(f"cannot begin saving from {self._state}")
            self._state = SAVING

    def finish_saving(self, sample_path: str) -> None:
        with self._lock:
            if self._state != SAVING:
                raise RuntimeError(f"cannot finish saving from {self._state}")
            self._state = HOLD
            self._saved_samples += 1
            self._last_sample = str(sample_path)
            self._error = None

    def fail(self, message: str, *, fatal: bool = False) -> None:
        with self._lock:
            self._state = HOLD
            self._error = str(message)
            self._fatal_error = bool(fatal)

    def preview_rebase(
        self,
        left_xr: np.ndarray,
        right_xr: np.ndarray,
        left_robot: np.ndarray,
        right_robot: np.ndarray,
    ) -> RebaseAnchors:
        with self._lock:
            if self._state != HOLD:
                raise RuntimeError(f"cannot rebase from {self._state}")
        return RebaseAnchors(
            left_robot=_pose(left_robot),
            right_robot=_pose(right_robot),
            left_xr=_pose(left_xr),
            right_xr=_pose(right_xr),
        )

    def initialize_rebase(
        self,
        left_xr: np.ndarray,
        right_xr: np.ndarray,
        left_robot: np.ndarray,
        right_robot: np.ndarray,
    ) -> None:
        """Anchor initial XR motion to the robot's current measured wrist poses."""
        anchors = RebaseAnchors(
            left_robot=_pose(left_robot),
            right_robot=_pose(right_robot),
            left_xr=_pose(left_xr),
            right_xr=_pose(right_xr),
        )
        with self._lock:
            if self._state != FOLLOW:
                raise RuntimeError(f"cannot initialize rebase from {self._state}")
            self._anchors = anchors

    def commit_rebase(self, anchors: RebaseAnchors) -> None:
        with self._lock:
            if self._state != HOLD:
                raise RuntimeError(f"cannot commit rebase from {self._state}")
            self._anchors = anchors
            self._resume_follow()

    def resume_without_rebase(self) -> None:
        """Resume deterministic replay after HOLD without consulting XR poses."""
        with self._lock:
            if self._state != HOLD:
                raise RuntimeError(f"cannot resume from {self._state}")
            self._resume_follow()

    def begin_resume(self) -> None:
        with self._lock:
            if self._state != HOLD:
                raise RuntimeError(f"cannot begin resume from {self._state}")
            self._state = RESUMING
            self._error = None

    def finish_resume(self) -> None:
        with self._lock:
            if self._state != RESUMING:
                raise RuntimeError(f"cannot finish resume from {self._state}")
            self._resume_follow()

    def _resume_follow(self) -> None:
        """Reset hold bookkeeping. Caller must own _lock."""
        self._state = FOLLOW
        self._hold_q = None
        self._stable_since = None
        self._q_history.clear()
        self._captured_frames = 0
        self._error = None
        self._fatal_error = False
        self._settling_window_speed = 0.0
        self._settling_window_span = 0.0
        self._settling_raw_dq = 0.0
        self._settling_progress = 0.0
        self._settling_blocker = None

    def apply_rebase(self, left_xr: np.ndarray, right_xr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            anchors = self._anchors
        if anchors is None:
            return _pose(left_xr), _pose(right_xr)
        return anchors.apply(left_xr, right_xr)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "HAND_EYE_STATE": self._state,
                "HAND_EYE_CAPTURED_FRAMES": self._captured_frames,
                "HAND_EYE_BURST_FRAMES": self.burst_frames,
                "HAND_EYE_SAVED_SAMPLES": self._saved_samples,
                "HAND_EYE_LAST_SAMPLE": self._last_sample,
                "HAND_EYE_ERROR": self._error,
                "HAND_EYE_FATAL_ERROR": self._fatal_error,
                "HAND_EYE_FOLLOW_ENABLED": self._follow_enabled,
                "HAND_EYE_SETTLING_WINDOW_SPEED": self._settling_window_speed,
                "HAND_EYE_SETTLING_WINDOW_SPAN": self._settling_window_span,
                "HAND_EYE_SETTLING_RAW_DQ": self._settling_raw_dq,
                "HAND_EYE_SETTLING_PROGRESS": self._settling_progress,
                "HAND_EYE_SETTLING_BLOCKER": self._settling_blocker,
                "HAND_EYE_MAX_JOINT_SPEED": self.max_joint_speed,
                "HAND_EYE_MAX_JOINT_SPAN": self.max_joint_span,
            }
