"""Align raw Gemini depth to the head RGB frame for VR preview."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_DEFAULT_CALIB = Path(
    "/home/robot/yx/project/IK_replay/config/camera/orbbec_rgbd_calibration.json"
)
_DEFAULT_ALIGNER_ROOT = Path("/home/robot/yx/project/calib/hand_eye_3D")

_aligner = None
_aligner_error: Optional[str] = None


def _load_aligner():
    global _aligner, _aligner_error
    if _aligner is not None or _aligner_error is not None:
        return _aligner
    calib_path = Path(os.environ.get("RGBD_CALIB", _DEFAULT_CALIB))
    root = Path(os.environ.get("HAND_EYE_3D_ROOT", _DEFAULT_ALIGNER_ROOT))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from backend.rgbd import RGBDCalibration, SoftwareDepthAligner
        _aligner = SoftwareDepthAligner(RGBDCalibration.from_file(calib_path))
    except Exception as exc:
        _aligner_error = str(exc)
        _aligner = None
    return _aligner


def align_depth_to_color(depth: np.ndarray) -> tuple[np.ndarray, Optional[str]]:
    """Return depth in the color camera frame, or the raw image plus an error."""
    aligner = _load_aligner()
    if aligner is None:
        return np.asarray(depth), _aligner_error or "depth aligner unavailable"
    aligned = aligner.align(np.asarray(depth))
    if aligned.dtype != np.uint16:
        mm = np.clip(np.nan_to_num(aligned, nan=0.0), 0, 65535)
        aligned = np.round(mm).astype(np.uint16)
    return aligned, None
