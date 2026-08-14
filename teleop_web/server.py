#!/usr/bin/env python3
"""Small, dependency-light web console for xr_teleoperate.

The HTTP server uses the Python standard library. Runtime teleoperation control
uses the existing ZeroMQ IPC endpoints exposed by teleop_hand_and_arm.py.
"""

from __future__ import annotations

import argparse
import http.client
import ipaddress
import json
import math
import mimetypes
import os
import re
import signal
import shlex
import shutil
import ssl
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urlparse

from teleop.robot_control.end_effectors import (
    PASSIVE_END_EFFECTORS,
    SIDE_END_EFFECTORS,
    SINGLE_SIDE_ACTIVE_END_EFFECTORS,
    canonical_end_effector,
)
from teleop_web.training_prep import TrainingPrepError, TrainingPrepManager, training_data_root

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow is expected in runtime envs.
    Image = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = PROJECT_ROOT.name
TELEOP_ROOT = PROJECT_ROOT / "teleop"
TELEIMAGER_SRC_ROOT = TELEOP_ROOT / "teleimager" / "src"
for import_root in (TELEOP_ROOT, TELEIMAGER_SRC_ROOT):
    if import_root.exists() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
STATIC_ROOT = Path(__file__).resolve().parent / "static"
ENTRYPOINT = PROJECT_ROOT / "teleop" / "teleop_hand_and_arm.py"
CONVERT_ENTRYPOINT = PROJECT_ROOT / "tools" / "convert_h2_to_lerobot.py"
PACKAGE_ENTRYPOINT = PROJECT_ROOT / "tools" / "package_lerobot_dataset.py"


def default_project_path(env_name: str, *parts: str) -> Path:
    configured = os.environ.get(env_name)
    if configured:
        return Path(configured).expanduser()
    return PROJECT_ROOT.joinpath(*parts)


def default_data_path(env_name: str, *parts: str) -> Path:
    configured = os.environ.get(env_name)
    if configured:
        return Path(configured).expanduser()
    base = Path(os.environ.get("XR_TELEOP_DATA_DIR", Path.home() / "data")).expanduser()
    return base.joinpath(*parts)


DEFAULT_DATA_DIR = Path(os.environ.get("XR_TELEOP_DATA_DIR", Path.home() / "data")).expanduser()
LEGACY_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_ROOT = default_data_path("XR_TELEOP_DATASET_DIR", "datasets", "robot")
DEFAULT_OPENPI_DIR = Path(os.environ.get("XR_TELEOP_OPENPI_DIR", Path.home() / "openpi")).expanduser()
DEFAULT_LEROBOT_HOME = default_data_path("HF_LEROBOT_HOME", "datasets", "lerobot")
DEFAULT_OPENPI_WORK_DIR = default_data_path("XR_TELEOP_OPENPI_WORK_DIR", "datasets", "openpi")
DEFAULT_OPENPI_ASSETS_DIR = DEFAULT_OPENPI_WORK_DIR / "assets"
DEFAULT_OPENPI_CONFIG_NAME = os.environ.get("XR_TELEOP_OPENPI_CONFIG_NAME", "pi05_h2_lerobot")
DEFAULT_CONFIG_FILE = default_project_path("XR_TELEOP_WEB_CONFIG", "config", "web_console.json")
DEFAULT_LOG_DIR = default_project_path("XR_TELEOP_WEB_LOG_DIR", "logs")
DEFAULT_DELIVERY_TEMPLATES_FILE = PROJECT_ROOT / "config" / "delivery_templates.json"


DEFAULT_DELIVERY_TEMPLATES: dict[str, Any] = {
    "version": 5,
    "templates": [
        {
            "id": "proxy_pull_dataset",
            "section": "data_upload",
            "title": "代理机：从 OSS 拉取数据包",
            "description": "在代理中转服务器执行，目录不存在会自动创建。",
            "body": """TASK_NAME='{{TASK_NAME}}'
DATA_FILE='{{DATA_PACKAGE}}'
OSS_URI='{{DATA_OSS_URI}}'

mkdir -p "/opt/packages/openpi/${TASK_NAME}"
ossutil cp -r "$OSS_URI" "/opt/packages/openpi/${TASK_NAME}/"
""",
        },
        {
            "id": "sync_dataset_to_train",
            "section": "data_upload",
            "title": "代理机：同步数据包到训练服务器",
            "description": "在代理中转服务器执行，将压缩包发送到训练服务器。",
            "body": """TASK_NAME='{{TASK_NAME}}'
DATA_FILE='{{DATA_PACKAGE}}'
TRAIN_HOST='dgzs-docker-gpu11.prod-2227'

ssh "$TRAIN_HOST" "mkdir -p /home/ubuntu/packages/openpi/${TASK_NAME}"
scp "/opt/packages/openpi/${TASK_NAME}/$DATA_FILE" "$TRAIN_HOST:/home/ubuntu/packages/openpi/${TASK_NAME}/"
""",
        },
        {
            "id": "extract_dataset_on_train",
            "section": "data_upload",
            "title": "训练服务器：解压 LeRobot 数据集",
            "description": "在训练服务器执行，解压到 HF_LEROBOT_HOME 对应目录。",
            "body": """TASK_NAME='{{TASK_NAME}}'
DATA_FILE='{{DATA_PACKAGE}}'

mkdir -p "/home/ubuntu/datasets/lerobot"
tar -zxvf "/home/ubuntu/packages/openpi/${TASK_NAME}/$DATA_FILE" \\
  -C "/home/ubuntu/datasets/lerobot"
""",
        },
        {
            "id": "tmux_session",
            "section": "training",
            "title": "训练服务器：创建 tmux 会话",
            "description": "训练建议放在 tmux 内执行，网络断开不影响训练进程。",
            "body": """tmux new -s openpi
tmux ls
tmux attach -t openpi

# 退出并删除会话：在 tmux 内执行 exit
""",
        },
        {
            "id": "run_training",
            "section": "training",
            "title": "训练服务器：启动训练",
            "description": "在训练服务器 tmux 会话内执行。",
            "body": """cd "/home/ubuntu/openpi"

env \\
  UV_NO_SYNC=1 \\
  HF_LEROBOT_HOME="/home/ubuntu/datasets/lerobot" \\
  OPENPI_DATA_HOME="/home/ubuntu/models/openpi" \\
  HF_HOME="/home/ubuntu/models/openpi/huggingface" \\
  OPENPI_H2_REPO_ID="{{REPO_ID}}" \\
  OPENPI_H2_ACTION_DIM="{{ACTION_DIM}}" \\
  OPENPI_H2_REAL_ACTION_DIM="{{REAL_ACTION_DIM}}" \\
  OPENPI_H2_ACTION_HORIZON="{{ACTION_HORIZON}}" \\
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \\
  XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0" \\
  uv run scripts/train.py \\
    "{{CONFIG_NAME}}" \\
    --exp-name "{{EXP_NAME}}" \\
    --assets-base-dir "/home/ubuntu/assets" \\
    --checkpoint-base-dir "/home/ubuntu/models/openpi/checkpoints" \\
    --fsdp-devices "{{FSDP_DEVICES}}" \\
    --batch-size "{{BATCH_SIZE}}" \\
    --num-train-steps "{{NUM_TRAIN_STEPS}}" \\
    --save-interval "{{SAVE_INTERVAL}}" \\
    --keep-period "{{KEEP_PERIOD}}" \\
    --overwrite \\
    --no-wandb-enabled \\
  2>&1 | tee -a "/home/ubuntu/models/openpi/logs/{{EXP_NAME}}_{{PACKAGE_TIMESTAMP}}.log" 2>&1
""",
        },
        {
            "id": "package_model_to_oss",
            "section": "model_return",
            "title": "训练服务器：打包并上传模型",
            "description": "训练完成后执行，将模型压缩包上传到 OSS 任务目录。",
            "body": """TASK_NAME='{{TASK_NAME}}'
MODEL_FILE='{{MODEL_FILE}}'
OSS_URI='{{MODEL_OSS_URI}}'
MODEL_TRAIN_DIR='{{MODEL_TRAIN_DIR}}'

mkdir -p "/home/ubuntu/packages/openpi/${TASK_NAME}/models"
tar -czvf "/home/ubuntu/packages/openpi/${TASK_NAME}/models/$MODEL_FILE" \\
  -C "/home/ubuntu/models/openpi/checkpoints/{{CONFIG_NAME}}/{{EXP_NAME}}" "$MODEL_TRAIN_DIR"
ossutil cp -r "/home/ubuntu/packages/openpi/${TASK_NAME}/models/$MODEL_FILE" "$OSS_URI"
""",
        },
        {
            "id": "proxy_pull_model",
            "section": "model_return",
            "title": "代理机：从 OSS 拉取模型文件",
            "description": "在代理中转服务器执行，文件名填写训练完成后上传到 OSS 的模型包。",
            "body": """TASK_NAME='{{TASK_NAME}}'
MODEL_FILE='{{MODEL_FILE}}'
OSS_URI='{{MODEL_OSS_URI}}'

mkdir -p "/opt/packages/openpi/${TASK_NAME}/models"
ossutil cp -r "$OSS_URI" "/opt/packages/openpi/${TASK_NAME}/models/"
""",
        },
        {
            "id": "sync_model_to_target",
            "section": "model_return",
            "title": "代理机：同步模型到部署服务器",
            "description": "在代理中转服务器执行，目标服务器按实际部署环境修改。",
            "body": """TASK_NAME='{{TASK_NAME}}'
MODEL_FILE='{{MODEL_FILE}}'
TARGET_HOST='robot@192.168.61.228'
TARGET_DIR='/data03/data/models/openpi_downloads/${TASK_NAME}'

ssh "$TARGET_HOST" "mkdir -p $TARGET_DIR"
scp "/opt/packages/openpi/${TASK_NAME}/models/$MODEL_FILE" "$TARGET_HOST:$TARGET_DIR/"
""",
        },
    ],
}


def default_delivery_templates_payload() -> dict[str, Any]:
    try:
        payload = json.loads(DEFAULT_DELIVERY_TEMPLATES_FILE.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("templates"), list):
            return payload
    except (OSError, json.JSONDecodeError):
        pass
    return DEFAULT_DELIVERY_TEMPLATES

POSTPROCESS_TASK_KEYS = {
    "postprocess_repo_id",
    "postprocess_robot_type",
    "postprocess_camera_map",
    "postprocess_image_size",
    "postprocess_image_encoding",
    "postprocess_jpeg_quality",
    "postprocess_video_backend",
    "postprocess_image_writer_processes",
    "postprocess_image_writer_threads",
    "postprocess_resume",
    "postprocess_overwrite",
    "postprocess_batch_size",
    "postprocess_start_episode",
    "postprocess_config_name",
    "postprocess_openpi_dir",
    "postprocess_max_frames",
    "postprocess_options_updated_at",
    "convert_started_at",
    "convert_exit_code",
    "convert_failed_at",
    "converted_at",
    "last_convert_record",
    "normalize_started_at",
    "normalize_exit_code",
    "normalize_failed_at",
    "normalized_at",
    "last_normalize_record",
    "package_started_at",
    "package_exit_code",
    "package_failed_at",
    "packaged_at",
    "last_package_record",
    "last_package",
    "data_package_started_at",
    "data_package_exit_code",
    "data_package_failed_at",
    "data_packaged_at",
    "last_data_package_record",
    "last_data_package",
    "assets_package_started_at",
    "assets_package_exit_code",
    "assets_package_failed_at",
    "assets_packaged_at",
    "last_assets_package_record",
    "last_assets_package",
}
POSTPROCESS_STDOUT_DRAIN_SECONDS = 5.0
POSTPROCESS_TERMINATE_GRACE_SECONDS = 3.0
POSTPROCESS_INTERRUPTION_GRACE_SECONDS = 20.0
POSTPROCESS_JOB_LOG_LIMIT = 200
POSTPROCESS_STORE_RECENT_LIMIT = 120
POSTPROCESS_PROGRESS_RE = re.compile(r"(\d{1,3})%\|")
POSTPROCESS_RATIO_RE = re.compile(r"\|\s*(\d+)\s*/\s*(\d+)\s*\[")
POSTPROCESS_STAGE_RE = re.compile(r"^([^:]{2,64}):\s*")
POSTPROCESS_SPEED_RE = re.compile(r",\s*([0-9.]+\s*[^,\]]+/s)\]")
POSTPROCESS_SAVED_EPISODE_RE = re.compile(r"Saved episode\s+(\d+)\s*/\s*(\d+)")
POSTPROCESS_PACKAGE_RE = re.compile(r"Packaging files:\s*(\d{1,3})%\s*\|\s*(\d+)\s*/\s*(\d+)")
OSS_PROGRESS_PERCENT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
OSS_PROGRESS_SIZE_RE = re.compile(
    r"([0-9.]+\s*(?:B|KB|KiB|MB|MiB|GB|GiB|TB|TiB))\s*/\s*([0-9.]+\s*(?:B|KB|KiB|MB|MiB|GB|GiB|TB|TiB))",
    re.IGNORECASE,
)
OSS_PROGRESS_SPEED_RE = re.compile(r"([0-9.]+\s*(?:B|KB|KiB|MB|MiB|GB|GiB|TB|TiB)/s)", re.IGNORECASE)

ARMS = {"G1_29", "G1_23", "H1_2", "H1", "H2"}
END_EFFECTORS = set(SIDE_END_EFFECTORS)
INPUT_MODES = {"hand", "controller"}
DISPLAY_MODES = {"immersive", "ego", "pass-through"}
XR_VIEW_MODES = {"head", "quad"}
ARM_REFERENCE_MODES = {"world", "head_position", "head_yaw"}
TASK_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
CONFIG_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
CAMERA_MAP_RE = re.compile(r"^color_\d+:[A-Za-z][A-Za-z0-9_]*(,color_\d+:[A-Za-z][A-Za-z0-9_]*)*$")
DEFAULT_IMAGE_SERVER_IP = os.environ.get("XR_TELEOP_DEFAULT_IMAGE_SERVER_IP", "192.168.123.5")
DEFAULT_WEBRTC_SERVER_IP = os.environ.get("XR_TELEOP_DEFAULT_WEBRTC_SERVER_IP", "192.168.61.142")
DEFAULT_WEBRTC_SCHEME = os.environ.get("XR_TELEOP_WEBRTC_SCHEME", "https").strip().lower()
if DEFAULT_WEBRTC_SCHEME not in {"http", "https"}:
    DEFAULT_WEBRTC_SCHEME = "http"
DEFAULT_CAMERA_MAP = "auto"
DEFAULT_LEROBOT_PYTHON = Path(os.environ.get("XR_TELEOP_LEROBOT_PYTHON", "/home/robot/miniconda3/envs/lerobot/bin/python")).expanduser()
DEFAULT_OPENPI_PYTHON = os.environ.get("XR_TELEOP_OPENPI_PYTHON", "").strip()
DEFAULT_NETWORK_INTERFACE = os.environ.get("XR_TELEOP_DEFAULT_NETWORK_INTERFACE", "enp86s0").strip() or "enp86s0"
DEFAULT_H2_INIT_ARM_POSE_FILE = os.environ.get(
    "XR_TELEOP_DEFAULT_H2_INIT_ARM_POSE_FILE",
    "config/h2_pose_init.json",
).strip()
DEFAULT_IK_REPLAY_LIVE_URL = os.environ.get(
    "IK_REPLAY_LIVE_URL",
    "http://192.168.61.228:8000/api/live/state",
).strip()
LEGACY_NETWORK_INTERFACES = {"eth0", "enp87s0"}

DEFAULT_DEVICE = {
    "arm": "H2",
    "ee": "none",
    "left_ee": "none",
    "right_ee": "inspire_dfx",
    "input_mode": "hand",
    "display_mode": "pass-through",
    "xr_view": "head",
    "arm_reference_mode": "head_position",
    "img_server_ip": DEFAULT_IMAGE_SERVER_IP,
    "webrtc_server_ip": DEFAULT_WEBRTC_SERVER_IP,
    "data_dir": str(DEFAULT_DATA_DIR),
    "network_interface": DEFAULT_NETWORK_INTERFACE,
    "frequency": 30,
    "init_arm_pose_file": DEFAULT_H2_INIT_ARM_POSE_FILE,
    "init_arm_pose_duration": 5,
    "headless": False,
    "motion": True,
    "hand_eye_record": False,
    "ik_replay_live_enable": False,
    "ik_replay_live_url": DEFAULT_IK_REPLAY_LIVE_URL,
    "ik_replay_live_fps": 10,
}


class ValidationError(ValueError):
    pass


class DailyFileLogger:
    """Tiny daily rotating file logger for the dependency-free web console."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self._lock = threading.RLock()

    def _path_for_today(self) -> Path:
        return self.log_dir / "system" / f"teleop_{time.strftime('%Y-%m-%d')}.log"

    def write(self, level: str, message: str, **fields: Any) -> None:
        with self._lock:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            extra = ""
            if fields:
                try:
                    extra = " " + json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str)
                except TypeError:
                    extra = f" {fields}"
            line = f"{timestamp} [{level.upper()}] {message}{extra}\n"
            try:
                path = self._path_for_today()
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as log_file:
                    log_file.write(line)
            except OSError as exc:
                print(f"{timestamp} [WARNING] failed to write log file {self._path_for_today()}: {exc}")
            print(line, end="")


def _nonempty_string(value: Any, field: str, max_length: int = 200) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} 不能为空")
    result = value.strip()
    if len(result) > max_length:
        raise ValidationError(f"{field} 不能超过 {max_length} 个字符")
    return result


def _english_instruction(value: Any) -> str:
    instruction = " ".join(_nonempty_string(value, "英文 instruction", 500).split())
    if not instruction.isascii():
        raise ValidationError("英文 instruction 仅支持英文、数字、空格和常用英文标点")
    return instruction


def _project_path_for_runtime(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _project_path_for_config(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return value
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def validate_device(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValidationError("设备信息格式不正确")
    device = dict(DEFAULT_DEVICE)
    device.update(raw)

    for field in ("ee", "left_ee", "right_ee"):
        device[field] = canonical_end_effector(device.get(field, "") or "")

    for field, choices in (
        ("arm", ARMS),
        ("ee", END_EFFECTORS),
        ("left_ee", END_EFFECTORS),
        ("right_ee", END_EFFECTORS),
        ("input_mode", INPUT_MODES),
        ("display_mode", DISPLAY_MODES),
        ("xr_view", XR_VIEW_MODES),
        ("arm_reference_mode", ARM_REFERENCE_MODES),
    ):
        if device[field] not in choices:
            raise ValidationError(f"不支持的 {field}: {device[field]}")

    if device["input_mode"] == "hand":
        device["display_mode"] = "pass-through"
        device["xr_view"] = "head"

    if device["arm"] == "H2" and device.get("arm_reference_mode") == "world":
        device["arm_reference_mode"] = "head_position"

    if not str(device.get("webrtc_server_ip", "")).strip():
        device["webrtc_server_ip"] = DEFAULT_WEBRTC_SERVER_IP

    data_dir = str(device.get("data_dir", "") or str(DEFAULT_DATA_DIR)).strip()
    if not data_dir or "\x00" in data_dir:
        raise ValidationError("数据集目录不正确")
    device["data_dir"] = str(Path(data_dir).expanduser())

    if raw.get("ee") and not raw.get("left_ee") and not raw.get("right_ee"):
        device["left_ee"] = device["ee"]
        device["right_ee"] = device["ee"]
    if device["left_ee"] == device["right_ee"]:
        device["ee"] = device["left_ee"]
    else:
        device["ee"] = "none"

    for field in ("img_server_ip", "webrtc_server_ip"):
        value = str(device.get(field, "")).strip()
        try:
            ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValidationError(f"{field} 不是有效的 IP 地址") from exc
        device[field] = value

    interface = _nonempty_string(device.get("network_interface"), "网络接口", 32)
    if not re.fullmatch(r"[a-zA-Z0-9_.:-]+", interface):
        raise ValidationError("网络接口包含不支持的字符")
    device["network_interface"] = interface

    try:
        frequency = float(device.get("frequency", 30))
    except (TypeError, ValueError) as exc:
        raise ValidationError("控制频率必须是数字") from exc
    if not 1 <= frequency <= 240:
        raise ValidationError("控制频率必须在 1 到 240 Hz 之间")
    device["frequency"] = frequency
    init_pose_file = str(device.get("init_arm_pose_file", "") or "").strip()
    if init_pose_file and "\x00" in init_pose_file:
        raise ValidationError("初始姿态文件路径不正确")
    if not init_pose_file and device.get("arm") == "H2":
        init_pose_file = DEFAULT_H2_INIT_ARM_POSE_FILE
    device["init_arm_pose_file"] = _project_path_for_config(init_pose_file) if init_pose_file else ""
    try:
        init_pose_duration = float(device.get("init_arm_pose_duration", 5))
    except (TypeError, ValueError) as exc:
        raise ValidationError("初始姿态移动时长必须是数字") from exc
    if not 0.5 <= init_pose_duration <= 60:
        raise ValidationError("初始姿态移动时长必须在 0.5 到 60 秒之间")
    device["init_arm_pose_duration"] = init_pose_duration
    device["headless"] = bool(device.get("headless", False))
    device["motion"] = bool(device.get("motion", False))
    device["hand_eye_record"] = bool(device.get("hand_eye_record", False))
    if device["hand_eye_record"] and device["arm"] != "H2":
        raise ValidationError("手眼标定采集模式目前仅支持 H2")
    device["ik_replay_live_enable"] = bool(device.get("ik_replay_live_enable", False))
    ik_replay_live_url = str(device.get("ik_replay_live_url", "") or "").strip()
    if device["ik_replay_live_enable"]:
        parsed_live_url = urlparse(ik_replay_live_url)
        if parsed_live_url.scheme not in {"http", "https"} or not parsed_live_url.netloc:
            raise ValidationError("IK 回放推送地址必须是 http(s) URL")
    device["ik_replay_live_url"] = ik_replay_live_url
    try:
        ik_replay_live_fps = float(device.get("ik_replay_live_fps", 10))
    except (TypeError, ValueError) as exc:
        raise ValidationError("IK 回放推送频率必须是数字") from exc
    if not 0.1 <= ik_replay_live_fps <= 60:
        raise ValidationError("IK 回放推送频率必须在 0.1 到 60 Hz 之间")
    device["ik_replay_live_fps"] = ik_replay_live_fps
    left_active = device["left_ee"] not in PASSIVE_END_EFFECTORS
    right_active = device["right_ee"] not in PASSIVE_END_EFFECTORS
    if device["input_mode"] == "controller" and (left_active or right_active):
        raise ValidationError("手柄模式不支持灵巧手；请将末端执行器设为不控制，或改用手部追踪模式")
    active_types = {value for value in (device["left_ee"], device["right_ee"]) if value not in PASSIVE_END_EFFECTORS}
    if len(active_types) > 1:
        raise ValidationError("左右末端执行器必须一致，或一侧主动、一侧被动；不支持两个不同主动末端")
    if left_active != right_active and next(iter(active_types), None) not in SINGLE_SIDE_ACTIVE_END_EFFECTORS:
        supported = " / ".join(sorted(SINGLE_SIDE_ACTIVE_END_EFFECTORS))
        raise ValidationError(f"单侧主动控制目前仅支持 {supported}")
    return device


def validate_task(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValidationError("任务信息格式不正确")
    name = _nonempty_string(raw.get("name"), "英文任务名", 64)
    if not TASK_NAME_RE.fullmatch(name):
        raise ValidationError("英文任务名仅支持字母、数字、下划线和连字符，且必须以字母或数字开头")
    instruction = _english_instruction(raw.get("instruction"))
    description = _nonempty_string(raw.get("description"), "中文任务描述", 500)
    return {"name": name, "instruction": instruction, "description": description}


def _qpos_from_arm_group(grouped: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(grouped, dict):
        return None
    left = grouped.get("left_arm")
    right = grouped.get("right_arm")
    if isinstance(left, dict):
        left = left.get("qpos")
    if isinstance(right, dict):
        right = right.get("qpos")
    if left is None or right is None:
        return None
    return [*left, *right]


def validate_ik_replay_target(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValidationError("IK 回放目标格式不正确")
    target = raw.get("target_q") or raw.get("target_joints") or raw.get("qpos")
    if target is None:
        target = _qpos_from_arm_group(raw.get("actions")) or _qpos_from_arm_group(raw.get("states"))
    if target is None:
        raise ValidationError("IK 回放目标缺少 14 维 target_q/target_joints/qpos")
    try:
        target_q = [float(value) for value in target]
    except (TypeError, ValueError) as exc:
        raise ValidationError("IK 回放目标必须是数字数组") from exc
    if len(target_q) != 14:
        raise ValidationError(f"IK 回放目标必须是 14 维，当前为 {len(target_q)}")
    return {"target_q": target_q, "source": str(raw.get("source") or "ik_replay")[:80]}


def data_dir_from_config(config_file: Path) -> Path | None:
    try:
        payload = json.loads(config_file.expanduser().read_text(encoding="utf-8"))
        data_dir = (((payload.get("device") or {}).get("config") or {}).get("data_dir"))
    except (OSError, AttributeError, json.JSONDecodeError):
        return None
    if not data_dir:
        return None
    try:
        return Path(str(data_dir)).expanduser()
    except (TypeError, ValueError):
        return None


def build_command(device: dict[str, Any], task: dict[str, str], dataset_root: Path) -> list[str]:
    python = os.environ.get("XR_TELEOP_PYTHON", sys.executable)
    command = [
        python,
        str(ENTRYPOINT),
        f"--arm={device['arm']}",
        f"--input-mode={device['input_mode']}",
        f"--display-mode={device['display_mode']}",
        f"--xr-view={device['xr_view']}",
        f"--arm-reference-mode={device['arm_reference_mode']}",
        f"--img-server-ip={device['img_server_ip']}",
        f"--network-interface={device['network_interface']}",
        f"--frequency={device['frequency']:g}",
        "--record",
        "--ipc",
        f"--task-dir={dataset_root}",
        f"--task-name={task['name']}",
        f"--task-goal={task['instruction']}",
        f"--task-desc={task['description']}",
    ]
    if device["left_ee"] == "rubber" and device["right_ee"] == "rubber":
        command.insert(3, f"--right-ee={device['right_ee']}")
        command.insert(3, f"--left-ee={device['left_ee']}")
    elif device["left_ee"] != device["right_ee"]:
        command.insert(3, f"--right-ee={device['right_ee']}")
        command.insert(3, f"--left-ee={device['left_ee']}")
    elif device["ee"] not in PASSIVE_END_EFFECTORS:
        command.insert(3, f"--ee={device['ee']}")
    if device["webrtc_server_ip"]:
        command.append(f"--webrtc-server-ip={device['webrtc_server_ip']}")
    command.append(f"--webrtc-scheme={DEFAULT_WEBRTC_SCHEME}")
    if device.get("init_arm_pose_file"):
        command.append(f"--init-arm-pose-file={_project_path_for_runtime(str(device['init_arm_pose_file']))}")
        command.append(f"--init-arm-pose-duration={device.get('init_arm_pose_duration', 5):g}")
    if device["headless"]:
        command.append("--headless")
    if device["motion"]:
        command.append("--motion")
    if device.get("hand_eye_record"):
        command.append("--hand-eye-record")
    if device.get("ik_replay_live_enable") and device.get("ik_replay_live_url"):
        command.append("--ik-replay-live-enable")
        command.append(f"--ik-replay-live-url={device['ik_replay_live_url']}")
        command.append(f"--ik-replay-live-fps={device.get('ik_replay_live_fps', 10):g}")
    return command


def display_command(command: list[str]) -> str:
    return shlex.join(command)


def validate_runtime_network_interface(interface: str) -> None:
    sys_net = Path("/sys/class/net")
    if not sys_net.exists():
        return
    iface_dir = sys_net / interface
    if not iface_dir.exists():
        raise ValidationError(
            f"DDS network interface {interface!r} does not exist on this host."
        )
    try:
        state = (iface_dir / "operstate").read_text(encoding="utf-8").strip().lower()
    except OSError:
        return
    if state not in {"up", "unknown"}:
        raise ValidationError(
            f"DDS network interface {interface!r} is {state}; "
            "connect the robot network cable or update the device config."
        )


def validate_repo_id(value: Any) -> str:
    repo_id = _nonempty_string(value, "LeRobot repo-id", 200)
    if not REPO_ID_RE.fullmatch(repo_id):
        raise ValidationError("LeRobot repo-id 格式应为 owner/name，例如 local/h2_pick_red_cup_01")
    return repo_id


def validate_config_name(value: Any) -> str:
    config_name = _nonempty_string(value, "OpenPI config-name", 128)
    if not CONFIG_NAME_RE.fullmatch(config_name):
        raise ValidationError("OpenPI config-name 仅支持字母、数字、点、下划线和连字符")
    return config_name


def lerobot_action_dim(repo_id: str) -> int:
    metadata_path = DEFAULT_LEROBOT_HOME / repo_id / "metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        action_dim = int(payload.get("action_dim"))
        if action_dim > 0:
            return action_dim
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return 14

def lerobot_dataset_dir(repo_id: str) -> Path:
    return DEFAULT_LEROBOT_HOME / repo_id


def lerobot_dataset_ready(repo_id: str) -> bool:
    root = lerobot_dataset_dir(repo_id)
    return root.is_dir() and (root / "meta" / "info.json").is_file()


def norm_stats_candidates(repo_id: str, config_name: str | None = None) -> list[Path]:
    names = [name for name in (config_name, DEFAULT_OPENPI_CONFIG_NAME, "pi05_h2_pick_red_cup") if name]
    seen: set[str] = set()
    candidates = [DEFAULT_OPENPI_ASSETS_DIR / repo_id]
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        candidates.append(DEFAULT_OPENPI_ASSETS_DIR / name / repo_id)
    return candidates


def norm_stats_path(repo_id: str, config_name: str | None = None) -> Path | None:
    for path in norm_stats_candidates(repo_id, config_name):
        if (path / "norm_stats.json").is_file():
            return path
    return None


def validate_camera_map(value: Any) -> str:
    camera_map = str(value or DEFAULT_CAMERA_MAP).strip()
    camera_map = re.sub(r"\s+", "", camera_map)
    if camera_map.lower() == "auto":
        return "auto"
    if not CAMERA_MAP_RE.fullmatch(camera_map):
        raise ValidationError("相机映射格式不正确，例如 color_0:image,color_1:torso_image")
    sources = []
    targets = []
    for item in camera_map.split(","):
        source, target = item.split(":", 1)
        sources.append(source)
        targets.append(target)
    if len(sources) != len(set(sources)):
        raise ValidationError("相机映射中存在重复 color_x")
    if len(targets) != len(set(targets)):
        raise ValidationError("相机映射中存在重复目标字段")
    return camera_map


def dataset_color_keys(task_dir: Path, max_episodes: int = 8, start_episode: int = 0) -> set[str]:
    colors: set[str] = set()
    if not task_dir.is_dir():
        return colors
    episode_dirs = sorted(
        (child for child in task_dir.iterdir() if child.is_dir() and re.fullmatch(r"episode_\d+", child.name)),
        key=lambda path: path.name,
    )
    start_episode = max(0, int(start_episode or 0))
    for episode_dir in episode_dirs[start_episode : start_episode + max_episodes]:
        data_file = episode_dir / "data.json"
        try:
            payload = json.loads(data_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        frames = payload.get("data")
        if not isinstance(frames, list):
            continue
        for frame in frames[:20]:
            frame_colors = frame.get("colors") if isinstance(frame, dict) else None
            if isinstance(frame_colors, dict):
                colors.update(str(key) for key in frame_colors)
        if colors:
            break
    return colors


def validate_camera_map_sources(camera_map: str, task_dir: Path, *, start_episode: int = 0) -> None:
    if camera_map == "auto":
        return
    available = dataset_color_keys(task_dir, start_episode=start_episode)
    if not available:
        return
    sources = {item.split(":", 1)[0] for item in camera_map.split(",") if item}
    missing = sorted(source for source in sources if source not in available)
    if not missing:
        return
    hint = ""
    if "color_3" in missing and "color_2" in available:
        hint = "；该三路数据集通常应选择“右手三路：head + torso(color_1) + right_wrist(color_2)”"
    raise ValidationError(
        f"相机映射引用了数据集中不存在的源相机：{', '.join(missing)}；"
        f"当前数据集包含：{', '.join(sorted(available))}{hint}"
    )


def validate_image_size(value: Any) -> str:
    text = str(value or "original").strip().lower()
    if text in {"", "original", "keep", "source"}:
        return "original"
    if text == "240x320":
        return text
    raise ValidationError("image-size 仅支持原始尺寸或 240x320")


def validate_image_encoding(value: Any) -> str:
    text = str(value or "auto").strip().lower()
    if text == "jpeg":
        return "jpg"
    if text in {"", "auto"}:
        return "auto"
    if text in {"jpg", "png", "video"}:
        return text
    raise ValidationError("image-encoding 仅支持 auto、jpg/jpeg、png 或 video")


def validate_jpeg_quality(value: Any) -> int:
    text = str(value if value is not None else "95").strip()
    if not text:
        return 95
    try:
        quality = int(text)
    except (TypeError, ValueError) as exc:
        raise ValidationError("jpeg-quality 必须是 1-100 的整数") from exc
    if quality < 1 or quality > 100:
        raise ValidationError("jpeg-quality 必须在 1-100 之间")
    return quality


def validate_optional_int(
    value: Any,
    *,
    field_name: str,
    default: int | None = None,
    minimum: int = 0,
    maximum: int = 64,
) -> int | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return default
    try:
        number = int(text)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} 必须是整数") from exc
    if number < minimum or number > maximum:
        raise ValidationError(f"{field_name} 必须在 {minimum}-{maximum} 之间")
    return number


def validate_video_backend(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValidationError("video-backend 仅支持字母、数字、点、下划线和连字符")
    return text


def resolve_openpi_python(openpi_dir: Path) -> Path:
    candidates = []
    if DEFAULT_OPENPI_PYTHON:
        candidates.append(Path(DEFAULT_OPENPI_PYTHON).expanduser())
    candidates.extend([
        openpi_dir / ".venv" / "bin" / "python",
        DEFAULT_LEROBOT_PYTHON,
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise ValidationError(
        "OpenPI Python 环境不存在。请先在 OpenPI 目录执行 uv sync，"
        f"或设置 XR_TELEOP_OPENPI_PYTHON。已检查: {checked}"
    )


def validate_openpi_runtime(
    python_path: Path,
    *,
    openpi_dir: Path,
    cwd: Path,
    env: dict[str, str],
) -> None:
    try:
        result = subprocess.run(
            [str(python_path), "-c", "import openpi.training.data_loader"],
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValidationError(f"无法检查 OpenPI Python 环境 {python_path}: {exc}") from exc
    if result.returncode == 0:
        return
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    detail = output.splitlines()[-1] if output else f"exit code {result.returncode}"
    raise ValidationError(
        f"OpenPI Python 环境与项目依赖不兼容: {python_path}。"
        f"请在 {openpi_dir} 执行 uv sync。错误: {detail}"
    )


def dataset_image_shape(task_dir: Path, max_episodes: int = 8) -> tuple[int, int] | None:
    if not task_dir.is_dir():
        return None
    episode_dirs = sorted(
        (child for child in task_dir.iterdir() if child.is_dir() and re.fullmatch(r"episode_\d+", child.name)),
        key=lambda path: path.name,
    )
    fallback: tuple[int, int] | None = None
    for episode_dir in episode_dirs[:max_episodes]:
        data_file = episode_dir / "data.json"
        try:
            payload = json.loads(data_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        image_info = ((payload.get("info") or {}).get("image") or {})
        if fallback is None:
            try:
                height = int(float(image_info.get("height")))
                width = int(float(image_info.get("width")))
                if height > 0 and width > 0:
                    fallback = (height, width)
            except (TypeError, ValueError):
                pass
        frames = payload.get("data")
        if not isinstance(frames, list):
            continue
        for frame in frames[:20]:
            colors = frame.get("colors") if isinstance(frame, dict) else None
            if not isinstance(colors, dict):
                continue
            for rel_path in colors.values():
                if not isinstance(rel_path, str):
                    continue
                image_path = episode_dir / rel_path
                if Image is None or not image_path.is_file():
                    continue
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                        if height > 0 and width > 0:
                            return height, width
                except OSError:
                    continue
    return fallback


def validate_image_size_for_dataset(image_size: str, task_dir: Path) -> None:
    if image_size == "original":
        return
    source_shape = dataset_image_shape(task_dir)
    if source_shape is None:
        return
    height, width = (int(part) for part in image_size.split("x", 1))
    if height > source_shape[0] or width > source_shape[1]:
        raise ValidationError(
            f"image-size {image_size} 大于原始图像 {source_shape[0]}x{source_shape[1]}，不允许放大转换"
        )


def episode_progress(task_dir: Path) -> dict[str, Any]:
    """Return existing episode progress without trusting unrelated directory names."""
    episode_ids = []
    completed = 0
    if task_dir.exists():
        for child in task_dir.iterdir():
            match = re.fullmatch(r"episode_(\d+)", child.name)
            if child.is_dir() and match:
                episode_ids.append(int(match.group(1)))
                if _is_completed_data_json(child / "data.json"):
                    completed += 1
    # EpisodeWriter starts at 0001 when the task directory already exists.
    last_id = max(episode_ids, default=0)
    return {
        "existing_episodes": completed,
        "last_episode": f"episode_{last_id:04d}" if episode_ids else None,
        "next_episode": f"episode_{last_id + 1:04d}",
        "resuming": bool(episode_ids),
    }


def _is_completed_data_json(data_file: Path) -> bool:
    try:
        with data_file.open("rb") as handle:
            handle.seek(max(0, data_file.stat().st_size - 16))
            tail = handle.read().replace(b"\r\n", b"\n").rstrip()
            return tail.endswith(b"]\n}") or tail.endswith(b"]}")
    except OSError:
        return False


def _safe_relative_file(root: Path, relative_path: str) -> Path:
    if not relative_path or "\x00" in relative_path:
        raise ValidationError("文件路径不正确")
    root_resolved = root.resolve()
    candidate = (root_resolved / relative_path).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValidationError("文件路径超出任务目录")
    if not candidate.is_file():
        raise ValidationError("文件不存在")
    return candidate


def _image_item(task_id: int, episode_dir: Path, name: str, relative_path: str) -> dict[str, Any] | None:
    try:
        image_path = _safe_relative_file(episode_dir, relative_path)
    except ValidationError:
        return None
    return {
        "name": name,
        "url": f"/api/tasks/file?task_id={task_id}&episode={episode_dir.name}&path={relative_path}",
        "size_bytes": image_path.stat().st_size,
    }


def _latest_color_images_from_dir(task_id: int, episode_dir: Path) -> list[dict[str, Any]]:
    color_dir = episode_dir / "colors"
    if not color_dir.is_dir():
        return []
    latest_by_color: dict[str, Path] = {}
    for image_path in sorted(color_dir.glob("*.jpg")):
        match = re.fullmatch(r"\d+_(color_\d+)\.jpg", image_path.name)
        if not match:
            continue
        latest_by_color[match.group(1)] = image_path
    images = []
    for name in sorted(latest_by_color, key=lambda value: int(value.split("_")[-1])):
        item = _image_item(task_id, episode_dir, name, f"colors/{latest_by_color[name].name}")
        if item:
            images.append(item)
    return images


def _summarize_episode(task_id: int, task_dir: Path, episode_dir: Path) -> dict[str, Any]:
    data_file = episode_dir / "data.json"
    color_dir = episode_dir / "colors"
    depth_dir = episode_dir / "depths"
    audio_dir = episode_dir / "audios"
    color_files = sorted(color_dir.glob("*.jpg")) if color_dir.is_dir() else []
    depth_files = sorted(depth_dir.glob("*.jpg")) if depth_dir.is_dir() else []
    audio_files = sorted(audio_dir.glob("*.npy")) if audio_dir.is_dir() else []
    summary: dict[str, Any] = {
        "name": episode_dir.name,
        "completed": _is_completed_data_json(data_file),
        "frame_count": 0,
        "first_idx": None,
        "last_idx": None,
        "file_count": sum(1 for path in episode_dir.rglob("*") if path.is_file()),
        "color_count": len(color_files),
        "depth_count": len(depth_files),
        "audio_count": len(audio_files),
        "size_bytes": sum(path.stat().st_size for path in [data_file, *color_files, *depth_files, *audio_files] if path.exists()),
        "preview_images": [],
        "state_keys": [],
        "action_keys": [],
        "left_ee_type": None,
        "right_ee_type": None,
        "error": None,
    }
    try:
        payload = json.loads(data_file.read_text(encoding="utf-8"))
        frames = payload.get("data") if isinstance(payload, dict) else []
        if isinstance(frames, list):
            summary["frame_count"] = len(frames)
            if frames:
                first = frames[0] if isinstance(frames[0], dict) else {}
                last = frames[-1] if isinstance(frames[-1], dict) else {}
                summary["first_idx"] = first.get("idx")
                summary["last_idx"] = last.get("idx")
                sample = last if last else first
                states = sample.get("states") if isinstance(sample, dict) else {}
                actions = sample.get("actions") if isinstance(sample, dict) else {}
                if isinstance(states, dict):
                    summary["state_keys"] = sorted(states.keys())
                    left_ee = states.get("left_ee")
                    right_ee = states.get("right_ee")
                    if isinstance(left_ee, dict):
                        summary["left_ee_type"] = left_ee.get("type")
                    if isinstance(right_ee, dict):
                        summary["right_ee_type"] = right_ee.get("type")
                if isinstance(actions, dict):
                    summary["action_keys"] = sorted(actions.keys())
                colors = sample.get("colors") if isinstance(sample, dict) else {}
                if isinstance(colors, dict):
                    for key, rel_path in sorted(colors.items()):
                        if not isinstance(rel_path, str):
                            continue
                        item = _image_item(task_id, episode_dir, key, rel_path)
                        if item:
                            summary["preview_images"].append(item)
        else:
            summary["error"] = "data.json 中 data 字段不是数组"
    except FileNotFoundError:
        summary["error"] = "缺少 data.json"
    except json.JSONDecodeError:
        summary["error"] = "data.json 尚未写完或格式不完整"
    except OSError as exc:
        summary["error"] = str(exc)
    if not summary["preview_images"]:
        summary["preview_images"] = _latest_color_images_from_dir(task_id, episode_dir)
    return summary


def _summarize_episode_row(task_id: int, task_dir: Path, episode_dir: Path) -> dict[str, Any]:
    data_file = episode_dir / "data.json"
    frame_count = 0
    first_idx = None
    last_idx = None
    error = None
    try:
        payload = json.loads(data_file.read_text(encoding="utf-8"))
        frames = payload.get("data") if isinstance(payload, dict) else []
        if isinstance(frames, list):
            frame_count = len(frames)
            if frames:
                first = frames[0] if isinstance(frames[0], dict) else {}
                last = frames[-1] if isinstance(frames[-1], dict) else {}
                first_idx = first.get("idx")
                last_idx = last.get("idx")
        else:
            error = "data.json 中 data 字段不是数组"
    except FileNotFoundError:
        error = "缺少 data.json"
    except json.JSONDecodeError:
        error = "data.json 尚未写完或格式不完整"
    except OSError as exc:
        error = str(exc)

    file_count = 0
    size_bytes = 0
    try:
        for path in episode_dir.rglob("*"):
            if path.is_file():
                file_count += 1
                size_bytes += path.stat().st_size
    except OSError as exc:
        error = error or str(exc)

    return {
        "task_id": task_id,
        "name": episode_dir.name,
        "completed": _is_completed_data_json(data_file),
        "frame_count": frame_count,
        "first_idx": first_idx,
        "last_idx": last_idx,
        "file_count": file_count,
        "size_bytes": size_bytes,
        "error": error,
    }


def _summarize_episode_frame_page(task_id: int, episode_dir: Path, page: Any = 1, page_size: Any = 3) -> dict[str, Any]:
    data_file = episode_dir / "data.json"
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 1
    try:
        page_size = int(page_size)
    except (TypeError, ValueError):
        page_size = 3
    page_size = min(max(page_size, 1), 12)
    payload = json.loads(data_file.read_text(encoding="utf-8"))
    frames = payload.get("data") if isinstance(payload, dict) else []
    if not isinstance(frames, list):
        raise ValidationError("data.json 中 data 字段不是数组")
    total = len(frames)
    total_pages = max(1, math.ceil(total / page_size))
    page = min(max(page, 1), total_pages)
    start = (page - 1) * page_size
    items = []
    for offset, frame in enumerate(frames[start : start + page_size], start=start + 1):
        frame_data = frame if isinstance(frame, dict) else {}
        colors = frame_data.get("colors") if isinstance(frame_data, dict) else {}
        images = []
        if isinstance(colors, dict):
            for key, rel_path in sorted(colors.items()):
                if not isinstance(rel_path, str):
                    continue
                item = _image_item(task_id, episode_dir, key, rel_path)
                if item:
                    images.append(item)
        items.append({"number": offset, "idx": frame_data.get("idx"), "images": images})
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "frames": items,
    }


class IpcBridge:
    """Send page actions to the teleop process and retain its latest heartbeat."""

    COMMANDS = {
        "start": "CMD_START",
        "record": "CMD_RECORD_TOGGLE",
        "hand_eye": "CMD_HAND_EYE_TOGGLE",
        "stop": "CMD_STOP",
    }

    def __init__(self) -> None:
        self._context = None
        self._subscriber = None
        self._requester = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._latest_state: dict[str, Any] = {}
        self._heartbeat_at = 0.0

    def start(self) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("缺少 pyzmq，无法连接遥操 IPC；请安装 teleimager 项目依赖") from exc
        self._context = zmq.Context()
        self._subscriber = self._context.socket(zmq.SUB)
        self._subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self._subscriber.setsockopt(zmq.RCVTIMEO, 200)
        self._subscriber.connect("ipc://@xr_teleoperate_hb.ipc")
        self._new_requester()
        self._running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def _new_requester(self) -> None:
        import zmq
        if self._requester is not None:
            self._requester.close(linger=0)
        self._requester = self._context.socket(zmq.REQ)
        self._requester.setsockopt(zmq.LINGER, 0)
        self._requester.connect("ipc://@xr_teleoperate_data.ipc")

    def _heartbeat_loop(self) -> None:
        import zmq
        while self._running:
            try:
                state = self._subscriber.recv_json()
                with self._state_lock:
                    self._latest_state = state
                    self._heartbeat_at = time.monotonic()
            except zmq.Again:
                continue
            except Exception:
                if self._running:
                    time.sleep(0.2)

    def state(self) -> dict[str, Any]:
        with self._state_lock:
            age = time.monotonic() - self._heartbeat_at if self._heartbeat_at else None
            return {
                "online": age is not None and age < 1.5,
                "heartbeat_age": round(age, 2) if age is not None else None,
                **self._latest_state,
            }

    def send(self, action: str) -> dict[str, Any]:
        import uuid
        import zmq
        if action not in self.COMMANDS:
            raise ValidationError("未知控制操作")
        with self._lock:
            request = {"reqid": str(uuid.uuid4()), "cmd": self.COMMANDS[action]}
            self._requester.send_json(request)
            if not self._requester.poll(1500, zmq.POLLIN):
                self._new_requester()
                raise RuntimeError("遥操进程暂未响应，请等待初始化完成后重试")
            reply = self._requester.recv_json()
        if reply.get("status") != "ok":
            raise RuntimeError(reply.get("msg", "遥操 IPC 返回错误"))
        return reply

    def send_arm_target(self, target_q: list[float], source: str = "ik_replay") -> dict[str, Any]:
        import uuid
        import zmq
        with self._lock:
            request = {
                "reqid": str(uuid.uuid4()),
                "cmd": "CMD_SET_ARM_TARGET",
                "target_q": target_q,
                "source": source,
            }
            self._requester.send_json(request)
            if not self._requester.poll(1500, zmq.POLLIN):
                self._new_requester()
                raise RuntimeError("遥操进程暂未响应 IK 目标，请确认遥操已启动")
            reply = self._requester.recv_json()
        if reply.get("status") != "ok":
            raise RuntimeError(reply.get("msg", "遥操 IPC 返回错误"))
        return reply

    def close(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        for socket in (self._subscriber, self._requester):
            if socket is not None:
                socket.close(linger=0)
        if self._context is not None:
            self._context.term()


class TeleopManager:
    def __init__(
        self,
        dataset_root: Path,
        config_file: Path,
        task_file: Path | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.config_file = config_file
        self.task_file = task_file or dataset_root / "tasks.json"
        self.postprocess_state_file = self.task_file.with_name("postprocess_state.json")
        self.postprocess_file = self.task_file.with_name("postprocess_jobs.json")
        self.data_dir = self._data_dir_from_dataset_root(self.dataset_root)
        self.lerobot_home = self.data_dir / "datasets" / "lerobot"
        self.openpi_work_dir = self.data_dir / "datasets" / "openpi"
        self.openpi_assets_dir = self.openpi_work_dir / "assets"
        self.cache_root = self.data_dir / "datasets" / "cache"
        self.delivery_templates_file = DEFAULT_DELIVERY_TEMPLATES_FILE
        self.legacy_delivery_templates_file = training_data_root(self.data_dir) / "delivery_templates.json"
        self.oss_root = os.environ.get("XR_TELEOP_OSS_ROOT", "oss://bwton-idc/openpi").rstrip("/")
        self.model_download_dir = Path(os.environ.get("XR_TELEOP_MODEL_DOWNLOAD_DIR", self.data_dir / "models" / "openpi_downloads")).expanduser()
        self.training_prep = TrainingPrepManager(
            self.data_dir,
            dataset_root=self.dataset_root,
            lerobot_home=self.lerobot_home,
            openpi_assets_dir=self.openpi_assets_dir,
        )
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        self.logger = DailyFileLogger(self.log_dir)
        self.process: subprocess.Popen[str] | None = None
        self.ipc: IpcBridge | None = None
        self.command: list[str] = []
        self.task: dict[str, Any] | None = None
        self.started_at: float | None = None
        self.logs: deque[str] = deque(maxlen=300)
        self.postprocess_jobs: dict[str, dict[str, Any]] = {}
        self.postprocess_processes: dict[str, subprocess.Popen[str]] = {}
        self.archive_jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._log_thread: threading.Thread | None = None
        self._camera_cache: dict[str, Any] | None = None
        self._camera_cache_key: tuple[str, str] | None = None
        self._camera_cache_at = 0.0
        self._episode_progress_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._dataset_shape_cache: dict[str, tuple[float, tuple[int, int] | None]] = {}
        self.logger.write(
            "info",
            "teleop web manager initialized",
            dataset_root=str(self.dataset_root),
            config_file=str(self.config_file),
            task_file=str(self.task_file),
            postprocess_state_file=str(self.postprocess_state_file),
            log_dir=str(self.log_dir),
        )
        self._apply_saved_data_dir()

    @staticmethod
    def _data_dir_from_dataset_root(dataset_root: Path) -> Path:
        parts = dataset_root.parts
        if len(parts) >= 2 and parts[-2:] == ("datasets", "robot"):
            return Path(*parts[:-2])
        return DEFAULT_DATA_DIR

    def _apply_data_dir(self, data_dir: str | Path) -> None:
        root = Path(data_dir).expanduser()
        self.data_dir = root
        self.dataset_root = root / "datasets" / "robot"
        self.task_file = self.dataset_root / "tasks.json"
        self.postprocess_state_file = self.task_file.with_name("postprocess_state.json")
        self.postprocess_file = self.task_file.with_name("postprocess_jobs.json")
        self.lerobot_home = root / "datasets" / "lerobot"
        self.openpi_work_dir = root / "datasets" / "openpi"
        self.openpi_assets_dir = self.openpi_work_dir / "assets"
        self.cache_root = root / "datasets" / "cache"
        self.delivery_templates_file = DEFAULT_DELIVERY_TEMPLATES_FILE
        self.legacy_delivery_templates_file = training_data_root(root) / "delivery_templates.json"
        self.model_download_dir = Path(os.environ.get("XR_TELEOP_MODEL_DOWNLOAD_DIR", root / "models" / "openpi_downloads")).expanduser()
        if hasattr(self, "_episode_progress_cache"):
            self._episode_progress_cache.clear()
        if hasattr(self, "_dataset_shape_cache"):
            self._dataset_shape_cache.clear()
        if hasattr(self, "training_prep"):
            self.training_prep.update_paths(
                root,
                dataset_root=self.dataset_root,
                lerobot_home=self.lerobot_home,
                openpi_assets_dir=self.openpi_assets_dir,
            )

    def _apply_saved_data_dir(self) -> None:
        try:
            payload = json.loads(self.config_file.read_text(encoding="utf-8"))
            data_dir = (((payload.get("device") or {}).get("config") or {}).get("data_dir"))
        except (OSError, AttributeError, json.JSONDecodeError):
            data_dir = None
        if data_dir:
            self._apply_data_dir(data_dir)

    def _lerobot_action_dim(self, repo_id: str) -> int:
        metadata_path = self.lerobot_home / repo_id / "metadata.json"
        info_path = self.lerobot_home / repo_id / "meta" / "info.json"
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            action_dim = int(payload.get("action_dim"))
            if action_dim > 0:
                return action_dim
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        try:
            payload = json.loads(info_path.read_text(encoding="utf-8"))
            features = payload.get("features") or {}
            for key in ("actions", "action"):
                shape = ((features.get(key) or {}).get("shape") or [])
                if isinstance(shape, list) and shape:
                    action_dim = int(shape[-1])
                    if action_dim > 0:
                        return action_dim
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        return 14

    def _lerobot_dataset_dir(self, repo_id: str) -> Path:
        return self.lerobot_home / repo_id

    def _latest_package_archive(self, task_name: str, marker: str) -> str | None:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name).strip("_") or "dataset"
        package_dir = self.lerobot_home / "packages" / safe_name
        if not package_dir.is_dir():
            return None
        try:
            candidates = [
                path
                for path in package_dir.glob("*.tar.gz")
                if marker in path.name and path.is_file()
            ]
        except OSError:
            return None
        if not candidates:
            return None
        return str(max(candidates, key=lambda path: path.stat().st_mtime))

    def _lerobot_dataset_info(self, repo_id: str) -> dict[str, Any]:
        root = self._lerobot_dataset_dir(repo_id)
        info_path = root / "meta" / "info.json"
        if not root.is_dir() or not info_path.is_file():
            return {"ready": False, "total_episodes": 0, "total_frames": 0}
        try:
            payload = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"ready": False, "total_episodes": 0, "total_frames": 0}
        return {
            "ready": True,
            "total_episodes": int(payload.get("total_episodes") or 0),
            "total_frames": int(payload.get("total_frames") or 0),
        }

    def _lerobot_dataset_ready(self, repo_id: str, task_dir: Path | None = None) -> bool:
        info = self._lerobot_dataset_info(repo_id)
        if not info["ready"]:
            return False
        if task_dir is None:
            return True
        expected = episode_progress(task_dir)["existing_episodes"]
        return bool(expected > 0 and info["total_episodes"] >= expected)

    def _conversion_config_path(self, repo_id: str) -> Path:
        return self._lerobot_dataset_dir(repo_id) / "conversion_config.json"

    def _read_conversion_config(self, repo_id: str) -> dict[str, Any]:
        try:
            payload = json.loads(self._conversion_config_path(repo_id).read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_conversion_config(self, repo_id: str, updates: dict[str, Any]) -> None:
        path = self._conversion_config_path(repo_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except (OSError, json.JSONDecodeError):
            existing = {}
        payload = {**existing, **updates, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        self._atomic_json(path, payload)

    def _finalize_conversion_config(self, record: dict[str, Any], exit_code: int, *, job: dict[str, Any] | None, finished_at: str, error: str | None) -> None:
        repo_id = record.get("postprocess_repo_id") or f"local/{record['name']}"
        task_dir = self.dataset_root / record["name"]
        lerobot_info = self._lerobot_dataset_info(str(repo_id))
        self._write_conversion_config(str(repo_id), {
            "status": "completed" if exit_code == 0 else ("cancelled" if exit_code in {-signal.SIGTERM, -signal.SIGKILL} else "failed"),
            "finished_at": finished_at,
            "exit_code": exit_code,
            "error": error,
            "raw_episode_count": episode_progress(task_dir)["existing_episodes"] if task_dir.is_dir() else None,
            "converted_episode_count": lerobot_info.get("total_episodes"),
            "converted_frame_count": lerobot_info.get("total_frames"),
            "last_job_id": (job or {}).get("id"),
        })

    def _episode_progress_cached(self, task_dir: Path, *, ttl: float = 2.0) -> dict[str, Any]:
        key = str(task_dir)
        now = time.monotonic()
        cached = self._episode_progress_cache.get(key)
        if cached and now - cached[0] < ttl:
            return dict(cached[1])
        progress = episode_progress(task_dir)
        self._episode_progress_cache[key] = (now, dict(progress))
        return progress

    def _dataset_image_shape_cached(self, task_dir: Path, *, ttl: float = 30.0) -> tuple[int, int] | None:
        key = str(task_dir)
        now = time.monotonic()
        cached = self._dataset_shape_cache.get(key)
        if cached and now - cached[0] < ttl:
            return cached[1]
        shape = dataset_image_shape(task_dir)
        self._dataset_shape_cache[key] = (now, shape)
        return shape

    def _norm_stats_candidates(self, repo_id: str, config_name: str | None = None) -> list[Path]:
        names = [name for name in (config_name, DEFAULT_OPENPI_CONFIG_NAME, "pi05_h2_pick_red_cup") if name]
        seen: set[str] = set()
        candidates = [self.openpi_assets_dir / repo_id]
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            candidates.append(self.openpi_assets_dir / name / repo_id)
        return candidates

    def _norm_stats_path(self, repo_id: str, config_name: str | None = None) -> Path | None:
        for path in self._norm_stats_candidates(repo_id, config_name):
            if (path / "norm_stats.json").is_file():
                return path
        return None

    @staticmethod
    def _safe_cache_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "dataset"

    def _postprocess_cache_dir(self, kind: str, repo_id: str) -> Path:
        return self.cache_root / kind / self._safe_cache_name(repo_id)

    def _apply_postprocess_cache_env(self, env: dict[str, str], cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        defaults = {
            "XDG_CACHE_HOME": cache_dir / "xdg",
            "HF_HOME": cache_dir / "huggingface",
            "HF_DATASETS_CACHE": cache_dir / "huggingface" / "datasets",
            "HUGGINGFACE_HUB_CACHE": cache_dir / "huggingface" / "hub",
            "TORCH_HOME": cache_dir / "torch",
        }
        for key, path in defaults.items():
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)

    def delivery_templates(self) -> dict[str, Any]:
        default_payload = json.loads(json.dumps(default_delivery_templates_payload(), ensure_ascii=False))
        path = self.delivery_templates_file
        self._migrate_legacy_delivery_templates(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("templates"), list):
                payload = self._upgrade_delivery_templates(payload)
                payload["path"] = str(path)
                return payload
        except (OSError, json.JSONDecodeError):
            pass
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_json(path, default_payload)
        except OSError:
            default_payload["error"] = f"模板文件无法写入：{path}"
        default_payload["path"] = str(path)
        return default_payload

    def _migrate_legacy_delivery_templates(self, target: Path) -> None:
        legacy = getattr(self, "legacy_delivery_templates_file", None)
        if not isinstance(legacy, Path) or legacy == target or not legacy.is_file():
            return
        try:
            payload = json.loads(legacy.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("templates"), list):
                return
            should_copy = not target.is_file() or legacy.stat().st_mtime > target.stat().st_mtime
            if not should_copy:
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy, target)
            self.logger.write(
                "info",
                "migrated legacy delivery templates to config",
                legacy_path=str(legacy),
                target_path=str(target),
            )
        except (OSError, json.JSONDecodeError):
            return

    def _upgrade_delivery_templates(self, payload: dict[str, Any]) -> dict[str, Any]:
        version = int(payload.get("version") or 1)
        changed = False
        for item in payload.get("templates") or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "")
            if item.get("section") == "training" and ("模型部署" in title or title.startswith("推理服务器：同步模型")):
                item["section"] = "model_deploy"
                changed = True
        if version < 2:
            removable_lines = {
                'mkdir -p "{{LOG_DIR}}"',
                'mkdir -p "{{CHECKPOINT_DIR}}"',
                'mkdir -p "{{HF_HOME}}"',
            }
            for item in payload.get("templates") or []:
                if not isinstance(item, dict) or item.get("id") != "run_training":
                    continue
                lines = str(item.get("body") or "").splitlines()
                updated_lines = [line for line in lines if line.strip() not in removable_lines]
                if updated_lines != lines:
                    item["body"] = "\n".join(updated_lines).strip() + "\n"
                    changed = True
            payload["version"] = 2
            changed = True
        if version < 3:
            replacements = {
                "{{PROXY_PACKAGE_ROOT}}": "/opt/packages/openpi",
                "{{TRAIN_HOST}}": "dgzs-docker-gpu11.prod-2227",
                "{{TRAIN_PACKAGE_DIR}}": "/home/ubuntu/packages/openpi",
                "{{TRAIN_LEROBOT_HOME}}": "/home/ubuntu/datasets/lerobot",
                "{{OPENPI_DIR}}": "/home/ubuntu/openpi",
                "{{OPENPI_DATA_HOME}}": "/home/ubuntu/models/openpi",
                "{{HF_HOME}}": "/home/ubuntu/models/openpi/huggingface",
                "{{ASSETS_DIR}}": "/home/ubuntu/assets",
                "{{CHECKPOINT_DIR}}": "/home/ubuntu/models/openpi/checkpoints",
                "{{LOG_DIR}}": "/home/ubuntu/models/openpi/logs",
                "{{FSDP_DEVICES}}": "2",
                "{{BATCH_SIZE}}": "32",
                "{{NUM_TRAIN_STEPS}}": "50000",
                "{{SAVE_INTERVAL}}": "5000",
                "{{KEEP_PERIOD}}": "25000",
                "{{TARGET_HOST}}": "robot@192.168.61.228",
                "{{TARGET_MODEL_DIR}}": "/data03/data/models/openpi_downloads",
            }
            for item in payload.get("templates") or []:
                if not isinstance(item, dict):
                    continue
                body = str(item.get("body") or "")
                updated = body
                for placeholder, value in replacements.items():
                    updated = updated.replace(placeholder, value)
                if updated != body:
                    item["body"] = updated
                    changed = True
            payload["version"] = 3
            changed = True
        if version < 5:
            for item in payload.get("templates") or []:
                if not isinstance(item, dict):
                    continue
                body = str(item.get("body") or "")
                updated = body
                if item.get("id") == "run_training":
                    updated = updated.replace("    --fsdp-devices 2 \\", "    --fsdp-devices \"{{FSDP_DEVICES}}\" \\")
                    updated = updated.replace("    --fsdp-devices \"2\" \\", "    --fsdp-devices \"{{FSDP_DEVICES}}\" \\")
                    updated = updated.replace("    --batch-size 32 \\", "    --batch-size \"{{BATCH_SIZE}}\" \\")
                    updated = updated.replace("    --batch-size \"32\" \\", "    --batch-size \"{{BATCH_SIZE}}\" \\")
                    updated = updated.replace("    --num-train-steps 50000 \\", "    --num-train-steps \"{{NUM_TRAIN_STEPS}}\" \\")
                    updated = updated.replace("    --num-train-steps \"50000\" \\", "    --num-train-steps \"{{NUM_TRAIN_STEPS}}\" \\")
                    updated = updated.replace("    --save-interval 5000 \\", "    --save-interval \"{{SAVE_INTERVAL}}\" \\")
                    updated = updated.replace("    --save-interval \"5000\" \\", "    --save-interval \"{{SAVE_INTERVAL}}\" \\")
                    updated = updated.replace("    --keep-period 25000 \\", "    --keep-period \"{{KEEP_PERIOD}}\" \\")
                    updated = updated.replace("    --keep-period \"10000\" \\", "    --keep-period \"{{KEEP_PERIOD}}\" \\")
                    updated = updated.replace("    --keep-period \"25000\" \\", "    --keep-period \"{{KEEP_PERIOD}}\" \\")
                    updated = updated.replace(
                        '  > "/home/ubuntu/models/openpi/logs/{{EXP_NAME}}.log" 2>&1',
                        '  2>&1 | tee -a "/home/ubuntu/models/openpi/logs/{{EXP_NAME}}_{{PACKAGE_TIMESTAMP}}.log" 2>&1',
                    )
                    updated = updated.replace(
                        '  2>&1 | tee -a "/home/ubuntu/models/openpi/logs/{{EXP_NAME}}_$(date +%Y%m%d_%H%M%S).log" 2>&1',
                        '  2>&1 | tee -a "/home/ubuntu/models/openpi/logs/{{EXP_NAME}}_{{PACKAGE_TIMESTAMP}}.log" 2>&1',
                    )
                elif item.get("id") == "package_model_to_oss":
                    updated = updated.replace(
                        "OSS_URI='{{MODEL_OSS_URI}}'\n\n",
                        "OSS_URI='{{MODEL_OSS_URI}}'\nMODEL_TRAIN_DIR='{{MODEL_TRAIN_DIR}}'\n\n",
                    )
                    updated = updated.replace(
                        '  -C "/home/ubuntu/models/openpi/checkpoints" "{{EXP_NAME}}"',
                        '  -C "/home/ubuntu/models/openpi/checkpoints/{{CONFIG_NAME}}/{{EXP_NAME}}" "$MODEL_TRAIN_DIR"',
                    )
                    updated = updated.replace(
                        '  -C "/home/ubuntu/models/openpi/checkpoints/{{EXP_NAME}}" "$MODEL_TRAIN_DIR"',
                        '  -C "/home/ubuntu/models/openpi/checkpoints/{{CONFIG_NAME}}/{{EXP_NAME}}" "$MODEL_TRAIN_DIR"',
                    )
                    updated = updated.replace(
                        '  -C "/home/ubuntu/models/openpi/checkpoints/{{CONFIG_NAME}}/{{EXP_NAME}}/" "_CHECKPOINT_METADATA" "assets" "params"',
                        '  -C "/home/ubuntu/models/openpi/checkpoints/{{CONFIG_NAME}}/{{EXP_NAME}}" "$MODEL_TRAIN_DIR"',
                    )
                if updated != body:
                    item["body"] = updated
                    changed = True
            payload["version"] = 5
            changed = True
        if changed:
            payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            try:
                self.delivery_templates_file.parent.mkdir(parents=True, exist_ok=True)
                self._atomic_json(self.delivery_templates_file, payload)
            except OSError:
                payload["error"] = f"模板文件无法写入：{self.delivery_templates_file}"
        return payload

    def save_delivery_templates(self, raw: Any) -> dict[str, Any]:
        payload = raw if isinstance(raw, dict) else {}
        templates = payload.get("templates")
        if not isinstance(templates, list):
            raise ValidationError("模板内容为空或格式不正确")
        normalized: list[dict[str, str]] = []
        allowed_sections = {"data_upload", "training", "model_return", "model_deploy"}
        for index, item in enumerate(templates, start=1):
            if not isinstance(item, dict):
                raise ValidationError(f"第 {index} 个模板格式不正确")
            template_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item.get("id") or f"template_{index}")).strip("_")
            section = str(item.get("section") or "training").strip()
            title = str(item.get("title") or "").strip()
            if section == "training" and "模型部署" in title:
                section = "model_deploy"
            body = str(item.get("body") or "").replace("\r\n", "\n").strip()
            if not title:
                raise ValidationError(f"第 {index} 个模板缺少标题")
            if not body:
                raise ValidationError(f"第 {index} 个模板缺少命令内容")
            if section not in allowed_sections:
                raise ValidationError(f"第 {index} 个模板分组不正确")
            normalized.append({
                "id": template_id or f"template_{index}",
                "section": section,
                "title": title,
                "description": str(item.get("description") or "").strip(),
                "body": body + "\n",
            })
        payload = {
            "version": 5,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "templates": normalized,
        }
        self.delivery_templates_file.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_json(self.delivery_templates_file, payload)
        return {"delivery": self.delivery_templates(), "state": self.state()}

    def reset_delivery_templates(self) -> dict[str, Any]:
        payload = json.loads(json.dumps(default_delivery_templates_payload(), ensure_ascii=False))
        payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.delivery_templates_file.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_json(self.delivery_templates_file, payload)
        return {"delivery": self.delivery_templates(), "state": self.state()}

    @staticmethod
    def _validate_oss_uri(raw: Any, *, default: str) -> str:
        value = str(raw or default).strip().rstrip("/")
        if not value.startswith("oss://") or any(ch in value for ch in "\r\n"):
            raise ValidationError("OSS 路径必须以 oss:// 开头")
        return value

    def _oss_task_uri(self, task_name: str, raw_root: Any = None) -> str:
        root = self._validate_oss_uri(raw_root, default=self.oss_root)
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_name or "")).strip("_")
        if not safe_task:
            raise ValidationError("任务名称为空，无法生成 OSS 目录")
        return f"{root}/{safe_task}/"

    def _local_package_roots(self) -> list[Path]:
        roots = [
            self.lerobot_home / "packages",
            self.training_prep.package_root if hasattr(self, "training_prep") else training_data_root(self.data_dir) / "packages",
        ]
        unique: list[Path] = []
        for root in roots:
            resolved = root.resolve()
            if resolved not in unique:
                unique.append(resolved)
        return unique

    def _validate_local_package_path(self, raw_path: Any) -> Path:
        path = Path(str(raw_path or "")).expanduser().resolve()
        if not path.is_file():
            raise ValidationError(f"本地文件不存在：{path}")
        if path.suffixes[-2:] != [".tar", ".gz"] and path.suffix != ".tgz":
            raise ValidationError("请选择 .tar.gz 或 .tgz 文件")
        try:
            path.relative_to(self.data_dir.resolve())
            return path
        except ValueError:
            pass
        for root in self._local_package_roots():
            try:
                path.relative_to(root)
                return path
            except ValueError:
                continue
        if path.parent == self.lerobot_home.resolve():
            return path
        raise ValidationError("只能选择数据目录下的本地包文件")

    def _validate_local_browser_dir(self, raw_dir: Any = None) -> Path:
        directory = Path(str(raw_dir or self.lerobot_home)).expanduser().resolve()
        try:
            directory.relative_to(self.data_dir.resolve())
        except ValueError as exc:
            raise ValidationError("只能浏览数据目录下的本地目录") from exc
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise ValidationError(f"本地路径不是目录：{directory}")
        return directory

    def oss_local_packages(self, raw: Any = None) -> dict[str, Any]:
        payload = raw if isinstance(raw, dict) else {}
        current_dir = self._validate_local_browser_dir(payload.get("directory"))
        entries: list[dict[str, Any]] = []
        for path in sorted(current_dir.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if path.name.startswith("."):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            is_package = path.is_file() and (path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz")
            if path.is_file() and not is_package:
                continue
            entries.append({
                "name": path.name,
                "path": str(path),
                "is_dir": path.is_dir(),
                "is_package": is_package,
                "size_bytes": 0 if path.is_dir() else stat.st_size,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
            })

        packages: list[dict[str, Any]] = []
        candidates: list[Path] = []
        for root in self._local_package_roots():
            if root.is_dir():
                candidates.extend(path for path in sorted(root.rglob("*")) if path.is_file())
        if self.lerobot_home.is_dir():
            candidates.extend(path for path in sorted(self.lerobot_home.iterdir()) if path.is_file())
        seen: set[Path] = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if path.suffixes[-2:] != [".tar", ".gz"] and path.suffix != ".tgz":
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            packages.append({
                "name": path.name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
            })
        packages.sort(key=lambda item: item.get("mtime") or "", reverse=True)
        parent = current_dir.parent if current_dir != self.data_dir.resolve() else current_dir
        return {
            "packages": packages,
            "local_dir": str(current_dir),
            "local_parent": str(parent),
            "local_entries": entries,
            "oss_root": self.oss_root,
            "model_download_dir": str(self.model_download_dir),
        }

    def oss_list(self, raw: Any) -> dict[str, Any]:
        payload = raw if isinstance(raw, dict) else {}
        uri = self._validate_oss_uri(payload.get("uri"), default=self.oss_root)
        command = ["ossutil", "ls", uri]
        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ValidationError("未找到 ossutil，请先在服务器安装并配置 OSS 访问凭证") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValidationError("OSS 列表请求超时，请检查网络或 OSS 配置") from exc
        lines = [line.rstrip() for line in completed.stdout.splitlines() if line.strip()]
        if completed.returncode != 0:
            raise ValidationError(lines[-1] if lines else f"ossutil ls 失败，退出码 {completed.returncode}")
        entries = []
        for line in lines:
            match = re.search(r"(oss://\S+)", line)
            if not match:
                continue
            item_uri = match.group(1)
            entries.append({
                "uri": item_uri,
                "name": item_uri.rstrip("/").rsplit("/", 1)[-1],
                "is_dir": item_uri.endswith("/"),
                "raw": line,
            })
        return {"uri": uri, "entries": entries, "raw": lines}

    def oss_upload(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValidationError("上传参数格式不正确")
        source = self._validate_local_package_path(raw.get("local_path"))
        task_name = raw.get("task_name") or source.name.split("_lerobot_", 1)[0].split("_openpi_assets_", 1)[0]
        target = self._oss_task_uri(str(task_name), raw.get("oss_root"))
        command = ["ossutil", "cp", "-r", str(source), target]
        return self._start_postprocess_job(
            kind="oss_upload",
            task_id=None,
            task_name=str(task_name),
            command=command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            metadata={
                "repo_id": None,
                "local_path": str(source),
                "oss_uri": target,
                "progress": {
                    "stage": "OSS 上传",
                    "percent": None,
                    "current": None,
                    "total": None,
                    "speed": "",
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            },
        )

    def oss_download(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValidationError("下载参数格式不正确")
        source = self._validate_oss_uri(raw.get("oss_uri"), default="")
        if source.endswith("/"):
            raise ValidationError("请选择具体模型文件，不要选择目录")
        target_dir = Path(str(raw.get("target_dir") or self.model_download_dir)).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        command = ["ossutil", "cp", source, str(target_dir)]
        task_name = source.rstrip("/").rsplit("/", 1)[-1]
        target_file = target_dir / task_name
        if target_file.is_file():
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            return self._record_completed_postprocess_job(
                kind="oss_download",
                task_id=None,
                task_name=task_name,
                command=command,
                metadata={
                    "repo_id": None,
                    "oss_uri": source,
                    "local_path": str(target_file),
                    "progress": {
                        "stage": "本地文件已存在，跳过下载",
                        "percent": 100,
                        "current": None,
                        "total": None,
                        "speed": "",
                        "updated_at": timestamp,
                    },
                },
                logs=[
                    f"local file already exists, skip download: {target_file}",
                    f"oss uri: {source}",
                ],
            )
        return self._start_postprocess_job(
            kind="oss_download",
            task_id=None,
            task_name=task_name,
            command=command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            metadata={
                "repo_id": None,
                "oss_uri": source,
                "local_path": str(target_dir),
                "progress": {
                    "stage": "模型下载",
                    "percent": None,
                    "current": None,
                    "total": None,
                    "speed": "",
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            },
        )

    def _load_store(self) -> dict[str, Any]:
        try:
            saved = json.loads(self.config_file.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise ValueError
        except (OSError, json.JSONDecodeError, ValueError):
            saved = {"device": None}

        task_file_loaded = False
        try:
            task_saved = json.loads(self.task_file.read_text(encoding="utf-8"))
            if not isinstance(task_saved, dict):
                raise ValueError
            task_file_loaded = True
        except (OSError, json.JSONDecodeError, ValueError):
            task_saved = {
                "tasks": saved.get("tasks", []),
                "next_task_id": saved.get("next_task_id", 1),
            }

        # Migrate the first version of the console, which stored one device only.
        if isinstance(saved.get("device"), dict) and "config" not in saved["device"]:
            device = validate_device(saved["device"])
            saved = {
                "device": {
                    "id": uuid.uuid4().hex[:8].upper(),
                    "name": f"{device['arm']}_default",
                    "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "config": device,
                },
            }
            self._save_store({**saved, **task_saved})
        # Migrate the short-lived multi-device version to the local single-device model.
        if "devices" in saved:
            saved["device"] = saved.get("devices", [None])[0] if saved.get("devices") else None
            saved.pop("devices", None)
            for task in saved.get("tasks", []):
                task.pop("device_id", None)
            self._save_store({**saved, **task_saved})
        if isinstance(saved.get("device"), dict) and isinstance(saved["device"].get("config"), dict):
            config = saved["device"]["config"]
            migrated = False
            old_network_interface = config.get("network_interface")
            if old_network_interface in LEGACY_NETWORK_INTERFACES and old_network_interface != DEFAULT_NETWORK_INTERFACE:
                config["network_interface"] = DEFAULT_NETWORK_INTERFACE
                migrated = True
                self.logger.write(
                    "info",
                    "migrated legacy DDS network interface",
                    old_network_interface=old_network_interface,
                    network_interface=DEFAULT_NETWORK_INTERFACE,
                    config_file=str(self.config_file),
                )
            if config.get("arm") == "H2" and not str(config.get("init_arm_pose_file", "") or "").strip():
                config["init_arm_pose_file"] = DEFAULT_H2_INIT_ARM_POSE_FILE
                migrated = True
                self.logger.write(
                    "info",
                    "migrated H2 init arm pose file",
                    init_arm_pose_file=DEFAULT_H2_INIT_ARM_POSE_FILE,
                    config_file=str(self.config_file),
                )
            if migrated:
                saved["device"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                self._save_store({**saved, **task_saved})
        saved.setdefault("device", None)
        task_saved.setdefault("tasks", [])
        task_saved.setdefault("next_task_id", 1)
        # Old combined files are migrated once, then contain device data only.
        if "tasks" in saved or "next_task_id" in saved:
            legacy_tasks = saved.pop("tasks", [])
            legacy_next_id = saved.pop("next_task_id", None)
            if isinstance(legacy_tasks, list):
                task_saved["tasks"] = self._merge_task_lists(
                    task_saved["tasks"] if task_file_loaded else [],
                    legacy_tasks,
                )
            if legacy_next_id is not None and not task_file_loaded:
                task_saved["next_task_id"] = legacy_next_id
            self._save_store({**saved, **task_saved})
        store = {**saved, **task_saved}
        corrected_next_id = self._next_task_id(store.get("tasks", []), store.get("next_task_id", 1))
        if corrected_next_id != store.get("next_task_id"):
            store["next_task_id"] = corrected_next_id
            self._save_tasks(store)
        if self._merge_task_postprocess_state(store):
            self._save_tasks(store)
        if self._migrate_inline_postprocess_logs(store):
            self._save_tasks(store)
        return store

    @staticmethod
    def _next_task_id(tasks: Any, current: Any = 1) -> int:
        try:
            next_id = int(current)
        except (TypeError, ValueError):
            next_id = 1
        if isinstance(tasks, list):
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                try:
                    next_id = max(next_id, int(task.get("id") or 0) + 1)
                except (TypeError, ValueError):
                    continue
        return max(1, next_id)

    @staticmethod
    def _merge_task_lists(primary: Any, secondary: Any) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()

        def key_for(task: dict[str, Any]) -> str:
            name = str(task.get("name") or "").strip()
            return f"name:{name}" if name else f"id:{task.get('id')}"

        for source in (primary, secondary):
            if not isinstance(source, list):
                continue
            for task in source:
                if not isinstance(task, dict):
                    continue
                key = key_for(task)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(dict(task))
        return merged

    def _save_store(self, store: dict[str, Any]) -> None:
        self._atomic_json(self.config_file, {"device": store.get("device")})
        self._save_tasks(store)

    @staticmethod
    def _task_without_inline_postprocess_logs(task: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(task)
        for key in ("last_convert_record", "last_normalize_record"):
            if isinstance(cleaned.get(key), dict):
                record = dict(cleaned[key])
                record.pop("logs", None)
                cleaned[key] = record
        return cleaned

    @staticmethod
    def _task_postprocess_key(task: dict[str, Any]) -> str:
        return str(task.get("id") or task.get("name") or "")

    @staticmethod
    def _task_postprocess_fields(task: dict[str, Any]) -> dict[str, Any]:
        fields = {key: task[key] for key in POSTPROCESS_TASK_KEYS if key in task}
        for key in ("last_convert_record", "last_normalize_record", "last_package_record"):
            if isinstance(fields.get(key), dict):
                record = dict(fields[key])
                record.pop("logs", None)
                fields[key] = record
        return fields

    def _load_postprocess_state(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.postprocess_state_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("tasks"), dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
        return {"tasks": {}}

    def _save_postprocess_state(self, payload: dict[str, Any]) -> None:
        self._atomic_json(self.postprocess_state_file, {
            "tasks": payload.get("tasks", {}),
        })

    def _merge_task_postprocess_state(self, store: dict[str, Any]) -> bool:
        payload = self._load_postprocess_state()
        states = payload.setdefault("tasks", {})
        changed = False
        for task in store.get("tasks", []):
            if not isinstance(task, dict):
                continue
            key = self._task_postprocess_key(task)
            if not key:
                continue
            existing = states.get(key)
            inline_fields = self._task_postprocess_fields(task)
            combined: dict[str, Any] = {}
            if isinstance(existing, dict):
                combined.update(existing)
            if inline_fields:
                combined.update(inline_fields)
                if states.get(key) != combined:
                    states[key] = combined
                    changed = True
            if combined:
                task.update(combined)
        if changed:
            self._save_postprocess_state(payload)
        return changed

    def _extract_task_postprocess_state(self, tasks: list[Any]) -> None:
        payload = self._load_postprocess_state()
        states = payload.setdefault("tasks", {})
        changed = False
        valid_keys: set[str] = set()
        for task in tasks:
            if not isinstance(task, dict):
                continue
            key = self._task_postprocess_key(task)
            if not key:
                continue
            valid_keys.add(key)
            fields = self._task_postprocess_fields(task)
            if fields or key in states:
                if states.get(key) != fields:
                    states[key] = fields
                    changed = True
        for key in list(states):
            if key not in valid_keys:
                states.pop(key, None)
                changed = True
        if changed:
            self._save_postprocess_state(payload)

    def _task_without_postprocess_state(self, task: dict[str, Any]) -> dict[str, Any]:
        cleaned = self._task_without_inline_postprocess_logs(task)
        for key in POSTPROCESS_TASK_KEYS:
            cleaned.pop(key, None)
        return cleaned

    def _save_tasks(self, store: dict[str, Any]) -> None:
        tasks = store.get("tasks", [])
        self._extract_task_postprocess_state(tasks if isinstance(tasks, list) else [])
        if self.task_file.is_file():
            try:
                backup = self.task_file.with_name(f"{self.task_file.name}.bak")
                shutil.copy2(self.task_file, backup)
            except OSError:
                pass
        self._atomic_json(self.task_file, {
            "tasks": [self._task_without_postprocess_state(task) for task in store.get("tasks", [])],
            "next_task_id": self._next_task_id(store.get("tasks", []), store.get("next_task_id", 1)),
        })

    def _migrate_inline_postprocess_logs(self, store: dict[str, Any]) -> bool:
        migrated = False
        for task in store.get("tasks", []):
            for key in ("last_convert_record", "last_normalize_record"):
                record = task.get(key)
                if not isinstance(record, dict):
                    continue
                logs = record.pop("logs", None)
                if logs is None:
                    continue
                migrated = True
                job_id = record.get("job_id")
                if not job_id or self._postprocess_job_from_store(str(job_id)) is not None:
                    continue
                job = {
                    "id": str(job_id),
                    "kind": record.get("kind"),
                    "task_id": task.get("id"),
                    "task_name": task.get("name"),
                    "repo_id": f"local/{task.get('name')}" if task.get("name") else None,
                    "pid": None,
                    "pgid": None,
                    "running": record.get("status") == "running",
                    "exit_code": record.get("exit_code"),
                    "started_at": record.get("started_at"),
                    "finished_at": record.get("finished_at"),
                    "command": record.get("command"),
                    "logs": deque(list(logs)[-POSTPROCESS_JOB_LOG_LIMIT:], maxlen=POSTPROCESS_JOB_LOG_LIMIT),
                    "progress": record.get("progress") or {},
                    "error": record.get("error"),
                }
                self._save_postprocess_job(job)
        return migrated

    def _load_postprocess_store(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.postprocess_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("jobs"), dict):
                logs_changed = self._migrate_postprocess_store_logs(payload["jobs"])
                compacted_jobs, changed = self._compact_postprocess_jobs(payload["jobs"])
                if changed or logs_changed:
                    payload = {**payload, "jobs": compacted_jobs}
                    try:
                        self._atomic_json(self.postprocess_file, payload)
                    except OSError:
                        pass
                return payload
        except (OSError, json.JSONDecodeError):
            pass
        return {"jobs": {}}

    def _migrate_postprocess_store_logs(self, jobs: dict[str, Any]) -> bool:
        changed = False
        for job_id, job in list(jobs.items()):
            if not isinstance(job, dict):
                continue
            had_logs = "logs" in job
            logs = job.pop("logs", None)
            if logs:
                job.setdefault("id", str(job_id))
                path = self._postprocess_log_path_for_job(job)
                if not Path(path).exists():
                    self._append_lines_to_postprocess_log_file(path, list(logs))
                job["log_path"] = str(path)
                changed = True
            elif had_logs:
                changed = True
        return changed

    @staticmethod
    def _job_sort_time(job: dict[str, Any]) -> str:
        return str(job.get("started_at") or job.get("updated_at") or job.get("finished_at") or "")

    @staticmethod
    def _compact_postprocess_jobs(jobs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        latest: dict[tuple[Any, Any], tuple[str, dict[str, Any]]] = {}
        keep_unkeyed: dict[str, Any] = {}
        for job_id, job in jobs.items():
            if not isinstance(job, dict):
                continue
            normalized_id = str(job.get("id") or job_id)
            key_target = job.get("task_id") or job.get("repo_id") or job.get("task_name")
            kind = job.get("kind")
            if job.get("running"):
                keep_unkeyed[normalized_id] = job
                continue
            if not kind or not key_target:
                keep_unkeyed[normalized_id] = job
                continue
            key = (kind, key_target)
            current = latest.get(key)
            current_job = current[1] if current else {}
            current_time = TeleopManager._job_sort_time(current_job)
            job_time = TeleopManager._job_sort_time(job)
            if current is None or job_time >= current_time:
                latest[key] = (normalized_id, job)
        compacted = {job_id: job for job_id, job in keep_unkeyed.items()}
        compacted.update({job_id: job for job_id, job in latest.values()})
        if len(compacted) > POSTPROCESS_STORE_RECENT_LIMIT:
            ordered = sorted(
                compacted.items(),
                key=lambda item: (bool((item[1] or {}).get("running")), TeleopManager._job_sort_time(item[1] if isinstance(item[1], dict) else {})),
                reverse=True,
            )
            compacted = dict(ordered[:POSTPROCESS_STORE_RECENT_LIMIT])
        return compacted, len(compacted) != len(jobs) or set(compacted) != {str(key) for key in jobs}

    def _save_postprocess_job(self, job: dict[str, Any]) -> None:
        payload = self._load_postprocess_store()
        jobs = payload.setdefault("jobs", {})
        self._migrate_job_logs_to_file(job)
        public = self._postprocess_job_public(job, include_logs=False)
        public["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if public.get("kind") not in {"oss_upload", "oss_download"}:
            for existing_id, existing in list(jobs.items()):
                if existing_id == str(job["id"]) or not isinstance(existing, dict):
                    continue
                same_kind = existing.get("kind") == public.get("kind")
                same_task = (
                    existing.get("task_id") == public.get("task_id")
                    or (existing.get("task_name") and existing.get("task_name") == public.get("task_name"))
                    or (existing.get("repo_id") and existing.get("repo_id") == public.get("repo_id"))
                )
                if same_kind and same_task:
                    jobs.pop(existing_id, None)
        jobs[str(job["id"])] = public
        self._atomic_json(self.postprocess_file, payload)

    @staticmethod
    def _safe_log_name(value: Any, *, fallback: str = "task") -> str:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._")
        return name or fallback

    def _postprocess_log_path_for_job(self, job: dict[str, Any]) -> Path:
        existing = str(job.get("log_path") or "").strip()
        if existing:
            return Path(existing)
        task_name = self._safe_log_name(job.get("task_name") or job.get("repo_id"), fallback="postprocess")
        started_at = str(job.get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        date_part = started_at[:10] if re.match(r"\d{4}-\d{2}-\d{2}", started_at) else time.strftime("%Y-%m-%d")
        kind = self._safe_log_name(job.get("kind"), fallback="job")
        job_id = self._safe_log_name(job.get("id"), fallback=time.strftime("%H%M%S"))
        return self.log_dir / "tasks" / task_name / date_part / f"{kind}_{job_id}.log"

    def _append_lines_to_postprocess_log_file(self, path: Path, lines: Iterable[Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", errors="replace") as handle:
            for line in lines:
                handle.write(str(line).rstrip("\n"))
                handle.write("\n")

    def _append_postprocess_log_header(self, job: dict[str, Any]) -> None:
        path = self._postprocess_log_path_for_job(job)
        job["log_path"] = str(path)
        self._append_lines_to_postprocess_log_file(path, [
            "",
            f"===== {job.get('started_at') or time.strftime('%Y-%m-%dT%H:%M:%S%z')} {job.get('kind')} {job.get('id')} =====",
            f"command: {job.get('command') or ''}",
        ])
        job["_logs_migrated_to_file"] = True

    @staticmethod
    def _read_log_tail(path: Path, limit: int = POSTPROCESS_JOB_LOG_LIMIT) -> list[str]:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                return [line.rstrip("\n") for line in deque(handle, maxlen=limit)]
        except OSError:
            return []

    def _migrate_job_logs_to_file(self, job: dict[str, Any]) -> None:
        logs = job.get("logs")
        if not logs:
            job.setdefault("log_path", str(self._postprocess_log_path_for_job(job)))
            return
        path = self._postprocess_log_path_for_job(job)
        if not job.get("_logs_migrated_to_file"):
            self._append_lines_to_postprocess_log_file(path, list(logs))
            job["_logs_migrated_to_file"] = True
        job["log_path"] = str(path)

    def _postprocess_job_from_store(self, job_id: str | None) -> dict[str, Any] | None:
        if not job_id:
            return None
        job = self._load_postprocess_store().get("jobs", {}).get(str(job_id))
        return job if isinstance(job, dict) else None

    def _postprocess_job_public(self, job: dict[str, Any], *, include_logs: bool = True) -> dict[str, Any]:
        result = {
            "id": job.get("id"),
            "kind": job.get("kind"),
            "task_id": job.get("task_id"),
            "task_name": job.get("task_name"),
            "repo_id": job.get("repo_id"),
            "pid": job.get("pid"),
            "pgid": job.get("pgid"),
            "running": bool(job.get("running")),
            "exit_code": job.get("exit_code"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "command": job.get("command"),
            "progress": job.get("progress") or {},
            "error": job.get("error"),
            "external": bool(job.get("external", False)),
        }
        if job.get("package_path"):
            result["package_path"] = job.get("package_path")
        if job.get("package_size_bytes") is not None:
            result["package_size_bytes"] = job.get("package_size_bytes")
        if job.get("oss_uri"):
            result["oss_uri"] = job.get("oss_uri")
        if job.get("local_path"):
            result["local_path"] = job.get("local_path")
        if job.get("log_path"):
            result["log_path"] = job.get("log_path")
        if include_logs:
            if job.get("log_path"):
                result["logs"] = self._read_log_tail(Path(str(job["log_path"])))
            else:
                logs = job.get("logs") or []
                result["logs"] = list(logs)[-POSTPROCESS_JOB_LOG_LIMIT:] if isinstance(logs, deque) else list(logs)[-POSTPROCESS_JOB_LOG_LIMIT:]
        return result

    def _record_with_postprocess_details(self, record: Any) -> Any:
        if not isinstance(record, dict):
            return record
        enriched = dict(record)
        stored = self._postprocess_job_from_store(enriched.get("job_id"))
        if stored:
            enriched.setdefault("command", stored.get("command"))
            enriched["log_path"] = stored.get("log_path")
            enriched["logs"] = self._read_log_tail(Path(str(stored["log_path"]))) if stored.get("log_path") else stored.get("logs", [])
            enriched["progress"] = stored.get("progress") or enriched.get("progress") or {}
            enriched["error"] = enriched.get("error") or stored.get("error")
        else:
            enriched.setdefault("logs", [])
            enriched.setdefault("progress", {})
        return enriched

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _append_runtime_log(self, line: str) -> None:
        text = line.rstrip()
        self.logs.append(text)
        self.logger.write("teleop", text)

    @staticmethod
    def _clean_progress_line(line: str) -> str:
        return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", str(line or "")).replace("\r", "").strip()

    def _parse_postprocess_progress(self, line: str, *, kind: str) -> dict[str, Any] | None:
        text = self._clean_progress_line(line)
        if not text:
            return None
        if kind in {"oss_upload", "oss_download"}:
            percent_match = OSS_PROGRESS_PERCENT_RE.search(text)
            size_match = OSS_PROGRESS_SIZE_RE.search(text)
            speed_match = OSS_PROGRESS_SPEED_RE.search(text)
            completed = any(marker in text.lower() for marker in ("success", "succeed", "finished", "complete"))
            if percent_match or size_match or completed:
                percent = float(percent_match.group(1)) if percent_match else (100.0 if completed else None)
                current = size_match.group(1).strip() if size_match else None
                total = size_match.group(2).strip() if size_match else None
                return {
                    "stage": "OSS 上传" if kind == "oss_upload" else "模型下载",
                    "percent": min(100, max(0, round(percent))) if percent is not None else None,
                    "current": current,
                    "total": total,
                    "speed": speed_match.group(1).strip() if speed_match else "",
                    "line": text,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
        saved_match = POSTPROCESS_SAVED_EPISODE_RE.search(text)
        if saved_match:
            current = int(saved_match.group(1))
            total = int(saved_match.group(2))
            return {
                "stage": "转换 episode",
                "percent": min(100, round(current / total * 100)) if total else None,
                "current": current,
                "total": total,
                "speed": "",
                "line": text,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        package_match = POSTPROCESS_PACKAGE_RE.search(text)
        if package_match:
            percent = int(package_match.group(1))
            current = int(package_match.group(2))
            total = int(package_match.group(3))
            return {
                "stage": "压缩数据",
                "percent": min(100, max(0, percent)),
                "current": current,
                "total": total,
                "speed": "",
                "line": text,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        percent_match = POSTPROCESS_PROGRESS_RE.search(text)
        ratio_match = POSTPROCESS_RATIO_RE.search(text)
        if not percent_match and not ratio_match:
            if "Package complete:" in text:
                return {
                    "stage": "压缩完成",
                    "percent": 100,
                    "current": None,
                    "total": None,
                    "speed": "",
                    "line": text,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
            if "LeRobot dataset written to:" in text:
                return {
                    "stage": "转换完成",
                    "percent": 100,
                    "current": None,
                    "total": None,
                    "speed": "",
                    "line": text,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
            if "Writing stats to:" in text:
                return {
                    "stage": "归一化完成",
                    "percent": 100,
                    "current": None,
                    "total": None,
                    "speed": "",
                    "line": text,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
        return None

    @staticmethod
    def _oss_logs_indicate_success(logs: Iterable[Any]) -> bool:
        for line in logs:
            text = str(line or "").lower()
            if "succeed:" in text and "ok num:" in text:
                return True
        return False

    @staticmethod
    def _oss_logs_indicate_skipped(logs: Iterable[Any]) -> bool:
        for line in logs:
            text = str(line or "").lower()
            if "skip" in text and ("skip size" in text or "skip " in text):
                return True
        return False
        current = int(ratio_match.group(1)) if ratio_match else None
        total = int(ratio_match.group(2)) if ratio_match else None
        percent = int(percent_match.group(1)) if percent_match else None
        if percent is None and current is not None and total:
            percent = round(current / total * 100)
        stage_match = POSTPROCESS_STAGE_RE.match(text)
        speed_match = POSTPROCESS_SPEED_RE.search(text)
        return {
            "stage": stage_match.group(1).strip() if stage_match else ("计算归一化" if kind == "normalize" else "处理中"),
            "percent": min(100, max(0, percent)) if percent is not None else None,
            "current": current,
            "total": total,
            "speed": speed_match.group(1).strip() if speed_match else "",
            "line": text,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def _append_job_log(self, job_id: str, line: str) -> None:
        text = line.rstrip()
        if "'torchcodec' is not available" in text and "falling back to 'pyav'" in text:
            text = "LeRobot: torchcodec is not installed; using pyav decoder fallback (non-fatal)."
        with self._lock:
            job = self.postprocess_jobs.get(job_id)
            if job is None:
                return
            if not isinstance(job.get("logs"), deque):
                job["logs"] = deque(job.get("logs") or [], maxlen=POSTPROCESS_JOB_LOG_LIMIT)
            job["logs"].append(text)
            log_path = self._postprocess_log_path_for_job(job)
            job["log_path"] = str(log_path)
            self._append_lines_to_postprocess_log_file(log_path, [text])
            job["_logs_migrated_to_file"] = True
            progress = self._parse_postprocess_progress(text, kind=str(job.get("kind") or ""))
            if progress is not None:
                job["progress"] = progress
            self._save_postprocess_job(job)
        self.logger.write("postprocess", text, job_id=job_id)

    def _running_external_postprocess_jobs(self) -> list[dict[str, Any]]:
        """Find postprocess jobs that outlived this web process.

        The web service can be restarted while a conversion or normalization
        subprocess keeps running. Those jobs are not in ``self.postprocess_jobs``
        anymore, but they still affect status and should block new jobs.
        """

        try:
            output = subprocess.check_output(
                ["ps", "-eo", "pid=,ppid=,args="],
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            return []

        jobs = []
        for line in output.splitlines():
            text = line.strip()
            if not text:
                continue
            first, _, remainder = text.partition(" ")
            second, _, command = remainder.strip().partition(" ")
            try:
                pid = int(first)
                ppid = int(second)
            except ValueError:
                continue
            kind = None
            repo_id = None
            task_name = None
            try:
                parts = shlex.split(command)
            except ValueError:
                parts = command.split()

            def argument_after(flag: str) -> str | None:
                if flag not in parts:
                    return None
                index = parts.index(flag)
                return parts[index + 1] if index + 1 < len(parts) else None

            if "convert_h2_to_lerobot.py" in command:
                kind = "convert"
                src = argument_after("--src")
                if src:
                    task_name = Path(src).name
                repo_id = argument_after("--repo-id")
            elif "compute_norm_stats.py" in command:
                kind = "normalize"
                try:
                    environ = Path(f"/proc/{pid}/environ").read_bytes().decode("utf-8", errors="ignore")
                except OSError:
                    environ = ""
                for entry in environ.split("\x00"):
                    if entry.startswith("OPENPI_H2_REPO_ID="):
                        repo_id = entry.split("=", 1)[1]
                        task_name = repo_id.split("/", 1)[-1]
                        break
            elif "package_lerobot_dataset.py" in command:
                kind = "package"
                repo_id = argument_after("--repo-id")
                if repo_id:
                    task_name = repo_id.split("/", 1)[-1]
            if kind is None:
                continue
            jobs.append({
                "id": f"external-{pid}",
                "kind": kind,
                "task_id": None,
                "task_name": task_name,
                "repo_id": repo_id,
                "pid": pid,
                "ppid": ppid,
                "running": True,
                "exit_code": None,
                "started_at": None,
                "finished_at": None,
                "command": command,
                "logs": deque(maxlen=0),
                "external": True,
            })
        matched_pids = {job["pid"] for job in jobs}
        return [job for job in jobs if job.get("ppid") not in matched_pids]

    def _running_postprocess_jobs(self) -> list[dict[str, Any]]:
        jobs = [job for job in self.postprocess_jobs.values() if job.get("running")]
        known_pids = {job.get("pid") for job in jobs}
        jobs.extend(
            job for job in self._running_external_postprocess_jobs()
            if job.get("pid") not in known_pids and job.get("ppid") not in known_pids
        )
        return jobs

    @staticmethod
    def _postprocess_job_matches_task(
        job: dict[str, Any],
        *,
        task_id: int,
        task_name: str,
        repo_id: str,
        task_dir: Path,
    ) -> bool:
        return (
            job.get("task_id") == task_id
            or job.get("task_name") == task_name
            or job.get("repo_id") == repo_id
            or str(task_dir) in str(job.get("command", ""))
        )

    def _wait_for_postprocess_exit(self, jobs: list[dict[str, Any]], timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            alive = False
            for job in jobs:
                pid = job.get("pid")
                pgid = job.get("pgid")
                process = self.postprocess_processes.get(job.get("id"))
                if process is not None:
                    if process.poll() is None:
                        alive = True
                    continue
                if pgid:
                    try:
                        os.killpg(int(pgid), 0)
                        alive = True
                        continue
                    except OSError:
                        pass
                if pid:
                    try:
                        os.kill(int(pid), 0)
                        alive = True
                    except OSError:
                        pass
            if not alive:
                return
            time.sleep(0.1)

    def _terminate_postprocess_job(
        self,
        job: dict[str, Any],
        *,
        process: subprocess.Popen[str] | None = None,
        reason: str = "cancelled",
    ) -> None:
        pid = job.get("pid")
        pgid = job.get("pgid")
        if pgid is None and pid:
            try:
                pgid = os.getpgid(int(pid))
                job["pgid"] = pgid
            except OSError:
                pgid = None

        targets = []
        if pgid:
            targets.append(("pgid", int(pgid)))
        elif pid:
            targets.append(("pid", int(pid)))

        for target_type, target in targets:
            try:
                if target_type == "pgid":
                    os.killpg(target, signal.SIGTERM)
                else:
                    os.kill(target, signal.SIGTERM)
            except ProcessLookupError:
                return
            except OSError as exc:
                self.logger.write(
                    "warning",
                    "failed to terminate dataset postprocess job",
                    job_id=job.get("id"),
                    pid=pid,
                    pgid=pgid,
                    reason=reason,
                    error=str(exc),
                )
                return

        deadline = time.time() + POSTPROCESS_TERMINATE_GRACE_SECONDS
        while time.time() < deadline:
            if process is not None and process.poll() is not None:
                break
            try:
                if pgid:
                    os.killpg(int(pgid), 0)
                elif pid:
                    os.kill(int(pid), 0)
                else:
                    break
            except OSError:
                break
            time.sleep(0.1)

        try:
            if pgid:
                os.killpg(int(pgid), signal.SIGKILL)
            elif pid:
                os.kill(int(pid), signal.SIGKILL)
        except OSError:
            pass

        self.logger.write(
            "warning",
            "terminated dataset postprocess process group",
            job_id=job.get("id"),
            pid=pid,
            pgid=pgid,
            reason=reason,
        )

    def _cancel_postprocess_jobs_for_task(
        self,
        *,
        task_id: int,
        task_name: str,
        repo_id: str,
        task_dir: Path,
        kinds: set[str] | None = None,
    ) -> None:
        matching_jobs = [
            job for job in self._running_postprocess_jobs()
            if self._postprocess_job_matches_task(job, task_id=task_id, task_name=task_name, repo_id=repo_id, task_dir=task_dir)
            and (kinds is None or str(job.get("kind") or "") in kinds)
        ]
        if not matching_jobs:
            return
        for job in matching_jobs:
            pid = job.get("pid")
            process = self.postprocess_processes.get(job.get("id"))
            try:
                self._terminate_postprocess_job(job, process=process, reason="same task restarted")
                job["running"] = False
                job["exit_code"] = -signal.SIGTERM
                job["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                self.logger.write(
                    "warning",
                    "cancelled previous dataset postprocess job",
                    job_id=job.get("id"),
                    kind=job.get("kind"),
                    task_id=task_id,
                    task_name=task_name,
                    pid=pid,
                )
            except OSError as exc:
                self.logger.write(
                    "warning",
                    "failed to cancel previous dataset postprocess job",
                    job_id=job.get("id"),
                    kind=job.get("kind"),
                    task_id=task_id,
                    task_name=task_name,
                    pid=pid,
                    error=str(exc),
                )
        self._wait_for_postprocess_exit(matching_jobs)

    def _ensure_no_running_postprocess_job(self, *, task_id: int, task_name: str, repo_id: str, task_dir: Path) -> None:
        running_jobs = self._running_postprocess_jobs()
        other_jobs = [
            job for job in running_jobs
            if not self._postprocess_job_matches_task(job, task_id=task_id, task_name=task_name, repo_id=repo_id, task_dir=task_dir)
        ]
        if not other_jobs:
            return
        summaries = []
        for job in other_jobs[:3]:
            label = "转换" if job.get("kind") == "convert" else "归一化"
            target = job.get("repo_id") or job.get("task_name") or job.get("id")
            summaries.append(f"{label} {target} PID {job.get('pid')}")
        raise ValidationError("已有数据转换/归一化任务正在运行，请等待完成后再启动新的任务：" + "；".join(summaries))

    def _mark_task_postprocess_started(
        self,
        task_id: int,
        kind: str,
        *,
        job_id: str,
        command: str,
        started_at: str,
    ) -> None:
        store = self._load_store()
        record = next((item for item in store["tasks"] if item["id"] == task_id), None)
        if record is None:
            return
        last_record = {
            "kind": kind,
            "job_id": job_id,
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "exit_code": None,
            "command": command,
            "progress": {},
            "error": None,
        }
        if kind == "convert":
            record["convert_started_at"] = started_at
            record["convert_exit_code"] = None
            record.pop("convert_failed_at", None)
            record["last_convert_record"] = last_record
        elif kind == "normalize":
            record["normalize_started_at"] = started_at
            record["normalize_exit_code"] = None
            record.pop("normalize_failed_at", None)
            record["last_normalize_record"] = last_record
        elif kind == "package":
            record["data_package_started_at"] = started_at
            record["data_package_exit_code"] = None
            record.pop("data_package_failed_at", None)
            record["last_data_package_record"] = last_record
            record["package_started_at"] = started_at
            record["package_exit_code"] = None
            record.pop("package_failed_at", None)
            record["last_package_record"] = last_record
        elif kind == "package_assets":
            record["assets_package_started_at"] = started_at
            record["assets_package_exit_code"] = None
            record.pop("assets_package_failed_at", None)
            record["last_assets_package_record"] = last_record
            record["package_started_at"] = started_at
            record["package_exit_code"] = None
            record.pop("package_failed_at", None)
            record["last_package_record"] = last_record
        self._save_tasks(store)

    @staticmethod
    def _postprocess_options_from_command(kind: str, command: str) -> dict[str, Any]:
        try:
            parts = shlex.split(command)
        except ValueError:
            parts = command.split()

        def argument_after(flag: str) -> str | None:
            if flag not in parts:
                return None
            index = parts.index(flag)
            return parts[index + 1] if index + 1 < len(parts) else None

        updates: dict[str, Any] = {}
        if kind == "convert":
            batch_size = argument_after("--batch-size")
            start_episode = argument_after("--start-episode")
            writer_processes = argument_after("--image-writer-processes")
            writer_threads = argument_after("--image-writer-threads")
            updates = {
                "postprocess_repo_id": argument_after("--repo-id"),
                "postprocess_robot_type": argument_after("--robot-type"),
                "postprocess_camera_map": argument_after("--camera-map"),
                "postprocess_image_size": argument_after("--image-size"),
                "postprocess_image_encoding": argument_after("--image-encoding"),
                "postprocess_jpeg_quality": argument_after("--jpeg-quality"),
                "postprocess_video_backend": argument_after("--video-backend"),
                "postprocess_image_writer_processes": int(writer_processes) if writer_processes and writer_processes.isdigit() else None,
                "postprocess_image_writer_threads": int(writer_threads) if writer_threads and writer_threads.isdigit() else None,
                "postprocess_resume": "--resume" in parts,
                "postprocess_overwrite": "--overwrite" in parts,
                "postprocess_batch_size": int(batch_size) if batch_size and batch_size.isdigit() else None,
                "postprocess_start_episode": int(start_episode) + 1 if start_episode and start_episode.isdigit() else 0,
            }
        elif kind == "normalize":
            max_frames = argument_after("--max-frames")
            updates = {
                "postprocess_config_name": argument_after("--config-name"),
                "postprocess_max_frames": int(max_frames) if max_frames and max_frames.isdigit() else None,
            }
        return {key: value for key, value in updates.items() if value is not None}

    def _mark_task_postprocess_finished(
        self,
        task_id: int,
        kind: str,
        exit_code: int,
        *,
        job: dict[str, Any] | None = None,
    ) -> None:
        store = self._load_store()
        record = next((item for item in store["tasks"] if item["id"] == task_id), None)
        if record is None:
            return
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        logs = []
        progress = {}
        cancelled = exit_code in {-signal.SIGTERM, -signal.SIGKILL}
        error_text = None if exit_code == 0 else ("任务已取消" if cancelled else self._summarize_postprocess_error(logs, exit_code))
        if job is not None:
            raw_logs = job.get("logs") or []
            logs = list(raw_logs)[-80:] if isinstance(raw_logs, deque) else list(raw_logs)[-80:]
            progress = dict(job.get("progress") or {})
            if exit_code == 0:
                progress["percent"] = 100
            job["progress"] = progress
            error_text = None if exit_code == 0 else ("任务已取消" if cancelled else self._summarize_postprocess_error(logs, exit_code))
            job["error"] = error_text
            self._save_postprocess_job(job)
        last_record = {
            "kind": kind,
            "job_id": (job or {}).get("id"),
            "status": "completed" if exit_code == 0 else ("cancelled" if cancelled else "failed"),
            "started_at": (job or {}).get("started_at"),
            "finished_at": now,
            "exit_code": exit_code,
            "command": (job or {}).get("command"),
            "progress": progress,
            "error": error_text,
        }
        if kind == "convert":
            record["convert_exit_code"] = exit_code
            if exit_code == 0:
                record["converted_at"] = now
                record.pop("convert_failed_at", None)
            else:
                record["convert_failed_at"] = now
            record["last_convert_record"] = last_record
            self._finalize_conversion_config(record, exit_code, job=job, finished_at=now, error=error_text)
        elif kind == "normalize":
            record["normalize_exit_code"] = exit_code
            if exit_code == 0:
                record["normalized_at"] = now
                record.pop("normalize_failed_at", None)
            else:
                record["normalize_failed_at"] = now
            record["last_normalize_record"] = last_record
        elif kind == "package":
            record["data_package_exit_code"] = exit_code
            if exit_code == 0:
                record["data_packaged_at"] = now
                record.pop("data_package_failed_at", None)
                package_path = (job or {}).get("package_path")
                if package_path:
                    record["last_data_package"] = package_path
                    record["last_package"] = package_path
            else:
                record["data_package_failed_at"] = now
            record["last_data_package_record"] = last_record
            record["package_exit_code"] = exit_code
            if exit_code == 0:
                record["packaged_at"] = now
                record.pop("package_failed_at", None)
                package_path = (job or {}).get("package_path")
                if package_path:
                    record["last_package"] = package_path
            else:
                record["package_failed_at"] = now
            record["last_package_record"] = last_record
        elif kind == "package_assets":
            record["assets_package_exit_code"] = exit_code
            if exit_code == 0:
                record["assets_packaged_at"] = now
                record.pop("assets_package_failed_at", None)
                package_path = (job or {}).get("package_path")
                if package_path:
                    record["last_assets_package"] = package_path
                    record["last_package"] = package_path
            else:
                record["assets_package_failed_at"] = now
            record["last_assets_package_record"] = last_record
            record["package_exit_code"] = exit_code
            if exit_code == 0:
                record["packaged_at"] = now
                record.pop("package_failed_at", None)
            else:
                record["package_failed_at"] = now
            record["last_package_record"] = last_record
        self._save_tasks(store)

    @staticmethod
    def _summarize_postprocess_error(logs: list[str], exit_code: int | None) -> str:
        important_fragments = (
            "output already exists",
            "re-run with",
            "does not exist",
            "no episodes selected",
            "no completed episodes",
            "larger than source image",
            "is not in --target-image-keys",
        )
        for line in reversed(logs):
            text = str(line).strip()
            if not text:
                continue
            lowered = text.lower()
            if any(fragment in lowered for fragment in important_fragments):
                return text
            if (
                "traceback" in lowered
                or "error" in lowered
                or "exception" in lowered
                or "failed" in lowered
                or "filenotfounderror" in lowered
                or "no space left" in lowered
            ):
                return text
        return f"后处理任务退出码 {exit_code}"

    def _update_task_postprocess_options(self, task_id: int, updates: dict[str, Any]) -> None:
        store = self._load_store()
        record = next((item for item in store["tasks"] if item["id"] == task_id), None)
        if record is None:
            return
        record.update(updates)
        record["postprocess_options_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._save_tasks(store)

    @staticmethod
    def _postprocess_attempt_failed(
        *,
        started_at: str | None,
        completed_at: str | None,
        exit_code: int | None,
        running: bool,
        record: Any = None,
    ) -> bool:
        if running or not started_at:
            return False
        if TeleopManager._postprocess_record_cancelled(record):
            return False
        try:
            started_ts = datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%S%z").timestamp()
        except (TypeError, ValueError):
            started_ts = None
        if started_ts is not None and time.time() - started_ts < POSTPROCESS_INTERRUPTION_GRACE_SECONDS:
            return False
        if exit_code == 0 and completed_at and completed_at >= started_at:
            return False
        if exit_code not in (None, 0):
            return True
        return not completed_at or completed_at < started_at

    @staticmethod
    def _postprocess_record_cancelled(record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        try:
            exit_code = int(record.get("exit_code"))
        except (TypeError, ValueError):
            exit_code = None
        return exit_code in {-signal.SIGTERM, -signal.SIGKILL} or record.get("status") == "cancelled"

    def _record_completed_postprocess_job(
        self,
        *,
        kind: str,
        task_id: int | None,
        task_name: str,
        command: list[str],
        metadata: dict[str, Any] | None = None,
        logs: Iterable[Any] = (),
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:10]
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        job = {
            "id": job_id,
            "kind": kind,
            "task_id": task_id,
            "task_name": task_name,
            "pid": None,
            "pgid": None,
            "running": False,
            "exit_code": 0,
            "started_at": timestamp,
            "finished_at": timestamp,
            "command": display_command(command),
            "log_path": "",
            "logs": deque([str(line) for line in logs], maxlen=POSTPROCESS_JOB_LOG_LIMIT),
            "progress": {},
            "error": None,
        }
        if metadata:
            job.update(metadata)
        job["log_path"] = str(self._postprocess_log_path_for_job(job))
        self._append_postprocess_log_header(job)
        self._append_lines_to_postprocess_log_file(Path(str(job["log_path"])), job.get("logs") or [])
        self._save_postprocess_job(job)
        self.logger.write(
            "info",
            "recorded completed postprocess job",
            job_id=job_id,
            kind=kind,
            task_id=task_id,
            task_name=task_name,
            command=job["command"],
        )
        return self.state()

    def _start_postprocess_job(
        self,
        *,
        kind: str,
        task_id: int | None,
        task_name: str,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        metadata: dict[str, Any] | None = None,
        on_started: Callable[[str, str, str], None] | None = None,
        on_finished: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:10]
        command_text = display_command(command)
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.logger.write(
            "info",
            "starting dataset postprocess job",
            job_id=job_id,
            kind=kind,
            task_id=task_id,
            task_name=task_name,
            cwd=str(cwd),
            command=command_text,
        )
        try:
            env = dict(env)
            env.setdefault("PYTHONUNBUFFERED", "1")
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            failed_job = {
                "id": job_id,
                "kind": kind,
                "task_id": task_id,
                "task_name": task_name,
                "pid": None,
                "running": False,
                "exit_code": 1,
                "started_at": started_at,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "command": command_text,
                "logs": deque([f"启动失败: {exc}"], maxlen=POSTPROCESS_JOB_LOG_LIMIT),
                "progress": {},
                "error": str(exc),
            }
            if metadata:
                failed_job.update(metadata)
            failed_job["log_path"] = str(self._postprocess_log_path_for_job(failed_job))
            self._append_postprocess_log_header(failed_job)
            self._append_lines_to_postprocess_log_file(Path(str(failed_job["log_path"])), failed_job.get("logs") or [])
            self._save_postprocess_job(failed_job)
            if on_started is not None:
                on_started(job_id, command_text, started_at)
            elif task_id is not None:
                self._mark_task_postprocess_started(
                    task_id,
                    kind,
                    job_id=job_id,
                    command=command_text,
                    started_at=started_at,
                )
            if on_finished is not None:
                on_finished(1, failed_job)
            elif task_id is not None:
                self._mark_task_postprocess_finished(task_id, kind, 1, job=failed_job)
            self.logger.write(
                "error",
                "failed to start dataset postprocess job",
                job_id=job_id,
                kind=kind,
                task_id=task_id,
                error=str(exc),
                command=command_text,
            )
            raise
        job = {
            "id": job_id,
            "kind": kind,
            "task_id": task_id,
            "task_name": task_name,
            "pid": process.pid,
            "pgid": process.pid,
            "running": True,
            "exit_code": None,
            "started_at": started_at,
            "finished_at": None,
            "command": command_text,
            "log_path": "",
            "logs": deque(maxlen=POSTPROCESS_JOB_LOG_LIMIT),
            "progress": {},
            "error": None,
        }
        if metadata:
            job.update(metadata)
        job["log_path"] = str(self._postprocess_log_path_for_job(job))
        self._append_postprocess_log_header(job)
        if on_finished is not None:
            job["_on_finished"] = on_finished
        self.postprocess_jobs[job_id] = job
        self.postprocess_processes[job_id] = process
        self._save_postprocess_job(job)
        if on_started is not None:
            on_started(job_id, command_text, str(job["started_at"]))
        elif task_id is not None:
            self._mark_task_postprocess_started(
                task_id,
                kind,
                job_id=job_id,
                command=command_text,
                started_at=str(job["started_at"]),
            )
        thread = threading.Thread(target=self._capture_job_logs, args=(job_id, process), daemon=True)
        thread.start()
        return self.state()

    def _read_postprocess_stdout(self, job_id: str, process: subprocess.Popen[str]) -> None:
        if process.stdout is not None:
            try:
                for line in process.stdout:
                    self._append_job_log(job_id, line)
            except ValueError:
                return

    def _capture_job_logs(self, job_id: str, process: subprocess.Popen[str]) -> None:
        reader = threading.Thread(
            target=self._read_postprocess_stdout,
            args=(job_id, process),
            daemon=True,
        )
        reader.start()
        exit_code = process.wait()
        reader.join(timeout=POSTPROCESS_STDOUT_DRAIN_SECONDS)
        stdout_stuck = reader.is_alive()
        with self._lock:
            job = self.postprocess_jobs.get(job_id)
            if job is not None:
                if stdout_stuck:
                    self._append_job_log(
                        job_id,
                        "Postprocess parent exited but stdout is still open; terminating leftover child processes.",
                    )
                    self._terminate_postprocess_job(
                        job,
                        process=process,
                        reason="parent exited while child processes kept stdout open",
                    )
                    if process.stdout is not None:
                        try:
                            process.stdout.close()
                        except OSError:
                            pass
                    if exit_code == 0:
                        exit_code = -signal.SIGTERM
                job["running"] = False
                job["exit_code"] = exit_code
                job["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                package_path = job.get("package_path")
                if package_path and Path(str(package_path)).is_file():
                    try:
                        job["package_size_bytes"] = Path(str(package_path)).stat().st_size
                    except OSError:
                        pass
                if exit_code == 0 and job.get("kind") in {"oss_upload", "oss_download"}:
                    job_logs = list(job.get("logs") or [])
                    skipped = self._oss_logs_indicate_skipped(job_logs)
                    progress = dict(job.get("progress") or {})
                    if skipped:
                        progress["stage"] = "OSS 文件已存在，跳过上传" if job.get("kind") == "oss_upload" else "本地文件已存在，跳过下载"
                    else:
                        progress["stage"] = "OSS 上传完成" if job.get("kind") == "oss_upload" else "模型下载完成"
                    progress["percent"] = 100
                    progress.setdefault("current", None)
                    progress.setdefault("total", None)
                    progress.setdefault("speed", "")
                    progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    job["progress"] = progress
                on_finished = job.get("_on_finished")
                if callable(on_finished):
                    on_finished(exit_code, job)
                elif job.get("task_id") is not None:
                    self._mark_task_postprocess_finished(
                        int(job["task_id"]),
                        str(job["kind"]),
                        exit_code,
                        job=job,
                    )
                self._save_postprocess_job(job)
            self.postprocess_processes.pop(job_id, None)
        self.logger.write(
            "info" if exit_code == 0 else "error",
            "dataset postprocess job finished",
            job_id=job_id,
            exit_code=exit_code,
        )

    def save_device(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValidationError("设备信息格式不正确")
        name = _nonempty_string(raw.get("name"), "设备名称", 64)
        device = validate_device(raw)
        with self._lock:
            store = self._load_store()
            existing = store.get("device") or {}
            current_data_dir = str(((existing.get("config") or {}).get("data_dir")) or self.data_dir)
            if Path(device["data_dir"]).expanduser() != Path(current_data_dir).expanduser():
                if self.process is not None and self.process.poll() is None:
                    raise ValidationError("遥操任务运行中不能切换数据集目录，请先安全停止当前任务")
                self._apply_data_dir(device["data_dir"])
            store["device"] = {
                "id": existing.get("id") or uuid.uuid4().hex[:8].upper(),
                "name": name,
                "registered_at": existing.get("registered_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "config": device,
            }
            self._save_store(store)
            self.logger.write(
                "info",
                "device configuration saved",
                device_id=store["device"]["id"],
                device_name=name,
                config=device,
                config_file=str(self.config_file),
            )
            return self.state()

    def create_task(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValidationError("任务信息格式不正确")
        task = validate_task(raw)
        try:
            target = int(raw.get("target_episodes", 10))
        except (TypeError, ValueError) as exc:
            raise ValidationError("目标采集条数必须是整数") from exc
        if not 1 <= target <= 10000:
            raise ValidationError("目标采集条数必须在 1 到 10000 之间")
        with self._lock:
            store = self._load_store()
            if not store.get("device"):
                raise ValidationError("请先填写并保存当前设备信息")
            if any(item["name"] == task["name"] for item in store["tasks"]):
                raise ValidationError("英文任务名已存在；请从任务列表继续采集")
            record = {
                "id": store["next_task_id"],
                "name": task["name"],
                "instruction": task["instruction"],
                "description": task["description"],
                "target_episodes": target,
                "completed_episodes": 0,
                "progress_percent": 0,
                "status": "未开始",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            store["next_task_id"] += 1
            store["tasks"].append(record)
            self._save_store(store)
            self.logger.write(
                "info",
                "collection task created",
                task_id=record["id"],
                task_name=record["name"],
                target_episodes=target,
                task_file=str(self.task_file),
            )
            return self.state()

    def _running_archive_job(self) -> dict[str, Any] | None:
        return next((job for job in self.archive_jobs.values() if job.get("running")), None)

    def _archive_job_for_task(self, task_id: int) -> dict[str, Any] | None:
        jobs = [job for job in self.archive_jobs.values() if job.get("task_id") == task_id]
        return max(jobs, key=lambda item: item.get("started_at") or "", default=None)

    def archive_task(self, task_id: Any) -> dict[str, Any]:
        """Create a dated tar.gz archive for one stopped dataset task."""
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                raise ValidationError("请先安全停止当前遥操任务，再归档数据集")
            running_archive = self._running_archive_job()
            if running_archive is not None:
                raise ValidationError(f"已有归档任务正在运行：{running_archive.get('task_name')}，请等待完成后再归档")
            store = self._load_store()
            try:
                numeric_id = int(task_id)
            except (TypeError, ValueError) as exc:
                raise ValidationError("任务 ID 不正确") from exc
            record = next((item for item in store["tasks"] if item["id"] == numeric_id), None)
            if record is None:
                raise ValidationError("任务不存在")
            task_dir = self.dataset_root / record["name"]
            if not task_dir.is_dir():
                raise ValidationError("该任务尚未生成数据集目录")
            if episode_progress(task_dir)["existing_episodes"] <= 0:
                raise ValidationError("该任务还没有已完成的录制数据，无法归档")

            archive_dir = self.dataset_root / "archives"
            archive_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archive_stem = f"{record['name']}_{timestamp}"
            archive_path = archive_dir / f"{archive_stem}.tar.gz"
            duplicate = 2
            while archive_path.exists():
                archive_path = archive_dir / f"{archive_stem}_{duplicate}.tar.gz"
                duplicate += 1
            temporary_base = archive_dir / f".{archive_stem}.{uuid.uuid4().hex}.tmp"
            temporary_archive = Path(f"{temporary_base}.tar.gz")
            started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            job_id = uuid.uuid4().hex[:10]
            job = {
                "id": job_id,
                "task_id": numeric_id,
                "task_name": record["name"],
                "running": True,
                "started_at": started_at,
                "finished_at": None,
                "exit_code": None,
                "archive_path": str(archive_path),
                "temporary_archive": str(temporary_archive),
                "error": None,
            }
            self.archive_jobs[job_id] = job
            record["archive_started_at"] = started_at
            record["archive_finished_at"] = None
            record["archive_failed_at"] = None
            record["archive_error"] = None
            self._save_tasks(store)
            self.logger.write(
                "info",
                "dataset archive job started",
                task_id=numeric_id,
                task_name=record["name"],
                archive_path=str(archive_path),
            )
            thread = threading.Thread(
                target=self._run_archive_job,
                args=(job_id, numeric_id, record["name"], archive_path, temporary_base, temporary_archive),
                daemon=True,
            )
            thread.start()
            result = self.state()
            result["archive_job"] = dict(job)
            return result
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                raise ValidationError("请先安全停止当前遥操任务，再归档数据集")
            store = self._load_store()
            try:
                numeric_id = int(task_id)
            except (TypeError, ValueError) as exc:
                raise ValidationError("任务 ID 不正确") from exc
            record = next((item for item in store["tasks"] if item["id"] == numeric_id), None)
            if record is None:
                raise ValidationError("任务不存在")

            task_dir = self.dataset_root / record["name"]
            if not task_dir.is_dir():
                raise ValidationError("该任务尚未生成数据集目录")
            if episode_progress(task_dir)["existing_episodes"] <= 0:
                raise ValidationError("该任务还没有已完成的录制数据，无法归档")

            archive_dir = self.dataset_root / "archives"
            archive_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archive_stem = f"{record['name']}_{timestamp}"
            archive_path = archive_dir / f"{archive_stem}.tar.gz"
            duplicate = 2
            while archive_path.exists():
                archive_path = archive_dir / f"{archive_stem}_{duplicate}.tar.gz"
                duplicate += 1

            temporary_base = archive_dir / f".{archive_stem}.{uuid.uuid4().hex}.tmp"
            temporary_archive = Path(f"{temporary_base}.tar.gz")
            try:
                created = Path(shutil.make_archive(
                    str(temporary_base),
                    "gztar",
                    root_dir=self.dataset_root,
                    base_dir=record["name"],
                ))
                created.replace(archive_path)
            except Exception:
                temporary_archive.unlink(missing_ok=True)
                raise

            archived_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            record["last_archive"] = str(archive_path)
            record["archived_at"] = archived_at
            self._save_tasks(store)
            result = self.state()
            result["archive"] = {
                "task_id": numeric_id,
                "name": archive_path.name,
                "path": str(archive_path),
                "size_bytes": archive_path.stat().st_size,
                "archived_at": archived_at,
            }
            self.logger.write(
                "info",
                "dataset task archived",
                task_id=numeric_id,
                task_name=record["name"],
                archive_path=str(archive_path),
                size_bytes=archive_path.stat().st_size,
            )
            return result

    def _run_archive_job(
        self,
        job_id: str,
        task_id: int,
        task_name: str,
        archive_path: Path,
        temporary_base: Path,
        temporary_archive: Path,
    ) -> None:
        try:
            created = Path(shutil.make_archive(
                str(temporary_base),
                "gztar",
                root_dir=self.dataset_root,
                base_dir=task_name,
            ))
            created.replace(archive_path)
            size_bytes = archive_path.stat().st_size
            finished_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            with self._lock:
                job = self.archive_jobs.get(job_id)
                if job is not None:
                    job.update({
                        "running": False,
                        "finished_at": finished_at,
                        "exit_code": 0,
                        "size_bytes": size_bytes,
                    })
                store = self._load_store()
                record = next((item for item in store["tasks"] if item["id"] == task_id), None)
                if record is not None:
                    record["last_archive"] = str(archive_path)
                    record["archived_at"] = finished_at
                    record["archive_finished_at"] = finished_at
                    record["archive_failed_at"] = None
                    record["archive_error"] = None
                    self._save_tasks(store)
            self.logger.write(
                "info",
                "dataset archive job finished",
                task_id=task_id,
                task_name=task_name,
                archive_path=str(archive_path),
                size_bytes=size_bytes,
            )
        except Exception as exc:
            temporary_archive.unlink(missing_ok=True)
            failed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            with self._lock:
                job = self.archive_jobs.get(job_id)
                if job is not None:
                    job.update({
                        "running": False,
                        "finished_at": failed_at,
                        "exit_code": 1,
                        "error": str(exc),
                    })
                store = self._load_store()
                record = next((item for item in store["tasks"] if item["id"] == task_id), None)
                if record is not None:
                    record["archive_failed_at"] = failed_at
                    record["archive_error"] = str(exc)
                    self._save_tasks(store)
            self.logger.write(
                "error",
                "dataset archive job failed",
                task_id=task_id,
                task_name=task_name,
                archive_path=str(archive_path),
                error=str(exc),
            )

    def convert_dataset(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValidationError("转换参数格式不正确")
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                raise ValidationError("请先安全停止当前遥操任务，再转换数据集")
            record = self._find_task_record(raw.get("task_id"))
            task_dir = self.dataset_root / record["name"]
            repo_id = validate_repo_id(raw.get("repo_id") or record.get("postprocess_repo_id") or f"local/{record['name']}")
            self._ensure_no_running_postprocess_job(
                task_id=int(record["id"]),
                task_name=record["name"],
                repo_id=repo_id,
                task_dir=task_dir,
            )
            if not task_dir.is_dir():
                raise ValidationError("该任务尚未生成数据集目录")
            if episode_progress(task_dir)["existing_episodes"] <= 0:
                raise ValidationError("该任务还没有已完成 episode，无法转换")
            if not CONVERT_ENTRYPOINT.is_file():
                raise ValidationError(f"转换脚本不存在: {CONVERT_ENTRYPOINT}")
            camera_map = validate_camera_map(raw.get("camera_map") or DEFAULT_CAMERA_MAP)
            robot_type = _nonempty_string(raw.get("robot_type") or "robot", "robot_type", 64)
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", robot_type):
                raise ValidationError("robot_type 仅支持字母、数字、点、下划线和连字符")
            overwrite = bool(raw.get("overwrite", True))
            resume = bool(raw.get("resume", False))
            if resume:
                overwrite = False
            start_episode_raw = str(raw.get("start_episode") or "").strip()
            batch_size_raw = str(raw.get("batch_size") or "").strip()
            start_episode = 0
            script_start_episode = 0
            batch_size: int | None = None
            if start_episode_raw:
                try:
                    start_episode = int(start_episode_raw)
                except (TypeError, ValueError) as exc:
                    raise ValidationError("start_episode 必须是整数") from exc
                if start_episode < 0:
                    raise ValidationError("start_episode 必须大于或等于 0")
                script_start_episode = max(0, start_episode - 1)
            if batch_size_raw:
                try:
                    batch_size = int(batch_size_raw)
                except (TypeError, ValueError) as exc:
                    raise ValidationError("batch_size 必须是整数") from exc
                if batch_size <= 0:
                    raise ValidationError("batch_size 必须大于 0")
            image_size = validate_image_size(raw.get("image_size"))
            image_encoding = validate_image_encoding(raw.get("image_encoding"))
            jpeg_quality = validate_jpeg_quality(raw.get("jpeg_quality"))
            video_backend = validate_video_backend(raw.get("video_backend"))
            default_writer_processes = 2 if image_encoding == "video" else 0
            default_writer_threads = 4 if image_encoding == "video" else 2
            image_writer_processes = validate_optional_int(
                raw.get("image_writer_processes"),
                field_name="image-writer-processes",
                default=default_writer_processes,
                minimum=0,
                maximum=8,
            )
            image_writer_threads = validate_optional_int(
                raw.get("image_writer_threads"),
                field_name="image-writer-threads",
                default=default_writer_threads,
                minimum=1,
                maximum=32,
            )
            validate_image_size_for_dataset(image_size, task_dir)
            validate_camera_map_sources(camera_map, task_dir, start_episode=script_start_episode)
            output_path = self.lerobot_home / repo_id
            if output_path.exists() and not overwrite and not resume:
                raise ValidationError(
                    f"LeRobot 输出已存在：{output_path}。请勾选“覆盖同名 LeRobot 数据集”，"
                    "或选择“断点续转”，或换一个 repo-id。"
                )
            self._cancel_postprocess_jobs_for_task(
                task_id=int(record["id"]),
                task_name=record["name"],
                repo_id=repo_id,
                task_dir=task_dir,
            )
            command = [
                str(DEFAULT_LEROBOT_PYTHON),
                str(CONVERT_ENTRYPOINT),
                "--src",
                str(task_dir),
                "--repo-id",
                repo_id,
                "--robot-type",
                robot_type,
                "--task",
                record.get("instruction") or record["description"],
                "--urdf",
                str(PROJECT_ROOT / "assets" / "h2" / "H2.urdf"),
                "--camera-map",
                camera_map,
                "--image-size",
                image_size,
                "--image-encoding",
                image_encoding,
                "--jpeg-quality",
                str(jpeg_quality),
                "--image-writer-processes",
                str(image_writer_processes),
                "--image-writer-threads",
                str(image_writer_threads),
            ]
            if image_encoding == "video" and video_backend:
                command.extend(["--video-backend", video_backend])
            cache_dir = self._postprocess_cache_dir("convert", repo_id)
            command.extend(["--cache-dir", str(cache_dir)])
            if overwrite:
                command.append("--overwrite")
            if resume:
                command.extend(["--resume", "--resume-overlap", "2"])
            if script_start_episode:
                command.extend(["--start-episode", str(script_start_episode)])
            if batch_size is not None:
                command.extend(["--batch-size", str(batch_size)])
            env = os.environ.copy()
            env["HF_LEROBOT_HOME"] = str(self.lerobot_home)
            self._apply_postprocess_cache_env(env, cache_dir)
            started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._write_conversion_config(repo_id, {
                "schema_version": 1,
                "status": "running",
                "started_at": started_at,
                "finished_at": None,
                "exit_code": None,
                "task_id": int(record["id"]),
                "task_name": record["name"],
                "instruction": record.get("instruction") or record["description"],
                "source_dir": str(task_dir),
                "repo_id": repo_id,
                "lerobot_dir": str(output_path),
                "robot_type": robot_type,
                "camera_map": camera_map,
                "image_size": image_size,
                "image_encoding": image_encoding,
                "jpeg_quality": jpeg_quality,
                "video_backend": video_backend,
                "image_writer_processes": image_writer_processes,
                "image_writer_threads": image_writer_threads,
                "resume": resume,
                "overwrite": overwrite,
                "start_episode": start_episode,
                "script_start_episode": script_start_episode,
                "batch_size": batch_size,
                "cache_dir": str(cache_dir),
                "command": display_command(command),
                "raw_episode_count": episode_progress(task_dir)["existing_episodes"],
                "tool": {
                    "script": str(CONVERT_ENTRYPOINT),
                    "python": str(DEFAULT_LEROBOT_PYTHON),
                    "project_root": str(PROJECT_ROOT),
                },
            })
            return self._start_postprocess_job(
                kind="convert",
                task_id=int(record["id"]),
                task_name=record["name"],
                command=command,
                cwd=PROJECT_ROOT,
                env=env,
            )

    def normalize_dataset(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValidationError("归一化参数格式不正确")
        with self._lock:
            record = self._find_task_record(raw.get("task_id"))
            repo_id = validate_repo_id(raw.get("repo_id") or record.get("postprocess_repo_id") or f"local/{record['name']}")
            task_dir = self.dataset_root / record["name"]
            self._cancel_postprocess_jobs_for_task(
                task_id=int(record["id"]),
                task_name=record["name"],
                repo_id=repo_id,
                task_dir=task_dir,
                kinds={"normalize"},
            )
            self._ensure_no_running_postprocess_job(
                task_id=int(record["id"]),
                task_name=record["name"],
                repo_id=repo_id,
                task_dir=task_dir,
            )
            if not self._lerobot_dataset_ready(repo_id, task_dir):
                raise ValidationError(
                    f"请先转换为 LeRobot：未找到 {self._lerobot_dataset_dir(repo_id) / 'meta' / 'info.json'}"
                )
            config_name = validate_config_name(raw.get("config_name") or DEFAULT_OPENPI_CONFIG_NAME)
            openpi_dir = Path(str(raw.get("openpi_dir") or DEFAULT_OPENPI_DIR)).expanduser()
            if not openpi_dir.is_dir():
                raise ValidationError(f"OpenPI 目录不存在: {openpi_dir}")
            script = openpi_dir / "scripts" / "compute_norm_stats.py"
            if not script.is_file():
                raise ValidationError(f"OpenPI 归一化脚本不存在: {script}")
            openpi_python = resolve_openpi_python(openpi_dir)
            self.openpi_work_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(openpi_python),
                str(script),
                "--config-name",
                config_name,
            ]
            max_frames = raw.get("max_frames")
            max_frames_value: int | None = None
            if str(max_frames or "").strip():
                try:
                    max_frames_value = int(max_frames)
                except (TypeError, ValueError) as exc:
                    raise ValidationError("max_frames 必须是整数") from exc
                if max_frames_value <= 0:
                    raise ValidationError("max_frames 必须大于 0")
                command.extend(["--max-frames", str(max_frames_value)])
            self._update_task_postprocess_options(int(record["id"]), {
                "postprocess_repo_id": repo_id,
                "postprocess_config_name": config_name,
                "postprocess_openpi_dir": str(openpi_dir),
                "postprocess_max_frames": max_frames_value,
            })
            env = os.environ.copy()
            real_action_dim = self._lerobot_action_dim(repo_id)
            action_horizon = os.environ.get("OPENPI_ACTION_HORIZON") or os.environ.get("OPENPI_H2_ACTION_HORIZON") or "16"
            try:
                if int(action_horizon) <= 0:
                    action_horizon = "16"
            except (TypeError, ValueError):
                action_horizon = "16"
            env["HF_LEROBOT_HOME"] = str(self.lerobot_home)
            env.setdefault("HF_HUB_OFFLINE", "1")
            env.setdefault("OPENPI_H2_REPO_ID", repo_id)
            env["OPENPI_ACTION_DIM"] = "32"
            env["OPENPI_REAL_ACTION_DIM"] = str(real_action_dim)
            env["OPENPI_ACTION_HORIZON"] = str(action_horizon)
            env["OPENPI_H2_ACTION_DIM"] = "32"
            env["OPENPI_H2_REAL_ACTION_DIM"] = str(real_action_dim)
            env["OPENPI_H2_ACTION_HORIZON"] = str(action_horizon)
            self._apply_postprocess_cache_env(env, self._postprocess_cache_dir("normalize", repo_id))
            openpi_src = str(openpi_dir / "src")
            env["PYTHONPATH"] = (
                openpi_src
                if not env.get("PYTHONPATH")
                else f"{openpi_src}{os.pathsep}{env['PYTHONPATH']}"
            )
            validate_openpi_runtime(
                openpi_python,
                openpi_dir=openpi_dir,
                cwd=self.openpi_work_dir,
                env=env,
            )
            return self._start_postprocess_job(
                kind="normalize",
                task_id=int(record["id"]),
                task_name=record["name"],
                command=command,
                cwd=self.openpi_work_dir,
                env=env,
            )

    def create_training_set(self, raw: Any) -> dict[str, Any]:
        with self._lock:
            state = self.state()
            training_set = self.training_prep.create_training_set(raw, state.get("tasks", []))
            return {"training_set": training_set, "state": self.state()}

    def update_training_set(self, raw: Any) -> dict[str, Any]:
        with self._lock:
            training_set = self.training_prep.update_training_set(raw)
            return {"training_set": training_set, "state": self.state()}

    def add_tasks_to_training_set(self, raw: Any) -> dict[str, Any]:
        with self._lock:
            state = self.state()
            training_set = self.training_prep.add_tasks_to_training_set(raw, state.get("tasks", []))
            return {"training_set": training_set, "state": self.state()}

    def remove_task_from_training_set(self, raw: Any) -> dict[str, Any]:
        with self._lock:
            state = self.state()
            training_set = self.training_prep.remove_task_from_training_set(raw, state.get("tasks", []))
            return {"training_set": training_set, "state": self.state()}

    def package_lerobot_dataset(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValidationError("压缩参数格式不正确")
        with self._lock:
            record = self._find_task_record(raw.get("task_id"))
            repo_id = validate_repo_id(raw.get("repo_id") or record.get("postprocess_repo_id") or f"local/{record['name']}")
            task_dir = self.dataset_root / record["name"]
            source_dir = self._lerobot_dataset_dir(repo_id)
            info = self._lerobot_dataset_info(repo_id)
            if not source_dir.is_dir() or not info.get("ready"):
                raise ValidationError("请先完成 LeRobot 转换，再压缩数据")
            self._ensure_no_running_postprocess_job(
                task_id=int(record["id"]),
                task_name=record["name"],
                repo_id=repo_id,
                task_dir=task_dir,
            )
            if not PACKAGE_ENTRYPOINT.is_file():
                raise ValidationError(f"压缩脚本不存在：{PACKAGE_ENTRYPOINT}")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", record["name"]).strip("_") or "dataset"
            output_dir = self.lerobot_home / "packages" / safe_name
            output_dir.mkdir(parents=True, exist_ok=True)
            archive_path = output_dir / f"{safe_name}_lerobot_{timestamp}.tar.gz"
            duplicate = 1
            while archive_path.exists():
                archive_path = output_dir / f"{safe_name}_lerobot_{timestamp}_{duplicate}.tar.gz"
                duplicate += 1
            command = [
                sys.executable,
                str(PACKAGE_ENTRYPOINT),
                "--lerobot-home",
                str(self.lerobot_home),
                "--repo-id",
                repo_id,
                "--output",
                str(archive_path),
            ]
            env = os.environ.copy()
            self._apply_postprocess_cache_env(env, self._postprocess_cache_dir("package", repo_id))
            return self._start_postprocess_job(
                kind="package",
                task_id=int(record["id"]),
                task_name=record["name"],
                command=command,
                cwd=PROJECT_ROOT,
                env=env,
                metadata={
                    "repo_id": repo_id,
                    "package_path": str(archive_path),
                },
            )

    def package_openpi_assets(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValidationError("压缩参数格式不正确")
        with self._lock:
            record = self._find_task_record(raw.get("task_id"))
            repo_id = validate_repo_id(raw.get("repo_id") or record.get("postprocess_repo_id") or f"local/{record['name']}")
            config_name = _nonempty_string(
                raw.get("config_name") or record.get("postprocess_config_name") or DEFAULT_OPENPI_CONFIG_NAME,
                "OpenPI config-name",
                128,
            )
            task_dir = self.dataset_root / record["name"]
            source_dir = self._norm_stats_path(repo_id, config_name)
            if source_dir is None or not source_dir.is_dir():
                raise ValidationError("请先完成归一化计算，再压缩归一化值")
            self._ensure_no_running_postprocess_job(
                task_id=int(record["id"]),
                task_name=record["name"],
                repo_id=repo_id,
                task_dir=task_dir,
            )
            if not PACKAGE_ENTRYPOINT.is_file():
                raise ValidationError(f"压缩脚本不存在：{PACKAGE_ENTRYPOINT}")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", record["name"]).strip("_") or "dataset"
            output_dir = self.lerobot_home / "packages" / safe_name
            output_dir.mkdir(parents=True, exist_ok=True)
            archive_path = output_dir / f"{safe_name}_openpi_assets_{timestamp}.tar.gz"
            duplicate = 1
            while archive_path.exists():
                archive_path = output_dir / f"{safe_name}_openpi_assets_{timestamp}_{duplicate}.tar.gz"
                duplicate += 1
            command = [
                sys.executable,
                str(PACKAGE_ENTRYPOINT),
                "--source-dir",
                str(source_dir),
                "--arcname",
                f"{config_name}/{repo_id}",
                "--output",
                str(archive_path),
            ]
            env = os.environ.copy()
            self._apply_postprocess_cache_env(env, self._postprocess_cache_dir("package-assets", repo_id))
            return self._start_postprocess_job(
                kind="package_assets",
                task_id=int(record["id"]),
                task_name=record["name"],
                command=command,
                cwd=PROJECT_ROOT,
                env=env,
                metadata={
                    "repo_id": repo_id,
                    "package_path": str(archive_path),
                    "assets_dir": str(source_dir),
                },
            )

    def normalize_training_set(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValidationError("训练集归一化参数格式不正确")
        with self._lock:
            set_id = str(raw.get("training_set_id") or "").strip()
            training_set = self.training_prep.find_training_set(set_id)
            if training_set is None:
                raise ValidationError("训练集不存在，请刷新后重试")
            tasks = training_set.get("tasks") or []
            if not tasks:
                raise ValidationError("训练集尚未加入任务，请先在数据处理页选择任务并加入训练集")
            compatibility = training_set.get("compatibility") or {}
            if not compatibility.get("compatible"):
                issues = "；".join(compatibility.get("issues") or ["训练集不兼容"])
                raise ValidationError(f"训练集暂不能计算归一化：{issues}")
            running_jobs = self._running_postprocess_jobs()
            same_running = [
                job for job in running_jobs
                if job.get("kind") == "normalize" and job.get("task_name") == training_set.get("name")
            ]
            if same_running:
                raise ValidationError("该训练集归一化正在运行，请等待完成")
            if running_jobs:
                summaries = []
                for job in running_jobs[:3]:
                    label = "转换" if job.get("kind") == "convert" else "归一化"
                    target = job.get("repo_id") or job.get("task_name") or job.get("id")
                    summaries.append(f"{label} {target} PID {job.get('pid')}")
                raise ValidationError("已有数据转换/归一化任务正在运行，请等待完成后再启动新的任务：" + "；".join(summaries))

            config_name = validate_config_name(training_set.get("config_name") or DEFAULT_OPENPI_CONFIG_NAME)
            openpi_dir = Path(str(raw.get("openpi_dir") or DEFAULT_OPENPI_DIR)).expanduser()
            if not openpi_dir.is_dir():
                raise ValidationError(f"OpenPI 目录不存在: {openpi_dir}")
            script = openpi_dir / "scripts" / "compute_norm_stats.py"
            if not script.is_file():
                raise ValidationError(f"OpenPI 归一化脚本不存在: {script}")
            openpi_python = resolve_openpi_python(openpi_dir)
            self.openpi_work_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(openpi_python),
                str(script),
                "--config-name",
                config_name,
            ]
            repo_ids = [str(task.get("repo_id") or "").strip() for task in tasks if str(task.get("repo_id") or "").strip()]
            primary_repo_id = repo_ids[0] if repo_ids else ""
            env = os.environ.copy()
            env["HF_LEROBOT_HOME"] = str(self.lerobot_home)
            env.setdefault("HF_HUB_OFFLINE", "1")
            env["OPENPI_REPO_ID"] = primary_repo_id
            env["OPENPI_REPO_IDS"] = ",".join(repo_ids)
            env["OPENPI_H2_REPO_ID"] = primary_repo_id
            env["OPENPI_H2_REPO_IDS"] = ",".join(repo_ids)
            env["OPENPI_TRAINING_SET_NORM_DIR"] = str(training_set.get("norm_stats_dir") or "")
            env["OPENPI_ACTION_DIM"] = str(training_set.get("action_dim") or 32)
            env["OPENPI_REAL_ACTION_DIM"] = str(training_set.get("real_action_dim") or 14)
            env["OPENPI_ACTION_HORIZON"] = str(training_set.get("action_horizon") or 16)
            env["OPENPI_H2_ACTION_DIM"] = str(training_set.get("action_dim") or 32)
            env["OPENPI_H2_REAL_ACTION_DIM"] = str(training_set.get("real_action_dim") or 14)
            env["OPENPI_H2_ACTION_HORIZON"] = str(training_set.get("action_horizon") or 16)
            self._apply_postprocess_cache_env(env, self._postprocess_cache_dir("normalize", f"training_sets/{training_set.get('name')}"))
            openpi_src = str(openpi_dir / "src")
            env["PYTHONPATH"] = (
                openpi_src
                if not env.get("PYTHONPATH")
                else f"{openpi_src}{os.pathsep}{env['PYTHONPATH']}"
            )
            validate_openpi_runtime(
                openpi_python,
                openpi_dir=openpi_dir,
                cwd=self.openpi_work_dir,
                env=env,
            )

            def mark_started(job_id: str, command_text: str, started_at: str) -> None:
                self.training_prep.mark_normalize_started(
                    set_id,
                    job_id=job_id,
                    command=command_text,
                    started_at=started_at,
                )

            def mark_finished(exit_code: int, job: dict[str, Any]) -> None:
                self.training_prep.mark_normalize_finished(set_id, exit_code, job=job)

            return self._start_postprocess_job(
                kind="normalize",
                task_id=None,
                task_name=str(training_set.get("name") or set_id),
                command=command,
                cwd=self.openpi_work_dir,
                env=env,
                on_started=mark_started,
                on_finished=mark_finished,
            )

    def package_training_set(self, raw: Any) -> dict[str, Any]:
        with self._lock:
            package = self.training_prep.package_training_set(raw)
            return {"package": package, "state": self.state()}

    def _find_task_record(self, task_id: Any) -> dict[str, Any]:
        try:
            numeric_id = int(task_id)
        except (TypeError, ValueError) as exc:
            raise ValidationError("任务 ID 不正确") from exc
        store = self._load_store()
        record = next((item for item in store["tasks"] if item["id"] == numeric_id), None)
        if record is None:
            raise ValidationError("任务不存在")
        return record

    def preview_task(
        self,
        task_id: Any,
        limit: int = 20,
        episode: Any = None,
        frame_page: Any = 1,
        frame_page_size: Any = 3,
    ) -> dict[str, Any]:
        """Return a lightweight dataset preview for one task."""
        with self._lock:
            record = self._find_task_record(task_id)
            try:
                limit = int(limit)
            except (TypeError, ValueError):
                limit = 20
            limit = min(max(limit, 1), 100)
            task_dir = self.dataset_root / record["name"]
            if not task_dir.is_dir():
                return {
                    "task": record,
                    "dataset_dir": str(task_dir),
                    "exists": False,
                    "episodes": [],
                    "progress": episode_progress(task_dir),
                    "error": "该任务尚未生成数据集目录",
                }
            episode_dirs = sorted(
                [
                    child
                    for child in task_dir.iterdir()
                    if child.is_dir() and re.fullmatch(r"episode_\d+", child.name)
                ],
                key=lambda path: path.name,
                reverse=True,
            )
            episode_name = str(episode or "").strip()
            single_episode = False
            if episode_name:
                selected = next((path for path in episode_dirs if path.name == episode_name), None)
                if selected is None:
                    raise ValidationError(f"episode 不存在：{episode_name}")
                episode_dirs = [selected]
                single_episode = True
            episodes = [
                _summarize_episode(record["id"], task_dir, episode_dir)
                for episode_dir in episode_dirs[:limit]
            ]
            frame_page_payload = (
                _summarize_episode_frame_page(record["id"], episode_dirs[0], frame_page, frame_page_size)
                if single_episode and episode_dirs
                else None
            )
            return {
                "task": record,
                "dataset_dir": str(task_dir),
                "exists": True,
                "episode_total": len(episode_dirs),
                "shown_episode_count": len(episodes),
                "single_episode": single_episode,
                "frame_page": frame_page_payload,
                "progress": episode_progress(task_dir),
                "episodes": episodes,
                "error": None,
            }

    def data_list(self, raw: Any) -> dict[str, Any]:
        """Return paged episode rows for the data-list page."""
        payload = raw if isinstance(raw, dict) else {}
        with self._lock:
            store = self._load_store()
            tasks = list(store.get("tasks", []))
            task_id = str(payload.get("task_id") or "").strip()
            selected_task: dict[str, Any] | None = None
            if task_id:
                try:
                    numeric_id = int(task_id)
                except (TypeError, ValueError) as exc:
                    raise ValidationError("任务 ID 不正确") from exc
                selected_task = next((item for item in tasks if item.get("id") == numeric_id), None)
                if selected_task is None:
                    raise ValidationError("任务不存在")
            elif tasks:
                selected_task = tasks[0]

            try:
                page = max(1, int(payload.get("page") or 1))
            except (TypeError, ValueError):
                page = 1
            try:
                page_size = int(payload.get("page_size") or 50)
            except (TypeError, ValueError):
                page_size = 50
            page_size = min(max(page_size, 10), 200)
            rows: list[dict[str, Any]] = []
            total = 0
            dataset_dir = ""
            exists = False
            progress: dict[str, Any] = {}
            if selected_task is not None:
                task_dir = self.dataset_root / selected_task["name"]
                dataset_dir = str(task_dir)
                progress = episode_progress(task_dir)
                exists = task_dir.is_dir()
                if exists:
                    episode_dirs = sorted(
                        [
                            child
                            for child in task_dir.iterdir()
                            if child.is_dir() and re.fullmatch(r"episode_\d+", child.name)
                        ],
                        key=lambda path: path.name,
                    )
                    total = len(episode_dirs)
                    start = (page - 1) * page_size
                    page_dirs = episode_dirs[start : start + page_size]
                    rows = [
                        _summarize_episode_row(int(selected_task["id"]), task_dir, episode_dir)
                        for episode_dir in page_dirs
                    ]

            return {
                "tasks": [
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "description": item.get("description"),
                        "instruction": item.get("instruction"),
                    }
                    for item in tasks
                ],
                "task": selected_task,
                "dataset_dir": dataset_dir,
                "exists": exists,
                "progress": progress,
                "episodes": rows,
                "total": total,
                "page": page,
                "page_size": page_size,
            }

    def dataset_file(self, task_id: Any, episode: str, relative_path: str) -> Path:
        with self._lock:
            record = self._find_task_record(task_id)
            if not re.fullmatch(r"episode_\d+", str(episode or "")):
                raise ValidationError("episode 名称不正确")
            task_dir = self.dataset_root / record["name"]
            episode_dir = task_dir / episode
            if not episode_dir.is_dir():
                raise ValidationError("episode 不存在")
            return _safe_relative_file(episode_dir, relative_path)

    def delete_episode(self, task_id: Any, episode: str) -> dict[str, Any]:
        """Delete one saved episode directory and return refreshed task state."""
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                raise ValidationError("采集任务运行中，不能删除 episode；请先安全停止")
            record = self._find_task_record(task_id)
            if not re.fullmatch(r"episode_\d+", str(episode or "")):
                raise ValidationError("episode 名称不正确")
            task_dir = self.dataset_root / record["name"]
            episode_dir = task_dir / str(episode)
            if not episode_dir.is_dir():
                raise ValidationError("episode 不存在")
            shutil.rmtree(episode_dir)
            deleted = {"task_id": record["id"], "episode": str(episode)}
            progress = episode_progress(task_dir)
        return {
            **self.state(),
            "deleted": deleted,
            "preview": self.preview_task(deleted["task_id"]),
            "progress": progress,
        }

    def start_task(self, task_id: Any) -> dict[str, Any]:
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                raise ValidationError("已有遥操任务正在运行，请先安全停止")
            store = self._load_store()
            try:
                numeric_id = int(task_id)
            except (TypeError, ValueError) as exc:
                raise ValidationError("任务 ID 不正确") from exc
            record = next((item for item in store["tasks"] if item["id"] == numeric_id), None)
            if record is None:
                raise ValidationError("任务不存在")
            device_record = store.get("device")
            if device_record is None:
                raise ValidationError("当前设备信息尚未配置")
            device = validate_device(device_record["config"])
            validate_runtime_network_interface(device["network_interface"])
            task = {
                "name": record["name"],
                "instruction": record.get("instruction") or record["description"],
                "description": record["description"],
            }
            if self.ipc is not None:
                self.ipc.close()
                self.ipc = None
            task_dir = self.dataset_root / task["name"]
            task_dir.mkdir(parents=True, exist_ok=True)
            progress = episode_progress(task_dir)
            metadata = {
                "task_name": task["name"],
                "task_id": record["id"],
                "instruction": task["instruction"],
                "instruction_en": task["instruction"],
                "description_zh": task["description"],
                "dataset_dir": str(task_dir),
                "device": device,
                "created_at": record["created_at"],
                "last_started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            (task_dir / "task.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.command = build_command(device, task, self.dataset_root)
            self.logs.clear()
            command_text = display_command(self.command)
            self.logger.write(
                "info",
                "starting teleop task",
                task_id=record["id"],
                task_name=task["name"],
                device_name=device_record["name"],
                dataset_dir=str(task_dir),
                cwd=str(ENTRYPOINT.parent),
                command=command_text,
            )
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            bridge = IpcBridge()
            bridge.start()
            try:
                self.process = subprocess.Popen(
                    self.command,
                    # The teleop entrypoint still resolves assets such as
                    # ../assets/h2/H2.urdf relative to the process cwd.
                    cwd=str(ENTRYPOINT.parent),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except Exception as exc:
                bridge.close()
                self.logger.write(
                    "error",
                    "failed to start teleop task",
                    task_id=record["id"],
                    task_name=task["name"],
                    error=str(exc),
                    command=command_text,
                )
                raise
            self.logger.write(
                "info",
                "teleop process started",
                task_id=record["id"],
                task_name=task["name"],
                pid=self.process.pid,
                command=command_text,
            )
            self.task = {**record, **progress, "device_name": device_record["name"]}
            self.started_at = time.time()
            self.ipc = bridge
            self._log_thread = threading.Thread(target=self._capture_logs, daemon=True)
            self._log_thread.start()
            return self.state()

    def _capture_logs(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self._append_runtime_log(line)
        self.logger.write(
            "info",
            "teleop process output stream closed",
            pid=process.pid,
            exit_code=process.poll(),
        )

    def control(self, action: str) -> dict[str, Any]:
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                raise ValidationError("当前没有正在运行的遥操任务")
            if self.ipc is None:
                raise RuntimeError("IPC 控制器尚未启动")
            self.ipc.send(action)
            self.logger.write(
                "info",
                "teleop control command sent",
                action=action,
                pid=self.process.pid if self.process is not None else None,
                task_id=self.task.get("id") if self.task else None,
                task_name=self.task.get("name") if self.task else None,
            )
            return self.state()

    def ik_replay_target(self, raw: Any) -> dict[str, Any]:
        target = validate_ik_replay_target(raw)
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                raise ValidationError("当前没有正在运行的遥操任务，不能接收 IK 回放目标")
            if self.ipc is None:
                raise RuntimeError("IPC 控制器尚未启动")
            self.ipc.send_arm_target(target["target_q"], target["source"])
            self.logger.write(
                "info",
                "IK replay arm target forwarded",
                source=target["source"],
                pid=self.process.pid if self.process is not None else None,
                task_id=self.task.get("id") if self.task else None,
                task_name=self.task.get("name") if self.task else None,
            )
            return {"success": True, "teleop": self.ipc.state()}

    def _probe_webrtc_endpoint(self, port: Any) -> tuple[bool, str | None]:
        if not port:
            return False, "未配置 WebRTC 端口"
        try:
            port_num = int(port)
        except (TypeError, ValueError):
            return False, "WebRTC 端口无效"
        if DEFAULT_WEBRTC_SCHEME == "https":
            conn = http.client.HTTPSConnection("127.0.0.1", port_num, timeout=0.8, context=ssl._create_unverified_context())
        else:
            conn = http.client.HTTPConnection("127.0.0.1", port_num, timeout=0.8)
        try:
            conn.request("GET", "/?embed=1")
            response = conn.getresponse()
            response.read(1)
            if 200 <= response.status < 400:
                return True, None
            return False, f"HTTP {response.status}"
        except ConnectionRefusedError:
            return False, "端口未监听"
        except TimeoutError:
            return False, "端口连接超时"
        except http.client.RemoteDisconnected:
            return False, "端口连接成功但未返回 HTTP 数据"
        except OSError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, exc.__class__.__name__
        finally:
            conn.close()

    def camera_preview(self) -> dict[str, Any]:
        store = self._load_store()
        device_record = store.get("device")
        if not device_record:
            return {"cameras": [], "error": "尚未保存设备配置"}
        device = device_record["config"]
        img_server_ip = device["img_server_ip"]
        webrtc_server_ip = device.get("webrtc_server_ip") or DEFAULT_WEBRTC_SERVER_IP
        warning = None
        if webrtc_server_ip.startswith("192.168.123."):
            warning = "当前 WebRTC 服务 IP 仍是 192.168.123.x；浏览器通常无法访问，请改为 192.168.61.x 地址。"
        cache_key = (img_server_ip, webrtc_server_ip)
        now = time.monotonic()
        if self._camera_cache_key == cache_key and self._camera_cache and now - self._camera_cache_at < 3.0:
            return self._camera_cache
        try:
            from teleimager.image_client import ZMQ_Requester
            requester = ZMQ_Requester(img_server_ip, 60000)
            try:
                cam_config = requester.request()
            finally:
                requester.close()
            if not isinstance(cam_config, dict):
                raise RuntimeError("图像服务返回空配置")
        except Exception as exc:
            result = {"cameras": [], "error": f"无法读取图像服务相机配置: {exc}"}
            self._camera_cache = result
            self._camera_cache_key = cache_key
            self._camera_cache_at = now
            return result
        cameras = []
        color_idx = 0
        for name, cfg in cam_config.items():
            if not isinstance(cfg, dict):
                continue
            record_colors = []
            if cfg.get("enable_zmq") and cfg.get("data_format", "jpeg") == "jpeg":
                if name == "head_camera" and cfg.get("binocular", False):
                    record_colors = [f"color_{color_idx}", f"color_{color_idx + 1}"]
                    color_idx += 2
                else:
                    record_colors = [f"color_{color_idx}"]
                    color_idx += 1
            item = {
                "name": name,
                "enable_zmq": bool(cfg.get("enable_zmq")),
                "zmq_port": cfg.get("zmq_port"),
                "enable_webrtc": bool(cfg.get("enable_webrtc")),
                "webrtc_port": cfg.get("webrtc_port"),
                "image_shape": cfg.get("image_shape"),
                "binocular": bool(cfg.get("binocular", False)),
                "record_colors": record_colors,
            }
            if item["enable_webrtc"] and item["webrtc_port"]:
                item["url"] = f"{DEFAULT_WEBRTC_SCHEME}://{webrtc_server_ip}:{item['webrtc_port']}/"
                item["embed_url"] = f"{DEFAULT_WEBRTC_SCHEME}://{webrtc_server_ip}:{item['webrtc_port']}/?autostart=1&embed=1"
                ready, error = self._probe_webrtc_endpoint(item["webrtc_port"])
                item["webrtc_ready"] = ready
                item["webrtc_error"] = error
            cameras.append(item)
        unavailable = [item for item in cameras if item.get("enable_webrtc") and item.get("webrtc_ready") is False]
        if unavailable:
            warning = warning or f"{len(unavailable)}/{len(cameras)} 路 WebRTC 相机不可用"
        result = {"cameras": cameras, "error": None, "warning": warning}
        self._camera_cache = result
        self._camera_cache_key = cache_key
        self._camera_cache_at = now
        return result

    def state(self) -> dict[str, Any]:
        with self._lock:
            running = self.process is not None and self.process.poll() is None
            exit_code = None if self.process is None or running else self.process.returncode
            store = self._load_store()
            current_task_id = self.task.get("id") if self.task else None
            device = store.get("device")
            if device:
                device = {**device, "status": "使用中" if running else "空闲"}
            tasks = []
            task_file_changed = False
            running_postprocess_jobs = self._running_postprocess_jobs()
            running_dataset_jobs = [
                job for job in running_postprocess_jobs
                if job.get("kind") in {"convert", "normalize", "package", "package_assets"}
            ]
            for item in store["tasks"]:
                task_dir = self.dataset_root / item["name"]
                progress = self._episode_progress_cached(task_dir)
                count = progress["existing_episodes"]
                target = item["target_episodes"]
                is_active = running and item["id"] == current_task_id
                status = "采集中" if is_active else ("已完成" if count >= target else ("已暂停" if count else "未开始"))
                percent = min(100, round(count / target * 100))
                repo_id = item.get("postprocess_repo_id") or f"local/{item['name']}"
                config_name = item.get("postprocess_config_name") or DEFAULT_OPENPI_CONFIG_NAME
                conversion_config = self._read_conversion_config(str(repo_id))
                if conversion_config.get("repo_id"):
                    repo_id = str(conversion_config["repo_id"])
                lerobot_dir = self._lerobot_dataset_dir(repo_id)
                lerobot_info = self._lerobot_dataset_info(repo_id)
                lerobot_ready = bool(lerobot_info.get("ready") and count > 0 and int(lerobot_info.get("total_episodes") or 0) >= count)
                stats_path = self._norm_stats_path(repo_id, config_name)
                convert_running = any(
                    job.get("kind") == "convert"
                    and (
                        job.get("task_id") == item["id"]
                        or job.get("task_name") == item["name"]
                        or job.get("repo_id") == repo_id
                        or str(task_dir) in str(job.get("command", ""))
                    )
                    for job in running_postprocess_jobs
                )
                normalize_running = any(
                    job.get("kind") == "normalize"
                    and (
                        job.get("task_id") == item["id"]
                        or job.get("task_name") == item["name"]
                        or job.get("repo_id") == repo_id
                    )
                    for job in running_postprocess_jobs
                )
                data_package_running = any(
                    job.get("kind") == "package"
                    and (
                        job.get("task_id") == item["id"]
                        or job.get("task_name") == item["name"]
                        or job.get("repo_id") == repo_id
                    )
                    for job in running_postprocess_jobs
                )
                assets_package_running = any(
                    job.get("kind") == "package_assets"
                    and (
                        job.get("task_id") == item["id"]
                        or job.get("task_name") == item["name"]
                        or job.get("repo_id") == repo_id
                    )
                    for job in running_postprocess_jobs
                )
                package_running = data_package_running or assets_package_running
                if conversion_config.get("status") == "running" and not convert_running:
                    self._write_conversion_config(str(repo_id), {
                        "status": "completed" if lerobot_ready else "interrupted",
                        "exit_code": 0 if lerobot_ready else None,
                        "error": None if lerobot_ready else "转换进程已不在运行，可能被服务重启、系统重启或手动中断。",
                        "converted_episode_count": lerobot_info.get("total_episodes"),
                        "converted_frame_count": lerobot_info.get("total_frames"),
                    })
                    conversion_config = self._read_conversion_config(str(repo_id))
                for job in running_postprocess_jobs:
                    if job.get("kind") != "convert":
                        continue
                    if not self._postprocess_job_matches_task(
                        job,
                        task_id=int(item["id"]),
                        task_name=str(item["name"]),
                        repo_id=repo_id,
                        task_dir=task_dir,
                    ):
                        continue
                    recovered = self._postprocess_options_from_command("convert", str(job.get("command") or ""))
                    conversion_updates = {}
                    recovered_repo_id = recovered.get("postprocess_repo_id") or repo_id
                    for task_key, config_key in (
                        ("postprocess_repo_id", "repo_id"),
                        ("postprocess_robot_type", "robot_type"),
                        ("postprocess_camera_map", "camera_map"),
                        ("postprocess_image_size", "image_size"),
                        ("postprocess_resume", "resume"),
                        ("postprocess_overwrite", "overwrite"),
                        ("postprocess_start_episode", "start_episode"),
                        ("postprocess_batch_size", "batch_size"),
                    ):
                        if task_key in recovered:
                            conversion_updates[config_key] = recovered[task_key]
                    if conversion_updates:
                        self._write_conversion_config(str(recovered_repo_id), conversion_updates)
                        conversion_config = self._read_conversion_config(str(recovered_repo_id))
                        if conversion_config.get("repo_id"):
                            repo_id = str(conversion_config["repo_id"])
                    break
                last_convert_record = self._record_with_postprocess_details(item.get("last_convert_record"))
                last_normalize_record = self._record_with_postprocess_details(item.get("last_normalize_record"))
                last_package_record = self._record_with_postprocess_details(item.get("last_package_record"))
                legacy_package_record = last_package_record or {}
                last_data_package_record = self._record_with_postprocess_details(item.get("last_data_package_record"))
                if last_data_package_record is None and legacy_package_record.get("kind") == "package":
                    last_data_package_record = legacy_package_record
                last_assets_package_record = self._record_with_postprocess_details(item.get("last_assets_package_record"))
                if last_assets_package_record is None and legacy_package_record.get("kind") == "package_assets":
                    last_assets_package_record = legacy_package_record
                convert_cancelled = self._postprocess_record_cancelled(last_convert_record)
                normalize_cancelled = self._postprocess_record_cancelled(last_normalize_record)
                data_package_cancelled = self._postprocess_record_cancelled(last_data_package_record)
                assets_package_cancelled = self._postprocess_record_cancelled(last_assets_package_record)
                package_cancelled = self._postprocess_record_cancelled(last_package_record)
                if convert_cancelled and not convert_running:
                    convert_running = False
                if normalize_cancelled and not normalize_running:
                    normalize_running = False
                convert_failed = self._postprocess_attempt_failed(
                    started_at=item.get("convert_started_at"),
                    completed_at=item.get("converted_at"),
                    exit_code=item.get("convert_exit_code"),
                    running=convert_running,
                    record=last_convert_record,
                )
                normalize_failed = self._postprocess_attempt_failed(
                    started_at=item.get("normalize_started_at"),
                    completed_at=item.get("normalized_at"),
                    exit_code=item.get("normalize_exit_code"),
                    running=normalize_running,
                    record=last_normalize_record,
                )
                package_failed = self._postprocess_attempt_failed(
                    started_at=item.get("package_started_at"),
                    completed_at=item.get("packaged_at"),
                    exit_code=item.get("package_exit_code"),
                    running=package_running,
                    record=last_package_record,
                )
                data_package_failed = self._postprocess_attempt_failed(
                    started_at=item.get("data_package_started_at"),
                    completed_at=item.get("data_packaged_at"),
                    exit_code=item.get("data_package_exit_code"),
                    running=data_package_running,
                    record=last_data_package_record,
                )
                assets_package_failed = self._postprocess_attempt_failed(
                    started_at=item.get("assets_package_started_at"),
                    completed_at=item.get("assets_packaged_at"),
                    exit_code=item.get("assets_package_exit_code"),
                    running=assets_package_running,
                    record=last_assets_package_record,
                )
                if convert_cancelled and not convert_running:
                    convert_failed = False
                if normalize_cancelled and not normalize_running:
                    normalize_failed = False
                if package_cancelled and not package_running:
                    package_failed = False
                if data_package_cancelled and not data_package_running:
                    data_package_failed = False
                if assets_package_cancelled and not assets_package_running:
                    assets_package_failed = False
                latest_data_package = (
                    item.get("last_data_package")
                    or (item.get("last_package") if "_lerobot_" in str(item.get("last_package") or "") else None)
                    or self._latest_package_archive(str(item.get("name") or ""), "_lerobot_")
                )
                latest_assets_package = (
                    item.get("last_assets_package")
                    or (item.get("last_package") if "_openpi_assets_" in str(item.get("last_package") or "") else None)
                    or self._latest_package_archive(str(item.get("name") or ""), "_openpi_assets_")
                )
                if convert_failed and not item.get("convert_failed_at"):
                    failed_at = item.get("convert_started_at")
                    item["convert_failed_at"] = failed_at
                    last_record = item.get("last_convert_record") or {
                        "kind": "convert",
                        "started_at": item.get("convert_started_at"),
                        "command": None,
                        "progress": {},
                    }
                    last_record.update({
                        "status": "failed",
                        "finished_at": failed_at,
                        "exit_code": item.get("convert_exit_code"),
                        "error": last_record.get("error") or "转换中断：服务重启、服务器断电或进程退出，未记录完成状态。",
                    })
                    item["last_convert_record"] = last_record
                    task_file_changed = True
                if normalize_failed and not item.get("normalize_failed_at"):
                    failed_at = item.get("normalize_started_at")
                    item["normalize_failed_at"] = failed_at
                    last_record = item.get("last_normalize_record") or {
                        "kind": "normalize",
                        "started_at": item.get("normalize_started_at"),
                        "command": None,
                        "progress": {},
                    }
                    last_record.update({
                        "status": "failed",
                        "finished_at": failed_at,
                        "exit_code": item.get("normalize_exit_code"),
                        "error": last_record.get("error") or "归一化中断：服务重启、服务器断电或进程退出，未记录完成状态。",
                    })
                    item["last_normalize_record"] = last_record
                    task_file_changed = True
                archive_job = self._archive_job_for_task(int(item["id"]))
                archive_running = bool(archive_job and archive_job.get("running"))
                archive_started_at = (archive_job or {}).get("started_at") or item.get("archive_started_at")
                archive_finished_at = (
                    (archive_job or {}).get("finished_at")
                    or item.get("archive_finished_at")
                    or item.get("archived_at")
                )
                archive_interrupted = (
                    bool(archive_started_at)
                    and not archive_running
                    and (not archive_finished_at or str(archive_finished_at) < str(archive_started_at))
                    and not item.get("archive_failed_at")
                )
                if archive_interrupted:
                    item["archive_failed_at"] = archive_started_at
                    item["archive_error"] = "归档中断：服务重启、服务器断电或进程退出，未记录完成状态"
                    task_file_changed = True
                archive_status = {
                    "running": archive_running,
                    "started_at": archive_started_at,
                    "finished_at": archive_finished_at,
                    "failed_at": item.get("archive_failed_at"),
                    "error": (archive_job or {}).get("error") or item.get("archive_error"),
                    "last_archive": item.get("last_archive"),
                    "archive_path": (archive_job or {}).get("archive_path") or item.get("last_archive"),
                    "exit_code": (archive_job or {}).get("exit_code"),
                }
                postprocess_status = {
                    "repo_id": repo_id,
                    "robot_type": conversion_config.get("robot_type") or item.get("postprocess_robot_type") or "robot",
                    "camera_map": conversion_config.get("camera_map") or item.get("postprocess_camera_map") or DEFAULT_CAMERA_MAP,
                    "camera_map_updated_at": item.get("postprocess_options_updated_at"),
                    "image_size": conversion_config.get("image_size") or item.get("postprocess_image_size") or "original",
                    "image_encoding": conversion_config.get("image_encoding") or item.get("postprocess_image_encoding") or "auto",
                    "jpeg_quality": conversion_config.get("jpeg_quality") or item.get("postprocess_jpeg_quality") or 95,
                    "video_backend": conversion_config.get("video_backend") or item.get("postprocess_video_backend") or "",
                    "image_writer_processes": (
                        conversion_config.get("image_writer_processes")
                        if conversion_config.get("image_writer_processes") is not None
                        else item.get("postprocess_image_writer_processes")
                    ),
                    "image_writer_threads": (
                        conversion_config.get("image_writer_threads")
                        if conversion_config.get("image_writer_threads") is not None
                        else item.get("postprocess_image_writer_threads")
                    ),
                    "source_image_shape": list(self._dataset_image_shape_cached(task_dir) or []),
                    "resume": conversion_config.get("resume") if conversion_config.get("resume") is not None else item.get("postprocess_resume", False),
                    "overwrite": conversion_config.get("overwrite") if conversion_config.get("overwrite") is not None else item.get("postprocess_overwrite", True),
                    "start_episode": conversion_config.get("start_episode") if conversion_config.get("start_episode") is not None else item.get("postprocess_start_episode", 0),
                    "batch_size": conversion_config.get("batch_size") if conversion_config.get("batch_size") is not None else item.get("postprocess_batch_size"),
                    "config_name": item.get("postprocess_config_name") or config_name,
                    "openpi_dir": item.get("postprocess_openpi_dir") or str(DEFAULT_OPENPI_DIR),
                    "max_frames": item.get("postprocess_max_frames"),
                    "robot_dir": str(task_dir),
                    "lerobot_dir": str(lerobot_dir),
                    "lerobot_ready": lerobot_ready,
                    "lerobot_episodes": lerobot_info.get("total_episodes", 0),
                    "lerobot_frames": lerobot_info.get("total_frames", 0),
                    "lerobot_expected_episodes": count,
                    "convert_running": convert_running,
                    "convert_cancelled": convert_cancelled,
                    "convert_failed": convert_failed,
                    "convert_started_at": item.get("convert_started_at"),
                    "converted_at": item.get("converted_at"),
                    "convert_failed_at": item.get("convert_failed_at") or (item.get("convert_started_at") if convert_failed else None),
                    "convert_exit_code": item.get("convert_exit_code"),
                    "last_convert_record": last_convert_record,
                    "norm_stats_dir": str(stats_path) if stats_path else str(self.openpi_assets_dir / config_name / repo_id),
                    "norm_stats_ready": stats_path is not None,
                    "normalize_running": normalize_running,
                    "normalize_cancelled": normalize_cancelled,
                    "normalize_failed": normalize_failed,
                    "normalize_started_at": item.get("normalize_started_at"),
                    "normalized_at": item.get("normalized_at"),
                    "normalize_failed_at": item.get("normalize_failed_at") or (item.get("normalize_started_at") if normalize_failed else None),
                    "normalize_exit_code": item.get("normalize_exit_code"),
                    "last_normalize_record": last_normalize_record,
                    "package_running": package_running,
                    "package_cancelled": package_cancelled,
                    "package_failed": package_failed,
                    "package_started_at": item.get("package_started_at"),
                    "packaged_at": item.get("packaged_at"),
                    "package_failed_at": item.get("package_failed_at") or (item.get("package_started_at") if package_failed else None),
                    "package_exit_code": item.get("package_exit_code"),
                    "last_package_record": last_package_record,
                    "last_package": item.get("last_package"),
                    "data_package_running": data_package_running,
                    "data_package_cancelled": data_package_cancelled,
                    "data_package_failed": data_package_failed,
                    "data_package_started_at": item.get("data_package_started_at"),
                    "data_packaged_at": item.get("data_packaged_at"),
                    "data_package_failed_at": item.get("data_package_failed_at") or (item.get("data_package_started_at") if data_package_failed else None),
                    "data_package_exit_code": item.get("data_package_exit_code"),
                    "last_data_package_record": last_data_package_record,
                    "last_data_package": latest_data_package,
                    "assets_package_running": assets_package_running,
                    "assets_package_cancelled": assets_package_cancelled,
                    "assets_package_failed": assets_package_failed,
                    "assets_package_started_at": item.get("assets_package_started_at"),
                    "assets_packaged_at": item.get("assets_packaged_at"),
                    "assets_package_failed_at": item.get("assets_package_failed_at") or (item.get("assets_package_started_at") if assets_package_failed else None),
                    "assets_package_exit_code": item.get("assets_package_exit_code"),
                    "last_assets_package_record": last_assets_package_record,
                    "last_assets_package": latest_assets_package,
                    "postprocess_busy": bool(running_dataset_jobs),
                }
                for key, value in (
                    ("completed_episodes", count),
                    ("progress_percent", percent),
                    ("status", status),
                ):
                    if item.get(key) != value:
                        item[key] = value
                        task_file_changed = True
                tasks.append({
                    **item,
                    **progress,
                    "device_name": device["name"] if device else "未配置",
                    "progress_percent": percent,
                    "status": status,
                    "active": is_active,
                    "postprocess_status": postprocess_status,
                    "archive_status": archive_status,
                })
            if task_file_changed:
                self._save_tasks(store)
            postprocess_jobs = []
            seen_postprocess_job_ids: set[str] = set()

            def add_latest_postprocess_job(record: Any) -> None:
                if not isinstance(record, dict):
                    return
                job_id = str(record.get("job_id") or record.get("id") or "")
                if not job_id or job_id in seen_postprocess_job_ids:
                    return
                live_job = self.postprocess_jobs.get(job_id)
                if live_job is not None:
                    postprocess_jobs.append(self._postprocess_job_public(live_job, include_logs=True))
                    seen_postprocess_job_ids.add(job_id)
                    return
                stored_job = self._postprocess_job_from_store(job_id)
                if stored_job is not None:
                    postprocess_jobs.append(stored_job)
                    seen_postprocess_job_ids.add(job_id)

            for task in tasks:
                status_payload = task.get("postprocess_status") or {}
                add_latest_postprocess_job(status_payload.get("last_convert_record"))
                add_latest_postprocess_job(status_payload.get("last_normalize_record"))
                add_latest_postprocess_job(status_payload.get("last_package_record"))
            for running_job in running_postprocess_jobs:
                job_id = str(running_job.get("id") or "")
                if not job_id or job_id in seen_postprocess_job_ids:
                    continue
                postprocess_jobs.append(self._postprocess_job_public(running_job, include_logs=True))
                seen_postprocess_job_ids.add(job_id)
            transfer_store = self._load_postprocess_store()
            transfer_jobs = transfer_store.get("jobs") if isinstance(transfer_store, dict) else {}
            transfer_store_changed = False
            running_pids = {job.get("pid") for job in running_postprocess_jobs if job.get("pid")}
            running_ids = {str(job.get("id") or "") for job in running_postprocess_jobs}
            if isinstance(transfer_jobs, dict):
                for job_id, stored_job in transfer_jobs.items():
                    if not isinstance(stored_job, dict):
                        continue
                    if stored_job.get("kind") not in {"oss_upload", "oss_download"}:
                        continue
                    normalized_job_id = str(stored_job.get("id") or job_id)
                    if normalized_job_id in seen_postprocess_job_ids:
                        continue
                    stored_logs = (
                        self._read_log_tail(Path(str(stored_job["log_path"])))
                        if stored_job.get("log_path")
                        else list(stored_job.get("logs") or [])
                    )
                    success_from_logs = self._oss_logs_indicate_success(stored_logs)
                    skipped_from_logs = self._oss_logs_indicate_skipped(stored_logs)
                    if success_from_logs and (
                        stored_job.get("exit_code") != 0
                        or stored_job.get("error")
                        or (skipped_from_logs and "跳过" not in str((stored_job.get("progress") or {}).get("stage") or ""))
                    ):
                        stored_job = dict(stored_job)
                        stored_job["running"] = False
                        stored_job["exit_code"] = 0
                        stored_job["error"] = None
                        stored_job["finished_at"] = stored_job.get("finished_at") or stored_job.get("updated_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z")
                        progress = dict(stored_job.get("progress") or {})
                        if skipped_from_logs:
                            progress["stage"] = "OSS 文件已存在，跳过上传" if stored_job.get("kind") == "oss_upload" else "本地文件已存在，跳过下载"
                        else:
                            progress["stage"] = "OSS 上传完成" if stored_job.get("kind") == "oss_upload" else "模型下载完成"
                        progress["percent"] = 100
                        progress["updated_at"] = stored_job["finished_at"]
                        stored_job["progress"] = progress
                        transfer_jobs[job_id] = stored_job
                        transfer_store_changed = True
                    stale_running = (
                        bool(stored_job.get("running"))
                        and normalized_job_id not in running_ids
                        and stored_job.get("pid") not in running_pids
                    )
                    if stale_running:
                        stored_job = dict(stored_job)
                        stored_job["running"] = False
                        stored_job["exit_code"] = 0 if success_from_logs else (stored_job.get("exit_code") if stored_job.get("exit_code") is not None else -1)
                        stored_job["finished_at"] = stored_job.get("finished_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z")
                        if success_from_logs:
                            stored_job["error"] = None
                            progress = dict(stored_job.get("progress") or {})
                            if skipped_from_logs:
                                progress["stage"] = "OSS 文件已存在，跳过上传" if stored_job.get("kind") == "oss_upload" else "本地文件已存在，跳过下载"
                            else:
                                progress["stage"] = "OSS 上传完成" if stored_job.get("kind") == "oss_upload" else "模型下载完成"
                            progress["percent"] = 100
                            progress["updated_at"] = stored_job["finished_at"]
                            stored_job["progress"] = progress
                        else:
                            stored_job["error"] = stored_job.get("error") or "传输进程已不在，可能被服务重启、系统重启或手动中断"
                        transfer_jobs[job_id] = stored_job
                        transfer_store_changed = True
                    postprocess_jobs.append(self._postprocess_job_public(stored_job, include_logs=True))
                    seen_postprocess_job_ids.add(normalized_job_id)
            if transfer_store_changed:
                self._atomic_json(self.postprocess_file, transfer_store)
            training_state = self.training_prep.state(tasks)
            for training_set in training_state.get("training_sets") or []:
                last_record = training_set.get("last_normalize_record")
                if isinstance(last_record, dict):
                    enriched = self._record_with_postprocess_details(last_record)
                    job_id = str(enriched.get("job_id") or "")
                    live_job = self.postprocess_jobs.get(job_id)
                    if live_job is not None:
                        enriched["status"] = "running" if live_job.get("running") else enriched.get("status")
                        enriched["running"] = bool(live_job.get("running"))
                    training_set["last_normalize_record"] = enriched
                add_latest_postprocess_job(training_set.get("last_normalize_record"))
            postprocess_jobs.sort(key=lambda item: item.get("started_at") or item.get("updated_at") or "", reverse=True)
            postprocess_jobs = postprocess_jobs[:POSTPROCESS_STORE_RECENT_LIMIT]
            return {
                "process": {
                    "running": running,
                    "pid": self.process.pid if self.process is not None else None,
                    "exit_code": exit_code,
                    "started_at": self.started_at,
                    "task": self.task,
                    "command": display_command(self.command) if self.command else "",
                },
                "teleop": self.ipc.state() if self.ipc is not None else {"online": False},
                "logs": list(self.logs),
                "data_dir": str(self.data_dir),
                "dataset_root": str(self.dataset_root),
                "config_file": str(self.config_file),
                "task_file": str(self.task_file),
                "postprocess_state_file": str(self.postprocess_state_file),
                "log_dir": str(self.log_dir),
                "log_file": str(self.logger._path_for_today()),
                "device": device,
                "tasks": tasks,
                "postprocess_jobs": postprocess_jobs,
                "archive_jobs": [
                    dict(job)
                    for job in sorted(self.archive_jobs.values(), key=lambda item: item.get("started_at") or "", reverse=True)
                ],
                "postprocess_defaults": {
                    "repo_owner": "local",
                    "camera_map": DEFAULT_CAMERA_MAP,
                    "image_size": "original",
                    "image_encoding": "auto",
                    "jpeg_quality": 95,
                    "video_backend": "",
                    "image_writer_processes": "",
                    "image_writer_threads": "",
                    "robot_type": "robot",
                    "openpi_dir": str(DEFAULT_OPENPI_DIR),
                    "lerobot_home": str(self.lerobot_home),
                    "openpi_work_dir": str(self.openpi_work_dir),
                    "openpi_assets_dir": str(self.openpi_assets_dir),
                    "cache_root": str(self.cache_root),
                    "config_prefix": "pi05",
                    "openpi_config_name": DEFAULT_OPENPI_CONFIG_NAME,
                },
                "training": training_state,
                "oss_transfer": {
                    "oss_root": self.oss_root,
                    "model_download_dir": str(self.model_download_dir),
                    "local_dir": "",
                    "local_parent": "",
                    "local_entries": [],
                    "packages": [],
                },
                "delivery": self.delivery_templates(),
                "camera_preview": self.camera_preview() if running else {"cameras": [], "error": None},
            }

    def close(self) -> None:
        if self.ipc is not None:
            self.ipc.close()


class AppHandler(BaseHTTPRequestHandler):
    manager: TeleopManager

    def do_GET(self) -> None:
        started = time.perf_counter()
        self._last_status = HTTPStatus.OK
        path = urlparse(self.path).path
        error: str | None = None
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/state":
                self._json(self.manager.state())
                return
            if path == "/api/tasks/file":
                query = parse_qs(parsed.query)
                try:
                    file_path = self.manager.dataset_file(
                        query.get("task_id", [""])[0],
                        query.get("episode", [""])[0],
                        query.get("path", [""])[0],
                    )
                except ValidationError as exc:
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                try:
                    body = file_path.read_bytes()
                except OSError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/":
                path = "/index.html"
            filename = path.removeprefix("/")
            if filename not in {"index.html", "app.js", "styles.css", "device.css"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            file_path = STATIC_ROOT / filename
            content_types = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}
            try:
                body = file_path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{content_types[file_path.suffix]}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except ValidationError as exc:
            error = str(exc)
            self.manager.logger.write(
                "warning",
                "http get validation failed",
                path=path,
                client=self.client_address[0],
                error=error,
            )
            self._json({"error": error}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            error = str(exc)
            self.manager.logger.write(
                "error",
                "http get failed",
                path=path,
                client=self.client_address[0],
                error=error,
                error_type=type(exc).__name__,
                traceback=traceback.format_exc(),
            )
            if path.startswith("/api/"):
                self._json({"error": error or "internal server error"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            else:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
        finally:
            self._log_access("GET", path, started, error=error)

    def do_POST(self) -> None:
        started = time.perf_counter()
        self._last_status = HTTPStatus.OK
        path = urlparse(self.path).path
        error: str | None = None
        try:
            payload = self._read_json()
            self.manager.logger.write(
                "info",
                "http post received",
                path=path,
                client=self.client_address[0],
            )
            if path == "/api/device/save":
                result = self.manager.save_device(payload.get("device"))
            elif path == "/api/tasks/create":
                result = self.manager.create_task(payload.get("task"))
            elif path == "/api/tasks/archive":
                result = self.manager.archive_task(payload.get("task_id"))
            elif path == "/api/tasks/preview":
                result = self.manager.preview_task(
                    payload.get("task_id"),
                    payload.get("limit", 20),
                    payload.get("episode"),
                    payload.get("frame_page", 1),
                    payload.get("frame_page_size", 3),
                )
            elif path == "/api/tasks/delete-episode":
                result = self.manager.delete_episode(payload.get("task_id"), payload.get("episode"))
            elif path == "/api/data/list":
                result = self.manager.data_list(payload)
            elif path == "/api/tasks/convert":
                result = self.manager.convert_dataset(payload)
            elif path == "/api/tasks/normalize":
                result = self.manager.normalize_dataset(payload)
            elif path == "/api/tasks/package-lerobot":
                result = self.manager.package_lerobot_dataset(payload)
            elif path == "/api/tasks/package-assets":
                result = self.manager.package_openpi_assets(payload)
            elif path == "/api/training/create-set":
                result = self.manager.create_training_set(payload)
            elif path == "/api/training/update-set":
                result = self.manager.update_training_set(payload)
            elif path == "/api/training/add-tasks":
                result = self.manager.add_tasks_to_training_set(payload)
            elif path == "/api/training/remove-task":
                result = self.manager.remove_task_from_training_set(payload)
            elif path == "/api/training/normalize":
                result = self.manager.normalize_training_set(payload)
            elif path == "/api/training/package":
                result = self.manager.package_training_set(payload)
            elif path == "/api/oss/local-packages":
                result = self.manager.oss_local_packages(payload)
            elif path == "/api/oss/list":
                result = self.manager.oss_list(payload)
            elif path == "/api/oss/upload":
                result = self.manager.oss_upload(payload)
            elif path == "/api/oss/download":
                result = self.manager.oss_download(payload)
            elif path == "/api/delivery/templates/save":
                result = self.manager.save_delivery_templates(payload)
            elif path == "/api/delivery/templates/reset":
                result = self.manager.reset_delivery_templates()
            elif path == "/api/start":
                result = self.manager.start_task(payload.get("task_id"))
            elif path == "/api/control":
                action = payload.get("action", "")
                self.manager.logger.write(
                    "info",
                    "teleop control request received",
                    action=action,
                    client=self.client_address[0],
                    referer=self.headers.get("Referer", ""),
                    user_agent=self.headers.get("User-Agent", ""),
                )
                result = self.manager.control(action)
            elif path == "/api/ik-replay/target":
                result = self.manager.ik_replay_target(payload)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._json(result)
        except (ValidationError, TrainingPrepError) as exc:
            error = str(exc)
            self.manager.logger.write(
                "warning",
                "http post validation failed",
                path=urlparse(self.path).path,
                client=self.client_address[0],
                error=error,
            )
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            error = str(exc)
            self.manager.logger.write(
                "error",
                "http post failed",
                path=urlparse(self.path).path,
                client=self.client_address[0],
                error=error,
                error_type=type(exc).__name__,
                traceback=traceback.format_exc(),
            )
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        finally:
            self._log_access("POST", path, started, error=error)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValidationError("Content-Length 不正确") from exc
        if length <= 0 or length > 64 * 1024:
            raise ValidationError("请求内容为空或过大")
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValidationError("请求不是有效 JSON") from exc
        if not isinstance(payload, dict):
            raise ValidationError("请求格式不正确")
        return payload

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._last_status = status
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        self._last_status = HTTPStatus(code)
        super().send_error(code, message, explain)

    def _log_access(self, method: str, path: str, started: float, *, error: str | None = None) -> None:
        status = int(getattr(self, "_last_status", HTTPStatus.OK))
        if method == "GET" and path == "/api/state" and status < 400 and not error:
            return
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        level = "error" if status >= 500 else "warning" if status >= 400 else "access"
        fields: dict[str, Any] = {
            "method": method,
            "path": path,
            "status": status,
            "duration_ms": duration_ms,
            "client": self.client_address[0],
        }
        if error:
            fields["error"] = error
        self.manager.logger.write(level, "http request", **fields)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Suppress BaseHTTPRequestHandler access logs such as the frequent
        # "GET /api/state" polling lines. Business events, POST actions and
        # errors are logged explicitly elsewhere.
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{PROJECT_NAME} 数采控制台")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址；局域网访问可设为 0.0.0.0")
    parser.add_argument("--port", type=int, default=18088, help="HTTP 端口")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="数据集根目录；未设置时读取配置文件 device.config.data_dir")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE, help="设备配置文件")
    parser.add_argument("--task-file", type=Path, default=None, help="数采任务清单 JSON 文件")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="按天保存的平台日志目录")
    args = parser.parse_args()

    config_file = args.config.expanduser().resolve()
    configured_data_dir = data_dir_from_config(config_file)
    dataset_root = (
        args.dataset_dir.expanduser().resolve()
        if args.dataset_dir is not None
        else ((configured_data_dir / "datasets" / "robot").resolve() if configured_data_dir else DEFAULT_DATASET_ROOT.expanduser().resolve())
    )
    legacy_dataset_root = (LEGACY_DATA_DIR / "datasets" / "robot").resolve()
    if dataset_root == legacy_dataset_root and not os.environ.get("XR_TELEOP_DATASET_DIR"):
        dataset_root = DEFAULT_DATASET_ROOT.expanduser().resolve()
    task_file = args.task_file.expanduser().resolve() if args.task_file else dataset_root / "tasks.json"
    legacy_task_file = legacy_dataset_root / "tasks.json"
    if task_file == legacy_task_file and dataset_root != legacy_dataset_root and not os.environ.get("XR_TELEOP_TASK_FILE"):
        task_file = dataset_root / "tasks.json"
    log_dir = args.log_dir.expanduser().resolve()
    manager = TeleopManager(dataset_root, config_file, task_file, log_dir)
    AppHandler.manager = manager
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    manager.logger.write(
        "info",
        "teleop web server started",
        host=args.host,
        port=args.port,
        dataset_root=str(manager.dataset_root),
        config_file=str(manager.config_file),
        task_file=str(manager.task_file),
        log_dir=str(manager.log_dir),
    )
    print(f"{PROJECT_NAME} 数采控制台: http://{args.host}:{args.port}")
    print(f"数据保存目录: {manager.dataset_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.logger.write("info", "teleop web server stopping")
        manager.close()
        server.server_close()


if __name__ == "__main__":
    main()
