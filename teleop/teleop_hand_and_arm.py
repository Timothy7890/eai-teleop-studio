import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
import cv2
import json
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for import_dir in (
    parent_dir,
    os.path.join(current_dir, "teleimager", "src"),
    os.path.join(current_dir, "robot_control", "dex-retargeting", "src"),
):
    if import_dir not in sys.path:
        sys.path.insert(0, import_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import (
    G1_29_ArmController,
    G1_23_ArmController,
    H1_2_ArmController,
    H1_ArmController,
    H2_ArmController,
    H2_JointIndex,
)
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK
from teleop.robot_control.end_effectors import (
    ACTIVE_END_EFFECTORS,
    DISPLAY_END_EFFECTORS,
    PASSIVE_END_EFFECTORS,
    SIDE_END_EFFECTORS,
    SINGLE_SIDE_ACTIVE_END_EFFECTORS,
    canonical_end_effector,
)
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.hand_eye_capture import (
    CAPTURING as HAND_EYE_CAPTURING,
    FOLLOW as HAND_EYE_FOLLOW,
    HOLD as HAND_EYE_HOLD,
    RESUMING as HAND_EYE_RESUMING,
    SAVING as HAND_EYE_SAVING,
    SETTLING as HAND_EYE_SETTLING,
    HandEyeCaptureState,
    build_hand_eye_hud_status,
)
from teleop.utils.hand_eye_recorder import HandEyeRecorder
from teleop.utils.hand_eye_trajectory import (
    HandEyeTrajectoryRecorder,
    HandEyeTrajectoryReplay,
)
from teleop.utils.ik_replay_live import IKReplayLivePusher, build_ik_replay_live_payload
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


def default_webrtc_scheme():
    scheme = os.environ.get('XR_TELEOP_WEBRTC_SCHEME', 'http').strip().lower()
    return scheme if scheme in {'http', 'https'} else 'http'


def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
EXTERNAL_ARM_TARGET = None
EXTERNAL_ARM_TARGET_LOCK = threading.Lock()
EXTERNAL_ARM_TARGET_TIMEOUT = 0.5
HAND_EYE_CAPTURE = None
HAND_EYE_REPLAY = None
HAND_EYE_TRAJECTORY_PATH = None
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    elif key == 'c' and START == True and HAND_EYE_CAPTURE is not None:
        HAND_EYE_CAPTURE.request_toggle()
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    with EXTERNAL_ARM_TARGET_LOCK:
        external_target = dict(EXTERNAL_ARM_TARGET or {})
    external_age = time.time() - external_target.get("received_at", 0.0) if external_target else None
    state = {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
        "EXTERNAL_ARM_TARGET_ACTIVE": external_age is not None and external_age <= EXTERNAL_ARM_TARGET_TIMEOUT,
        "EXTERNAL_ARM_TARGET_SOURCE": external_target.get("source"),
    }
    if HAND_EYE_CAPTURE is not None:
        state.update({"HAND_EYE_ENABLED": True, **HAND_EYE_CAPTURE.snapshot()})
    else:
        state["HAND_EYE_ENABLED"] = False
    if HAND_EYE_REPLAY is not None:
        state.update({"TRAJECTORY_REPLAY_ENABLED": True, **HAND_EYE_REPLAY.progress})
    else:
        state["TRAJECTORY_REPLAY_ENABLED"] = False
    state["HAND_EYE_TRAJECTORY_PATH"] = HAND_EYE_TRAJECTORY_PATH
    return state


def update_vr_hud(tv_wrapper, *, started: bool, motion_ready: bool = True) -> None:
    if HAND_EYE_CAPTURE is not None:
        title, detail, level = build_hand_eye_hud_status(
            HAND_EYE_CAPTURE.snapshot(),
            started=started,
            motion_ready=motion_ready,
        )
    elif not started:
        title, detail, level = (
            "等待开始遥操",
            "确认追踪后按 A，或在电脑点击“开始遥操”",
            "info",
        )
    elif not motion_ready:
        title, detail, level = (
            "等待 VR 追踪",
            "确认控制器权限和手柄连接",
            "warning",
        )
    else:
        title, detail, level = "遥操运行中", "A：结束遥操", "success"
    tv_wrapper.set_hud_status(title, detail, level)


def set_external_arm_target(msg: dict):
    global EXTERNAL_ARM_TARGET
    raw = msg.get("target_q") or msg.get("target_joints") or msg.get("qpos")
    if raw is None:
        actions = msg.get("actions")
        if isinstance(actions, dict):
            left = ((actions.get("left_arm") or {}).get("qpos"))
            right = ((actions.get("right_arm") or {}).get("qpos"))
            if left is not None and right is not None:
                raw = [*left, *right]
    if raw is None:
        raise ValueError("CMD_SET_ARM_TARGET requires target_q/target_joints/qpos or actions.left_arm/right_arm.qpos")
    target = np.asarray(raw, dtype=float)
    if target.shape != (14,):
        raise ValueError(f"external arm target must contain 14 values, got {target.shape}")
    with EXTERNAL_ARM_TARGET_LOCK:
        EXTERNAL_ARM_TARGET = {
            "target_q": target,
            "source": str(msg.get("source") or "external"),
            "received_at": time.time(),
        }


def get_external_arm_target():
    with EXTERNAL_ARM_TARGET_LOCK:
        target = dict(EXTERNAL_ARM_TARGET or {})
    if not target:
        return None
    if time.time() - target.get("received_at", 0.0) > EXTERNAL_ARM_TARGET_TIMEOUT:
        return None
    return target["target_q"]

XR_QUAD_CAMERA_ORDER = [
    "head_camera",
    "torso_camera",
    "left_wrist_camera",
    "right_wrist_camera",
]

XR_QUAD_CAMERA_LABELS = {
    "head_camera": "HEAD",
    "torso_camera": "TORSO",
    "left_wrist_camera": "LEFT WRIST",
    "right_wrist_camera": "RIGHT WRIST",
}

def ordered_unique(names):
    seen = set()
    result = []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result

def get_camera_image_shape(camera_config, camera_name="head_camera"):
    camera_cfg = camera_config.get(camera_name, {})
    image_shape = camera_cfg.get("image_shape", [480, 640])
    if len(image_shape) < 2:
        return (480, 640)
    return (int(image_shape[0]), int(image_shape[1]))

def compose_xr_quad_view(image_frames, camera_names, output_shape):
    """Compose up to four BGR camera frames into one 2x2 BGR image for XR display."""
    output_h, output_w = output_shape
    output_h = max(2, int(output_h))
    output_w = max(2, int(output_w))
    cell_h = output_h // 2
    cell_w = output_w // 2
    canvas = np.zeros((cell_h * 2, cell_w * 2, 3), dtype=np.uint8)

    for idx, camera_name in enumerate(camera_names[:4]):
        row = idx // 2
        col = idx % 2
        y0 = row * cell_h
        x0 = col * cell_w
        tele_img = image_frames.get(camera_name)
        frame = tele_img.bgr if tele_img is not None else None

        if frame is None:
            tile = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            cv2.putText(
                tile,
                f"{XR_QUAD_CAMERA_LABELS.get(camera_name, camera_name)} NO SIGNAL",
                (18, max(32, cell_h // 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            tile = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

        cv2.rectangle(tile, (0, 0), (cell_w - 1, cell_h - 1), (255, 255, 255), 1)
        cv2.rectangle(tile, (0, 0), (min(cell_w - 1, 230), 34), (0, 0, 0), -1)
        cv2.putText(
            tile,
            XR_QUAD_CAMERA_LABELS.get(camera_name, camera_name),
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = tile

    return canvas

H2_DUAL_ARM_POSITION_NAMES = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

H2_POSITION_JOINT_INDEX = {
    "left_hip_pitch": H2_JointIndex.kLeftHipPitch,
    "left_hip_roll": H2_JointIndex.kLeftHipRoll,
    "left_hip_yaw": H2_JointIndex.kLeftHipYaw,
    "left_knee": H2_JointIndex.kLeftKnee,
    "left_ankle_pitch": H2_JointIndex.kLeftAnklePitch,
    "left_ankle_roll": H2_JointIndex.kLeftAnkleRoll,
    "right_hip_pitch": H2_JointIndex.kRightHipPitch,
    "right_hip_roll": H2_JointIndex.kRightHipRoll,
    "right_hip_yaw": H2_JointIndex.kRightHipYaw,
    "right_knee": H2_JointIndex.kRightKnee,
    "right_ankle_pitch": H2_JointIndex.kRightAnklePitch,
    "right_ankle_roll": H2_JointIndex.kRightAnkleRoll,
    "waist_yaw": H2_JointIndex.kWaistYaw,
    "waist_roll": H2_JointIndex.kWaistRoll,
    "waist_pitch": H2_JointIndex.kWaistPitch,
    "left_shoulder_pitch": H2_JointIndex.kLeftShoulderPitch,
    "left_shoulder_roll": H2_JointIndex.kLeftShoulderRoll,
    "left_shoulder_yaw": H2_JointIndex.kLeftShoulderYaw,
    "left_elbow": H2_JointIndex.kLeftElbow,
    "left_wrist_roll": H2_JointIndex.kLeftWristRoll,
    "left_wrist_pitch": H2_JointIndex.kLeftWristPitch,
    "left_wrist_yaw": H2_JointIndex.kLeftWristyaw,
    "right_shoulder_pitch": H2_JointIndex.kRightShoulderPitch,
    "right_shoulder_roll": H2_JointIndex.kRightShoulderRoll,
    "right_shoulder_yaw": H2_JointIndex.kRightShoulderYaw,
    "right_elbow": H2_JointIndex.kRightElbow,
    "right_wrist_roll": H2_JointIndex.kRightWristRoll,
    "right_wrist_pitch": H2_JointIndex.kRightWristPitch,
    "right_wrist_yaw": H2_JointIndex.kRightWristYaw,
    "head_pitch": H2_JointIndex.kHeadPitch,
    "head_yaw": H2_JointIndex.kHeadYaw,
}
H2_DUAL_ARM_JOINT_VALUES = {
    H2_POSITION_JOINT_INDEX[name].value for name in H2_DUAL_ARM_POSITION_NAMES
}


def load_pose_payload(path):
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        return json.load(f)


def pose_unit(payload):
    unit = str(payload.get("unit", "rad")).lower()
    if unit not in {"rad", "radian", "radians", "deg", "degree", "degrees"}:
        raise ValueError(f"Unsupported pose unit in --init-arm-pose-file: {unit}")
    return unit


def convert_pose_values(values, unit):
    qpos = np.asarray(values, dtype=float)
    if unit in {"deg", "degree", "degrees"}:
        qpos = np.deg2rad(qpos)
    return qpos


def load_dual_arm_pose(path):
    payload = load_pose_payload(path)
    unit = pose_unit(payload)
    if "qpos" in payload:
        qpos = convert_pose_values(payload.get("qpos"), unit)
    elif isinstance(payload.get("positions"), dict):
        positions = payload["positions"]
        missing = [name for name in H2_DUAL_ARM_POSITION_NAMES if name not in positions]
        if missing:
            raise ValueError(f"--init-arm-pose-file positions missing dual-arm joints: {missing}")
        qpos = convert_pose_values([positions[name] for name in H2_DUAL_ARM_POSITION_NAMES], unit)
    else:
        raise ValueError("--init-arm-pose-file must contain qpos or positions")
    if qpos.shape != (14,):
        raise ValueError(f"--init-arm-pose-file must contain 14 qpos values, got {qpos.shape}")
    return qpos


def load_h2_pose_targets(path):
    payload = load_pose_payload(path)
    init_arm_q = load_dual_arm_pose(path)
    locked_joint_targets = {}
    if isinstance(payload.get("positions"), dict):
        unit = pose_unit(payload)
        positions = payload["positions"]
        for name, joint_index in H2_POSITION_JOINT_INDEX.items():
            if joint_index.value in H2_DUAL_ARM_JOINT_VALUES or name not in positions:
                continue
            locked_joint_targets[joint_index] = float(convert_pose_values([positions[name]], unit)[0])
    return init_arm_q, locked_joint_targets

def smoothstep_progress(progress):
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress * progress * (3.0 - 2.0 * progress)


def move_dual_arm_to_pose(
    arm_ctrl,
    target_q,
    duration=5.0,
    respect_stop=True,
    gravity_torques=None,
):
    duration = max(float(duration), 0.1)
    target_q = np.asarray(target_q, dtype=float)
    start_q = arm_ctrl.get_current_dual_arm_q()
    logger_mp.info(f"Moving dual arms smoothly over {duration:.2f}s: {target_q}")
    start_time = time.monotonic()
    deadline = start_time + duration
    while time.monotonic() < deadline and (not respect_stop or not STOP):
        alpha = smoothstep_progress((time.monotonic() - start_time) / duration)
        command_q = start_q + (target_q - start_q) * alpha
        tau = (
            gravity_torques(command_q)
            if gravity_torques is not None
            else np.zeros_like(command_q)
        )
        arm_ctrl.ctrl_dual_arm(command_q, tau)
        time.sleep(0.004)
    if respect_stop and STOP:
        return
    final_tau = (
        gravity_torques(target_q)
        if gravity_torques is not None
        else np.zeros_like(target_q)
    )
    arm_ctrl.ctrl_dual_arm(target_q, final_tau)



def move_h2_to_pose(
    arm_ctrl,
    target_arm_q,
    locked_joint_targets=None,
    duration=5.0,
    respect_stop=True,
    gravity_torques=None,
):
    locked_joint_targets = locked_joint_targets or {}
    if not locked_joint_targets:
        move_dual_arm_to_pose(
            arm_ctrl,
            target_arm_q,
            duration,
            respect_stop,
            gravity_torques,
        )
        return

    logger_mp.info(
        f"Moving H2 arms and locked body joints to init pose over {duration:.2f}s; "
        f"locked joints: {[joint.name for joint in locked_joint_targets]}"
    )
    start_time = time.monotonic()
    duration = max(duration, 0.1)
    deadline = start_time + duration
    start_arm_q = arm_ctrl.get_current_dual_arm_q()
    current_motor_q = arm_ctrl.get_current_motor_q()
    start_locked_targets = {
        joint: float(current_motor_q[joint]) for joint in locked_joint_targets
    }

    while time.monotonic() < deadline and (not respect_stop or not STOP):
        alpha = smoothstep_progress((time.monotonic() - start_time) / duration)
        arm_q = start_arm_q + (target_arm_q - start_arm_q) * alpha
        body_q = {
            joint: start_locked_targets[joint] + (target_q - start_locked_targets[joint]) * alpha
            for joint, target_q in locked_joint_targets.items()
        }
        arm_ctrl.set_locked_joint_targets(body_q)
        tau = (
            gravity_torques(arm_q)
            if gravity_torques is not None
            else np.zeros_like(arm_q)
        )
        arm_ctrl.ctrl_dual_arm(arm_q, tau)
        time.sleep(0.004)

    if respect_stop and STOP:
        return
    arm_ctrl.set_locked_joint_targets(locked_joint_targets)
    final_tau = (
        gravity_torques(target_arm_q)
        if gravity_torques is not None
        else np.zeros_like(target_arm_q)
    )
    arm_ctrl.ctrl_dual_arm(target_arm_q, final_tau)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--xr-view', type=str, choices=['quad', 'head'], default='head',
                        help='XR camera view. quad shows head/torso/left_wrist/right_wrist in one 2x2 view; head keeps the original head camera view.')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=DISPLAY_END_EFFECTORS, help='Select end effector controller')
    parser.add_argument('--left-ee', type=str, choices=SIDE_END_EFFECTORS, help='Select left end effector controller')
    parser.add_argument('--right-ee', type=str, choices=SIDE_END_EFFECTORS, help='Select right end effector controller')
    # network parameters
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--webrtc-server-ip', type=str, default=None,
                        help='Browser-reachable WebRTC server IP. Defaults to --img-server-ip.')
    parser.add_argument('--webrtc-scheme', type=str, choices=['http', 'https'],
                        default=default_webrtc_scheme(),
                        help='WebRTC video URL scheme. Defaults to XR_TELEOP_WEBRTC_SCHEME or http.')
    parser.add_argument('--network-interface', type=str, default='enp86s0', help='Network interface for dds communication, e.g., enp86s0, enp87s0, eth0, wlan0.')
    parser.add_argument('--init-arm-pose-file', type=str, default=None, help='JSON file containing a 14-dim H2/G1_29/H1_2 dual-arm qpos init pose. H2 defaults to config/h2_pose_init.json when present.')
    parser.add_argument('--init-arm-pose-duration', type=float, default=5.0, help='Seconds used to move to --init-arm-pose-file before waiting for start.')
    parser.add_argument('--exit-arm-pose-duration', type=float, default=None, help='Seconds used to move back to the init arm pose during safe exit. Defaults to --init-arm-pose-duration.')
    parser.add_argument('--arm-reference-mode', type=str, choices=['world', 'head_position', 'head_yaw'], default=None,
                        help='XR arm reference frame for IK. H2 defaults to world to avoid head motion driving the arms.')
    parser.add_argument('--ik-replay-live-enable', action='store_true',
                        help='Enable best-effort live arm-state push to an IK replay service.')
    parser.add_argument('--ik-replay-live-url', type=str, default=os.environ.get('IK_REPLAY_LIVE_URL', ''),
                        help='POST URL for IK replay live state, e.g. http://192.168.61.228:8000/api/live/state.')
    parser.add_argument('--ik-replay-live-fps', type=float, default=float(os.environ.get('IK_REPLAY_LIVE_FPS', '10')),
                        help='Max live-state push frequency when IK replay live push is enabled.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    parser.add_argument('--no-camera', action='store_true',help='Disable all camera input and use XR pass-through display mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--hand-eye-record', action='store_true',
                        help='Enable sparse hand-eye RGB-D capture with absolute XR-to-IK following.')
    parser.add_argument('--hand-eye-settle-seconds', type=float, default=0.5,
                        help='Required stable time before a hand-eye burst is captured.')
    parser.add_argument('--hand-eye-max-joint-speed', type=float, default=0.05,
                        help='Maximum absolute right-arm joint speed in rad/s while settling.')
    parser.add_argument('--hand-eye-max-joint-span', type=float, default=0.003,
                        help='Maximum right-arm joint position span in rad over the settling window.')
    parser.add_argument('--hand-eye-max-hold-error', type=float, default=0.30,
                        help='Maximum measured right-arm hold drift in rad before capture is blocked.')
    parser.add_argument('--hand-eye-resume-seconds', type=float, default=2.0,
                        help='Seconds used to move smoothly to the current absolute XR target after saving.')
    parser.add_argument('--hand-eye-burst-frames', type=int, default=5,
                        help='Number of unique RGB-D frames saved per hand-eye sample.')
    parser.add_argument('--hand-eye-replay', type=str, default='',
                        help='Completed trajectory directory or trajectory.npz to replay without XR control.')
    parser.add_argument('--hand-eye-replay-time-scale', type=float, default=1.0,
                        help='Replay speed in (0, 1]; preserves every recorded joint-space sample.')
    parser.add_argument('--hand-eye-replay-start-tolerance', type=float, default=0.05,
                        help='Maximum initial joint mismatch in rad before replay is rejected.')
    parser.add_argument('--hand-eye-replay-tracking-limit', type=float, default=0.08,
                        help='Maximum persistent measured trajectory error in rad.')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    args.hand_eye_replay = args.hand_eye_replay.strip()
    if args.hand_eye_replay:
        args.hand_eye_record = True
    if args.hand_eye_record and args.arm != "H2":
        raise ValueError("--hand-eye-record currently supports only --arm=H2.")
    if args.hand_eye_record and not args.record:
        logger_mp.info("--hand-eye-record enables recording without the continuous EpisodeWriter.")
    if args.hand_eye_record:
        HAND_EYE_CAPTURE = HandEyeCaptureState(
            settle_seconds=args.hand_eye_settle_seconds,
            max_joint_speed=args.hand_eye_max_joint_speed,
            max_joint_span=args.hand_eye_max_joint_span,
            max_hold_error=args.hand_eye_max_hold_error,
            burst_frames=args.hand_eye_burst_frames,
        )
    if args.hand_eye_replay:
        HAND_EYE_REPLAY = HandEyeTrajectoryReplay(
            args.hand_eye_replay,
            start_tolerance=args.hand_eye_replay_start_tolerance,
            tracking_error_limit=args.hand_eye_replay_tracking_limit,
            time_scale=args.hand_eye_replay_time_scale,
        )
    if args.arm_reference_mode is None:
        args.arm_reference_mode = 'head_position' if args.arm == 'H2' else 'head_yaw'
    PASSIVE_EE = set(PASSIVE_END_EFFECTORS)
    ACTIVE_EE = set(ACTIVE_END_EFFECTORS)
    if args.ee and (args.left_ee or args.right_ee):
        raise ValueError("--ee cannot be used together with --left-ee/--right-ee.")
    if args.ee:
        left_ee = canonical_end_effector(args.ee)
        right_ee = canonical_end_effector(args.ee)
    else:
        left_ee = canonical_end_effector(args.left_ee or 'none')
        right_ee = canonical_end_effector(args.right_ee or 'none')
    left_ee_active = left_ee not in PASSIVE_EE
    right_ee_active = right_ee not in PASSIVE_EE
    if left_ee_active and left_ee not in ACTIVE_EE:
        raise ValueError(f"Unsupported left end effector: {left_ee}")
    if right_ee_active and right_ee not in ACTIVE_EE:
        raise ValueError(f"Unsupported right end effector: {right_ee}")
    if args.input_mode == "controller" and (left_ee_active or right_ee_active):
        raise ValueError("Controller input mode does not support active end-effector control.")
    active_ee_set = {ee for ee in (left_ee, right_ee) if ee not in PASSIVE_EE}
    if len(active_ee_set) > 1:
        raise ValueError(f"End effectors must match, or use one active side with one passive side; mixed active end effectors are not supported: left={left_ee}, right={right_ee}")
    if left_ee_active != right_ee_active and next(iter(active_ee_set), None) not in SINGLE_SIDE_ACTIVE_END_EFFECTORS:
        supported = "/".join(sorted(SINGLE_SIDE_ACTIVE_END_EFFECTORS))
        raise ValueError(f"Single-side active control is currently supported only for {supported}.")
    ee_type = next(iter(active_ee_set), None)
    args.ee = ee_type
    logger_mp.info(f"End effector config: left={left_ee}, right={right_ee}, active_type={ee_type}")
    if args.arm == "H2" and args.init_arm_pose_file is None:
        default_h2_init_pose = os.path.join(parent_dir, "config", "h2_pose_init.json")
        if os.path.exists(default_h2_init_pose):
            args.init_arm_pose_file = default_h2_init_pose
            logger_mp.info(f"Using default H2 init arm pose file: {args.init_arm_pose_file}")
        else:
            logger_mp.warning(f"Default H2 init arm pose file not found: {default_h2_init_pose}")
    if args.exit_arm_pose_duration is None:
        args.exit_arm_pose_duration = args.init_arm_pose_duration
    if args.hand_eye_record:
        args.exit_arm_pose_duration = max(args.exit_arm_pose_duration, 5.0)
    logger_mp.debug(f"args: {args}")

    # 先设置为空，避免无相机模式退出时找不到该变量
    img_client = None
    loco_wrapper = None
    motion_switcher = None
    init_arm_q = None
    init_locked_joint_targets = {}
    hand_eye_start_q = None
    hand_eye_resume_transition = None
    ik_replay_pusher = None
    recorder = None
    hand_eye_recorder = None
    trajectory_recorder = None
    left_fixed_q = None
    left_fixed_wrist_pose = None

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press, get_state=get_state, on_arm_target=set_external_arm_target)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        if args.no_camera:
            args.display_mode = 'pass-through'

            camera_config = {
                'head_camera': {
                    'enable_zmq': False,
                    'enable_webrtc': False,
                    'webrtc_port': 60001,
                    'binocular': False,
                    'image_shape': [480, 640],
                },
                'left_wrist_camera': {
                    'enable_zmq': False,
                },
                'torso_camera': {
                    'enable_zmq': False,
                },
                'right_wrist_camera': {
                    'enable_zmq': False,
                },
            }

            logger_mp.info(
                "Camera disabled: only robot states and actions will be recorded."
            )
        else:
            img_client = ImageClient(
                host=args.img_server_ip,
                request_bgr=True,
                auto_subscribe=not args.hand_eye_record,
            )
            camera_config = img_client.get_cam_config()
            logger_mp.debug(f"Camera config: {camera_config}")
        hand_eye_camera_config = camera_config.get("head_rgbd_camera", {})
        if args.hand_eye_record:
            if args.no_camera:
                raise ValueError("--hand-eye-record requires camera input.")
            if (
                not isinstance(hand_eye_camera_config, dict)
                or not hand_eye_camera_config.get("enable_zmq")
                or hand_eye_camera_config.get("data_format") != "rgbd"
            ):
                raise RuntimeError(
                    "head_rgbd_camera must be an enabled ZMQ RGB-D stream for hand-eye recording."
                )
            hand_eye_recorder = HandEyeRecorder(
                os.path.join(args.task_dir, args.task_name)
            )
        record_camera_names = [
            camera_name
            for camera_name, camera_cfg in camera_config.items()
            if isinstance(camera_cfg, dict)
            and camera_cfg.get('enable_zmq')
            and camera_cfg.get('data_format', 'jpeg') == 'jpeg'
        ]
        logger_mp.info(f"Recording camera streams from config order: {record_camera_names}")
        xr_quad_view = (
            args.xr_view == 'quad'
            and not args.no_camera
            and args.display_mode != 'pass-through'
        )
        if args.hand_eye_record and xr_quad_view:
            logger_mp.warning(
                "Hand-eye recording uses the head camera only; falling back from XR quad view to head view."
            )
            xr_quad_view = False
        xr_camera_names = [
            camera_name
            for camera_name in XR_QUAD_CAMERA_ORDER
            if isinstance(camera_config.get(camera_name), dict)
            and camera_config[camera_name].get('enable_zmq')
            and camera_config[camera_name].get('data_format', 'jpeg') == 'jpeg'
        ]
        if xr_quad_view and not xr_camera_names:
            logger_mp.warning("XR quad view requested, but no ZMQ camera is enabled. Falling back to head view.")
            xr_quad_view = False

        xr_display_shape = get_camera_image_shape(camera_config, 'head_camera')
        xr_binocular = camera_config['head_camera']['binocular'] and not xr_quad_view
        xr_use_webrtc = camera_config['head_camera']['enable_webrtc'] and not xr_quad_view
        xr_use_zmq = (
            camera_config['head_camera']['enable_zmq']
            if not xr_quad_view
            else bool(xr_camera_names)
        )
        xr_need_local_img = not (args.display_mode == 'pass-through' or xr_use_webrtc)
        runtime_record_camera_names = [] if args.hand_eye_record else record_camera_names
        runtime_camera_names = ordered_unique(
            runtime_record_camera_names
            + (xr_camera_names if xr_quad_view else ['head_camera'])
        )
        logger_mp.info(
            f"XR view: {'quad' if xr_quad_view else 'head'}, "
            f"local_render={xr_need_local_img}, cameras={xr_camera_names if xr_quad_view else ['head_camera']}"
        )

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        webrtc_server_ip = args.webrtc_server_ip or args.img_server_ip
        webrtc_url = f"{args.webrtc_scheme}://{webrtc_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer"
        logger_mp.info(f"XR WebRTC video URL: {webrtc_url}")
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=xr_binocular,
                                     img_shape=xr_display_shape,
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=xr_use_zmq,
                                     webrtc=xr_use_webrtc,
                                     webrtc_url=webrtc_url,
                                     arm_reference_mode=args.arm_reference_mode
                                     )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller" or args.arm == "H2":
                loco_wrapper = LocoClientWrapper(arm=args.arm)
                motion_status = loco_wrapper.GetStatusSummary()
                logger_mp.info(f"Motion loco status: {motion_status}")
                if args.arm == "H2":
                    arm_sdk_ready, arm_sdk_info = loco_wrapper.EnsureArmSDKEnabled()
                    if not arm_sdk_ready:
                        raise RuntimeError(f"H2 ArmSDK enable/check failed: {arm_sdk_info}")
                    logger_mp.info(f"H2 ArmSDK ready: {arm_sdk_info}")

                    motion_status = loco_wrapper.GetStatusSummary()
                    logger_mp.info(f"H2 motion status after ArmSDK check: {motion_status}")
                    h2_fsm_id = motion_status.get("fsm_id")
                    if motion_status.get("fsm_id_code") == 0 and h2_fsm_id in (0, 1, 2, 3):
                        logger_mp.warning(
                            "H2 --motion enables ArmSDK only and does not switch the full-body FSM. "
                            f"current_fsm_id={h2_fsm_id}; switch H2 to the desired locomotion/control "
                            "mode externally if base walking is expected."
                        )
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)
        elif args.arm == "H2":
            arm_ik = H2_ArmIK()
            arm_ctrl = H2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)

        if args.init_arm_pose_file:
            if args.arm == "H2":
                init_arm_q, init_locked_joint_targets = load_h2_pose_targets(args.init_arm_pose_file)
            else:
                init_arm_q = load_dual_arm_pose(args.init_arm_pose_file)
            current_dim = arm_ctrl.get_current_dual_arm_q().shape[0]
            if init_arm_q.shape[0] != current_dim:
                raise ValueError(f"Init arm pose dim {init_arm_q.shape[0]} does not match current arm dim {current_dim}")
            if args.arm == "H2":
                h2_locked_targets = {} if args.motion else init_locked_joint_targets
                startup_duration = args.init_arm_pose_duration
                if args.hand_eye_record and HAND_EYE_REPLAY is None:
                    startup_duration = max(startup_duration, 5.0)
                    tv_wrapper.set_hud_status(
                        "正在进入固定初始姿态…",
                        "手柄运动暂不参与控制，请等待到位",
                        "warning",
                    )
                move_h2_to_pose(
                    arm_ctrl,
                    init_arm_q,
                    h2_locked_targets,
                    startup_duration,
                    gravity_torques=arm_ik.gravity_torques,
                )
            else:
                move_dual_arm_to_pose(arm_ctrl, init_arm_q, args.init_arm_pose_duration)

        if args.hand_eye_record:
            current_arm_q = arm_ctrl.get_current_dual_arm_q()
            if HAND_EYE_REPLAY is not None:
                left_fixed_q = HAND_EYE_REPLAY.left_fixed_q.copy()
                logger_mp.info(
                    f"Hand-eye deterministic replay loaded: {HAND_EYE_REPLAY.directory}, "
                    f"frames={HAND_EYE_REPLAY.right_command_q.shape[0]}, "
                    f"events={len(HAND_EYE_REPLAY.events)}"
                )
            else:
                left_fixed_q = current_arm_q[:7].copy()
                trajectory_recorder = HandEyeTrajectoryRecorder(
                    os.path.join(args.task_dir, args.task_name),
                    left_fixed_q=left_fixed_q,
                    frequency=args.frequency,
                )
                HAND_EYE_TRAJECTORY_PATH = str(trajectory_recorder.directory)
                logger_mp.info(
                    f"Hand-eye trajectory recording started: {trajectory_recorder.directory}"
                )
                arm_ctrl.ctrl_dual_arm(
                    np.concatenate([left_fixed_q, current_arm_q[-7:]]),
                    arm_ik.gravity_torques(
                        np.concatenate([left_fixed_q, current_arm_q[-7:]])
                    ),
                )
            left_fixed_wrist_pose, _ = arm_ik.forward_wrist_poses(
                np.concatenate([left_fixed_q, current_arm_q[-7:]])
            )
            hand_eye_start_q = np.concatenate([
                left_fixed_q,
                current_arm_q[-7:],
            ])
            if HAND_EYE_REPLAY is not None:
                HAND_EYE_CAPTURE.enable_follow()

        # end-effector
        xr_motion_data_ready = Value('b', False, lock=True)        # [input] whether XR hand/controller motion data has arrived
        if args.ee in ("dex3", "inspire_ftp", "inspire_dfx") and args.input_mode == "controller":
            raise ValueError(f"{args.ee} does not support controller input mode.")
        elif args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array,
                                               simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready,
                                               enable_left=left_ee_active, enable_right=right_ee_active)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array,
                                               simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready,
                                               enable_left=left_ee_active, enable_right=right_ee_active)
        elif args.ee == "brainco" and args.input_mode == "hand":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_hand
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_hand(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                                dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "controller":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_ctrl
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        else:
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record and not args.hand_eye_record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        if args.ik_replay_live_enable and args.ik_replay_live_url.strip():
            ik_replay_pusher = IKReplayLivePusher(
                args.ik_replay_live_url,
                robot=args.arm.lower(),
                fps=args.ik_replay_live_fps,
                logger=logger_mp,
            )
            logger_mp.info(
                f"IK replay live push enabled: url={args.ik_replay_live_url}, "
                f"fps={args.ik_replay_live_fps:g}"
            )

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press controller [A], Web start, or [r] to begin teleoperation.")
        if args.hand_eye_record:
            logger_mp.info(
                "🟠  First controller [B] enables following; later [B] presses "
                "hold/capture and resume absolute following."
            )
        elif args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        update_vr_hud(tv_wrapper, started=False)
        a_button_was_pressed = False
        a_start_armed = False
        tele_data = tv_wrapper.get_tele_data()
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            tele_data = tv_wrapper.get_tele_data()
            a_button_pressed = bool(
                args.input_mode == "controller"
                and tele_data.motion_data_ready
                and tele_data.right_ctrl_aButton
            )
            if args.input_mode == "controller" and tele_data.motion_data_ready:
                if not a_button_pressed:
                    a_start_armed = True
                elif a_start_armed and not a_button_was_pressed:
                    START = True
                    logger_mp.info("Teleoperation start requested by controller A button.")
            a_button_was_pressed = a_button_pressed
            if xr_need_local_img and img_client is not None:
                image_frames = {}
                for camera_name in runtime_camera_names:
                    camera_cfg = camera_config.get(camera_name, {})
                    if camera_cfg.get('enable_zmq'):
                        image_frames[camera_name] = img_client.get_camera_frame(camera_name)
                if xr_quad_view:
                    tv_wrapper.render_to_xr(compose_xr_quad_view(image_frames, xr_camera_names, xr_display_shape))
                else:
                    head_img = image_frames.get('head_camera')
                    if head_img is not None and head_img.bgr is not None:
                        tv_wrapper.render_to_xr(head_img.bgr)

        # A may still be held after starting. Require a release and a new
        # rising edge before treating A as an exit request.
        a_button_was_pressed = bool(
            args.input_mode == "controller"
            and tele_data.motion_data_ready
            and tele_data.right_ctrl_aButton
        )
        if args.hand_eye_record and args.input_mode == "controller":
            HAND_EYE_CAPTURE.observe_button(
                bool(tele_data.motion_data_ready and tele_data.right_ctrl_bButton)
            )
            HAND_EYE_CAPTURE.consume_toggle()
        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        update_vr_hud(tv_wrapper, started=True, motion_ready=False)
        arm_ctrl.speed_gradual_max()

        image_frames = {camera_name: None for camera_name in record_camera_names}
        continuous_record = args.record and not args.hand_eye_record
        waiting_motion_log_count = 0
        arm_trace_last_log = 0.0
        arm_trace_start_q = None
        loco_last_warning_time = 0.0

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if img_client is not None:
                for camera_name in runtime_camera_names:
                    camera_cfg = camera_config.get(camera_name, {})
                    if camera_cfg.get('enable_zmq') and (continuous_record or xr_need_local_img):
                        image_frames[camera_name] = img_client.get_camera_frame(camera_name)
                if xr_need_local_img:
                    if xr_quad_view:
                        tv_wrapper.render_to_xr(compose_xr_quad_view(image_frames, xr_camera_names, xr_display_shape))
                    else:
                        head_img = image_frames.get('head_camera')
                        if head_img is not None and head_img.bgr is not None:
                            tv_wrapper.render_to_xr(head_img.bgr)

            # record mode
            if continuous_record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            external_arm_target = get_external_arm_target()
            if args.ee in ("dex3", "inspire_ftp", "inspire_dfx", "brainco")  and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "brainco" and args.input_mode == "controller":
                with left_gripper_trigger_in.get_lock():
                    left_gripper_trigger_in.value = tele_data.left_ctrl_triggerValue
                with left_gripper_squeeze_in.get_lock():
                    left_gripper_squeeze_in.value = tele_data.left_ctrl_squeezeValue
                with right_gripper_trigger_in.get_lock():
                    right_gripper_trigger_in.value = tele_data.right_ctrl_triggerValue
                with right_gripper_squeeze_in.get_lock():
                    right_gripper_squeeze_in.value = tele_data.right_ctrl_squeezeValue
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = tele_data.motion_data_ready
            if continuous_record:
                # Keep recorder readiness fresh even when XR motion is not valid.
                # The motion gate below may skip arm control, but the web console
                # still needs an accurate save/record-ready state.
                READY = recorder.is_ready()

            # Always read state before the motion gate so recording can continue
            # even if XR motion temporarily drops out. Control remains gated below.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            if arm_trace_start_q is None:
                arm_trace_start_q = current_lr_arm_q.copy()

            hand_eye_holding = False
            if args.hand_eye_record:
                replay_mode = HAND_EYE_REPLAY is not None
                if replay_mode and HAND_EYE_REPLAY.state == HAND_EYE_REPLAY.WAITING:
                    HAND_EYE_REPLAY.start(current_lr_arm_q)
                    logger_mp.info("Hand-eye deterministic replay started.")
                if (
                    replay_mode
                    and HAND_EYE_REPLAY.state == HAND_EYE_REPLAY.EVENT_HOLD
                    and HAND_EYE_CAPTURE.state == HAND_EYE_FOLLOW
                ):
                    try:
                        img_client.get_rgbd_frame(
                            "head_rgbd_camera",
                            request_bgr=False,
                        )
                        replay_hold_q = np.concatenate([left_fixed_q, current_lr_arm_q[-7:]])
                        HAND_EYE_CAPTURE.begin_hold(replay_hold_q)
                        logger_mp.info(
                            f"Hand-eye replay reached capture event at frame {HAND_EYE_REPLAY.index}; "
                            "RGB-D subscriber started."
                        )
                    except Exception as exc:
                        img_client.unsubscribe_rgbd(
                            "head_rgbd_camera",
                            request_bgr=False,
                        )
                        HAND_EYE_CAPTURE.fail(f"RGB-D subscription failed: {exc}")
                        HAND_EYE_REPLAY.abort(f"RGB-D subscription failed: {exc}")
                        logger_mp.error(f"Hand-eye replay RGB-D subscription failed: {exc}")

                if not replay_mode and args.input_mode == "controller":
                    b_button_fired = HAND_EYE_CAPTURE.observe_button(
                        tele_data.right_ctrl_bButton
                    )
                    if b_button_fired and not HAND_EYE_CAPTURE.follow_enabled:
                        HAND_EYE_CAPTURE.consume_toggle()
                        if not tele_data.motion_data_ready:
                            logger_mp.warning(
                                "Initial follow request ignored: XR motion data is not ready."
                            )
                        else:
                            initial_solution_q = np.concatenate([
                                left_fixed_q,
                                current_lr_arm_q[-7:],
                            ])
                            arm_ik.reset_solution(initial_solution_q)
                            HAND_EYE_CAPTURE.enable_follow()
                            logger_mp.info(
                                "Hand-eye absolute XR-to-IK follow enabled by first B press."
                            )
                if (
                    not replay_mode
                    and HAND_EYE_CAPTURE.follow_enabled
                    and HAND_EYE_CAPTURE.consume_toggle()
                ):
                    if HAND_EYE_CAPTURE.state == HAND_EYE_FOLLOW:
                        try:
                            hold_q, _ = arm_ik.solve_ik(
                                left_fixed_wrist_pose,
                                tele_data.right_wrist_pose,
                                current_lr_arm_q,
                                current_lr_arm_dq,
                            )
                            hold_q[:7] = left_fixed_q
                            if not np.all(np.isfinite(hold_q)):
                                raise RuntimeError(
                                    "capture target IK returned non-finite joints"
                                )
                            tv_wrapper.clear_depth_preview()
                            img_client.get_rgbd_frame(
                                "head_rgbd_camera",
                                request_bgr=False,
                            )
                            HAND_EYE_CAPTURE.begin_hold(hold_q)
                            if trajectory_recorder is not None:
                                trajectory_recorder.begin_capture_event()
                            logger_mp.info(
                                "Hand-eye capture: current absolute XR target latched; "
                                "moving to target and waiting for measured joints to settle; "
                                "RGB-D subscriber started."
                            )
                        except Exception as exc:
                            img_client.unsubscribe_rgbd(
                                "head_rgbd_camera",
                                request_bgr=False,
                            )
                            logger_mp.error(
                                f"Hand-eye capture target IK failed; continuing follow: {exc}"
                            )
                    elif HAND_EYE_CAPTURE.state == HAND_EYE_HOLD:
                        if HAND_EYE_CAPTURE.fatal_error:
                            logger_mp.error(
                                "Hand-eye absolute resume blocked after fatal hold safety error; "
                                "stop teleoperation."
                            )
                        elif not tele_data.motion_data_ready:
                            HAND_EYE_CAPTURE.fail("XR motion data is not ready; remaining in HOLD.")
                        else:
                            try:
                                hold_q = HAND_EYE_CAPTURE.hold_q
                                left_target = left_fixed_wrist_pose
                                right_target = tele_data.right_wrist_pose
                                arm_ik.reset_solution(hold_q)
                                resume_q, _ = arm_ik.solve_ik(
                                    left_target,
                                    right_target,
                                    hold_q,
                                    np.zeros_like(hold_q),
                                )
                                resume_q[:7] = left_fixed_q
                                resume_delta = float(np.max(np.abs(resume_q - hold_q)))
                                if not np.all(np.isfinite(resume_q)):
                                    raise RuntimeError("absolute resume IK returned non-finite joints")
                                hand_eye_resume_transition = {
                                    "start_q": current_lr_arm_q.copy(),
                                    "target_q": resume_q.copy(),
                                    "start_time": time.monotonic(),
                                    "duration": max(args.hand_eye_resume_seconds, 0.1),
                                }
                                HAND_EYE_CAPTURE.begin_resume()
                                tv_wrapper.clear_depth_preview()
                                logger_mp.info(
                                    "Hand-eye capture: smooth absolute resume started; "
                                    f"joint_delta={resume_delta:.5f} rad, "
                                    f"duration={hand_eye_resume_transition['duration']:.2f}s."
                                )
                            except Exception as exc:
                                HAND_EYE_CAPTURE.fail(f"Absolute resume failed: {exc}")
                                if trajectory_recorder is not None:
                                    trajectory_recorder.mark_capture_error(str(exc))
                                logger_mp.error(f"Hand-eye absolute resume failed: {exc}")
                    else:
                        logger_mp.warning(
                            f"Hand-eye toggle ignored while state={HAND_EYE_CAPTURE.state}."
                        )

                if HAND_EYE_CAPTURE.check_hold_drift(current_lr_arm_q):
                    img_client.unsubscribe_rgbd(
                        "head_rgbd_camera",
                        request_bgr=False,
                    )
                    hold_error = HAND_EYE_CAPTURE.snapshot().get("HAND_EYE_ERROR")
                    if trajectory_recorder is not None:
                        trajectory_recorder.mark_capture_error(hold_error or "hold drift")
                    if replay_mode:
                        HAND_EYE_REPLAY.abort(hold_error or "hold drift")
                    logger_mp.critical(f"Hand-eye hold safety violation: {hold_error}")

                if HAND_EYE_CAPTURE.state == HAND_EYE_SETTLING:
                    if HAND_EYE_CAPTURE.update_settling(current_lr_arm_q, current_lr_arm_dq):
                        hand_eye_recorder.begin_sample()
                        logger_mp.info("Hand-eye capture: joints settled; collecting RGB-D burst.")

                if HAND_EYE_CAPTURE.state == HAND_EYE_CAPTURING:
                    try:
                        rgbd = img_client.get_rgbd_frame(
                            "head_rgbd_camera",
                            request_bgr=False,
                        )
                        if (
                            rgbd
                            and rgbd.depth is not None
                            and rgbd.rgb_jpg is not None
                            and rgbd.metadata is not None
                        ):
                            joint_timestamp_ns = time.time_ns()
                            captured_frames = hand_eye_recorder.add_frame(
                                rgb_jpg=rgbd.rgb_jpg,
                                depth=rgbd.depth,
                                rgbd_metadata=rgbd.metadata,
                                right_arm_q=current_lr_arm_q[-7:],
                                joint_timestamp_ns=joint_timestamp_ns,
                                sample_timestamp_ns=time.time_ns(),
                            )
                            HAND_EYE_CAPTURE.set_capture_progress(captured_frames)
                            if captured_frames >= HAND_EYE_CAPTURE.burst_frames:
                                try:
                                    preview_stats = tv_wrapper.set_depth_preview(rgbd.depth)
                                    logger_mp.info(
                                        "Hand-eye depth preview displayed in VR: "
                                        f"valid={preview_stats['valid_ratio'] * 100:.1f}%, "
                                        f"range={preview_stats['min']}..{preview_stats['max']}."
                                    )
                                except Exception as preview_exc:
                                    logger_mp.error(
                                        f"Hand-eye depth preview failed: {preview_exc}"
                                    )
                                color_topic = hand_eye_camera_config.get("color_camera", "head_camera")
                                color_cfg = camera_config.get(color_topic, {})
                                hand_eye_recorder.submit_sample(
                                    {
                                        "robot": args.arm,
                                        "camera_stream": "head_rgbd_camera",
                                        "camera_serial": color_cfg.get("serial_number"),
                                        "right_arm_joint_order": [
                                            "right_shoulder_pitch",
                                            "right_shoulder_roll",
                                            "right_shoulder_yaw",
                                            "right_elbow",
                                            "right_wrist_roll",
                                            "right_wrist_pitch",
                                            "right_wrist_yaw",
                                        ],
                                    }
                                )
                                img_client.unsubscribe_rgbd(
                                    "head_rgbd_camera",
                                    request_bgr=False,
                                )
                                HAND_EYE_CAPTURE.begin_saving()
                                logger_mp.info(
                                    "Hand-eye capture: burst complete; RGB-D subscriber stopped; "
                                    "saving asynchronously."
                                )
                    except Exception as exc:
                        img_client.unsubscribe_rgbd(
                            "head_rgbd_camera",
                            request_bgr=False,
                        )
                        HAND_EYE_CAPTURE.fail(f"Capture failed: {exc}")
                        if trajectory_recorder is not None:
                            trajectory_recorder.mark_capture_error(str(exc))
                        if replay_mode:
                            HAND_EYE_REPLAY.abort(f"Capture failed: {exc}")
                        logger_mp.error(f"Hand-eye capture failed: {exc}")

                if HAND_EYE_CAPTURE.state == HAND_EYE_SAVING:
                    save_result = hand_eye_recorder.poll_result()
                    if save_result is not None:
                        if save_result.get("ok"):
                            HAND_EYE_CAPTURE.finish_saving(save_result["path"])
                            if trajectory_recorder is not None:
                                trajectory_recorder.mark_capture_saved(save_result["path"])
                            if replay_mode:
                                tv_wrapper.clear_depth_preview()
                                HAND_EYE_CAPTURE.resume_without_rebase()
                                HAND_EYE_REPLAY.resume_after_event()
                                logger_mp.info(
                                    f"Hand-eye replay resumed at trajectory frame {HAND_EYE_REPLAY.index}."
                                )
                            logger_mp.info(
                                f"Hand-eye sample saved: {save_result['path']} "
                                f"({save_result['frame_count']} frames)."
                            )
                        else:
                            HAND_EYE_CAPTURE.fail(save_result.get("error", "unknown save error"))
                            if trajectory_recorder is not None:
                                trajectory_recorder.mark_capture_error(
                                    save_result.get("error", "unknown save error")
                                )
                            if replay_mode:
                                HAND_EYE_REPLAY.abort(
                                    save_result.get("error", "unknown save error")
                                )
                            logger_mp.error(f"Hand-eye sample save failed: {save_result}")

                hand_eye_holding = HAND_EYE_CAPTURE.is_holding
                RECORD_RUNNING = HAND_EYE_CAPTURE.state in {
                    HAND_EYE_CAPTURING,
                    HAND_EYE_SAVING,
                }
                READY = (
                    HAND_EYE_CAPTURE.follow_enabled
                    and HAND_EYE_CAPTURE.state in {
                        HAND_EYE_FOLLOW,
                        HAND_EYE_HOLD,
                    }
                )

            update_vr_hud(
                tv_wrapper,
                started=True,
                motion_ready=tele_data.motion_data_ready,
            )
            control_input_ready = (
                tele_data.motion_data_ready
                or external_arm_target is not None
                or hand_eye_holding
                or HAND_EYE_REPLAY is not None
            )
            if (
                not tele_data.motion_data_ready
                and external_arm_target is None
                and not hand_eye_holding
                and HAND_EYE_REPLAY is None
            ):
                waiting_motion_log_count += 1
                if waiting_motion_log_count % max(1, int(args.frequency)) == 0:
                    logger_mp.warning(
                        "Waiting for valid XR motion data; skipping arm IK/control. "
                        f"input_mode={args.input_mode}, arm_reference_mode={args.arm_reference_mode}"
                    )
                if not RECORD_RUNNING:
                    time_elapsed = time.time() - start_time
                    time.sleep(max(0, (1 / args.frequency) - time_elapsed))
                    continue
            else:
                waiting_motion_log_count = 0
             
            # A toggles teleoperation only on a rising edge. This prevents the
            # same press used to start from immediately stopping the process.
            a_button_pressed = bool(
                args.input_mode == "controller"
                and tele_data.motion_data_ready
                and tele_data.right_ctrl_aButton
            )
            a_button_rising = a_button_pressed and not a_button_was_pressed
            a_button_was_pressed = a_button_pressed
            if a_button_rising:
                tv_wrapper.set_hud_status(
                    "正在结束遥操…",
                    "机器人将缓慢返回安全初始姿态",
                    "warning",
                )
                START = False
                STOP = True

            # high level locomotion control
            if args.input_mode == "controller" and args.motion and tele_data.motion_data_ready:
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    damp_code = loco_wrapper.Damp()
                    if damp_code not in (None, 0):
                        now = time.time()
                        if now - loco_last_warning_time >= 1.0:
                            logger_mp.warning(f"Loco {loco_wrapper.robot} Damp returned code={damp_code}")
                            loco_last_warning_time = now
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                if not args.hand_eye_record:
                    move_code = loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                                  -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                                  -tele_data.right_ctrl_thumbstickValue[0]* 0.3)
                    if move_code not in (None, 0):
                        now = time.time()
                        if now - loco_last_warning_time >= 1.0:
                            logger_mp.warning(f"Loco {loco_wrapper.robot} Move returned code={move_code}")
                            loco_last_warning_time = now

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            if hand_eye_resume_transition is not None:
                if HAND_EYE_CAPTURE.state != HAND_EYE_RESUMING:
                    raise RuntimeError(
                        "hand-eye resume transition exists outside RESUMING state"
                    )
                elapsed = (
                    time.monotonic()
                    - hand_eye_resume_transition["start_time"]
                )
                progress = min(
                    1.0,
                    elapsed / hand_eye_resume_transition["duration"],
                )
                alpha = smoothstep_progress(progress)
                transition_start_q = hand_eye_resume_transition["start_q"]
                transition_target_q = hand_eye_resume_transition["target_q"]
                sol_q = transition_start_q + (
                    transition_target_q - transition_start_q
                ) * alpha
                sol_tauff = arm_ik.gravity_torques(sol_q)
                if progress >= 1.0:
                    arm_ik.reset_solution(transition_target_q)
                    HAND_EYE_CAPTURE.finish_resume()
                    if trajectory_recorder is not None:
                        last_frame_index = trajectory_recorder.last_frame_index
                        next_frame_index = (
                            -1 if last_frame_index is None else last_frame_index
                        ) + 1
                        trajectory_recorder.finish_capture_event(next_frame_index)
                    hand_eye_resume_transition = None
                    logger_mp.info(
                        "Hand-eye capture: smooth absolute resume completed."
                    )
            elif (
                args.hand_eye_record
                and HAND_EYE_REPLAY is None
                and not HAND_EYE_CAPTURE.follow_enabled
            ):
                sol_q = hand_eye_start_q.copy()
                sol_tauff = arm_ik.gravity_torques(sol_q)
            elif hand_eye_holding:
                sol_q = HAND_EYE_CAPTURE.hold_q
                sol_tauff = np.zeros_like(sol_q)
            elif HAND_EYE_REPLAY is not None:
                if HAND_EYE_REPLAY.state == HAND_EYE_REPLAY.ERROR:
                    raise RuntimeError(HAND_EYE_REPLAY.error or "trajectory replay failed")
                HAND_EYE_REPLAY.check_tracking(current_lr_arm_q[-7:])
                sol_q = np.concatenate([
                    HAND_EYE_REPLAY.left_fixed_q,
                    HAND_EYE_REPLAY.target_right_q,
                ])
                sol_tauff = np.zeros_like(sol_q)
            elif not control_input_ready:
                sol_q = current_lr_arm_q.copy()
                sol_tauff = np.zeros_like(sol_q)
            elif external_arm_target is not None:
                sol_q = external_arm_target
                sol_tauff = np.zeros_like(sol_q)
            else:
                left_wrist_target = tele_data.left_wrist_pose
                right_wrist_target = tele_data.right_wrist_pose
                if args.hand_eye_record:
                    left_wrist_target = left_fixed_wrist_pose
                sol_q, sol_tauff = arm_ik.solve_ik(
                    left_wrist_target,
                    right_wrist_target,
                    current_lr_arm_q,
                    current_lr_arm_dq,
                )
            if args.hand_eye_record:
                sol_q[:7] = left_fixed_q
                # H2 arm_sdk needs gravity feedforward even while q is latched.
                # Zeroing tau_ff here lets a raised arm sag/fall before settling.
                sol_tauff = arm_ik.gravity_torques(sol_q)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            if control_input_ready:
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            if trajectory_recorder is not None:
                trajectory_recorder.add_frame(
                    right_command_q=sol_q[-7:],
                    right_measured_q=current_lr_arm_q[-7:],
                )
            if (
                HAND_EYE_REPLAY is not None
                and HAND_EYE_REPLAY.state == HAND_EYE_REPLAY.PLAYING
            ):
                HAND_EYE_REPLAY.advance()
            if ik_replay_pusher is not None and ik_replay_pusher.enabled:
                ik_replay_pusher.publish(build_ik_replay_live_payload(
                    robot=args.arm.lower(),
                    source="teleop",
                    current_lr_arm_q=current_lr_arm_q,
                    sol_q=sol_q,
                    extra={
                        "input_mode": args.input_mode,
                        "motion": args.motion,
                        "record_running": RECORD_RUNNING,
                    },
                ))
            if time.time() - arm_trace_last_log >= 1.0:
                target_error = float(np.max(np.abs(sol_q - current_lr_arm_q)))
                target_span = float(np.max(np.abs(sol_q - arm_trace_start_q)))
                state_span = float(np.max(np.abs(current_lr_arm_q - arm_trace_start_q)))
                logger_mp.info(
                    "Arm control trace: "
                    f"motion={args.motion}, input_mode={args.input_mode}, "
                    f"arm_reference_mode={args.arm_reference_mode}, "
                    f"target_error={target_error:.4f}, "
                    f"target_span={target_span:.4f}, state_span={state_span:.4f}"
                )
                arm_trace_last_log = time.time()

            # record data
            if continuous_record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6] if left_ee_active else []
                        right_ee_state = dual_hand_state_array[-6:] if right_ee_active else []
                        left_hand_action = dual_hand_action_array[:6] if left_ee_active else []
                        right_hand_action = dual_hand_action_array[-6:] if right_ee_active else []
                        current_body_state = []
                        current_body_action = []
                elif (args.ee == "brainco" and args.input_mode == "controller"):
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    # 只有正常相机模式才读取和保存图像
                    if not args.no_camera:
                        color_idx = 0
                        for camera_name in record_camera_names:
                            camera_cfg = camera_config[camera_name]
                            image = image_frames.get(camera_name)
                            if image is None or image.bgr is None:
                                logger_mp.warning(f"{camera_name} image is None!")
                                continue
                            if camera_name == 'head_camera' and camera_cfg.get('binocular'):
                                image_width = camera_cfg['image_shape'][1]
                                colors[f"color_{color_idx}"] = image.bgr[:, :image_width//2]
                                color_idx += 1
                                colors[f"color_{color_idx}"] = image.bgr[:, image_width//2:]
                                color_idx += 1
                            else:
                                colors[f"color_{color_idx}"] = image.bgr
                                color_idx += 1
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "type": left_ee,
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "type": right_ee,
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "type": left_ee,
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "type": right_ee,
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            loop_period = 1 / args.frequency
            if (
                HAND_EYE_REPLAY is not None
                and HAND_EYE_REPLAY.state == HAND_EYE_REPLAY.PLAYING
                and HAND_EYE_REPLAY.last_interval_seconds > 0
            ):
                loop_period = HAND_EYE_REPLAY.last_interval_seconds
            sleep_time = max(0, loop_period - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            if args.arm == "H2":
                if init_arm_q is not None:
                    h2_locked_targets = {} if args.motion else init_locked_joint_targets
                    move_h2_to_pose(
                        arm_ctrl,
                        init_arm_q,
                        h2_locked_targets,
                        args.exit_arm_pose_duration,
                        respect_stop=False,
                        gravity_torques=arm_ik.gravity_torques,
                    )
                else:
                    logger_mp.warning("Skip H2 ctrl_dual_arm_go_home because no init pose file is available.")
            else:
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to move arms to safe exit pose: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                if args.hand_eye_record:
                    img_client.unsubscribe_rgbd(
                        "head_rgbd_camera",
                        request_bgr=False,
                    )
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            if ik_replay_pusher is not None:
                ik_replay_pusher.close()
        except Exception as e:
            logger_mp.error(f"Failed to stop IK replay live pusher: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                status, result = motion_switcher.Exit_Debug_Mode()
                logger_mp.info(
                    f"Exit debug mode: {'Success' if status is not None else 'Failed'} "
                    f"status={status}, result={result}"
                )
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if recorder is not None:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        try:
            if hand_eye_recorder is not None:
                hand_eye_recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close hand-eye recorder: {e}")
        try:
            if trajectory_recorder is not None:
                trajectory_path = trajectory_recorder.close()
                logger_mp.info(f"Hand-eye trajectory finalized: {trajectory_path}")
        except Exception as e:
            logger_mp.error(f"Failed to close hand-eye trajectory recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
