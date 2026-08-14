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
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    with EXTERNAL_ARM_TARGET_LOCK:
        external_target = dict(EXTERNAL_ARM_TARGET or {})
    external_age = time.time() - external_target.get("received_at", 0.0) if external_target else None
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
        "EXTERNAL_ARM_TARGET_ACTIVE": external_age is not None and external_age <= EXTERNAL_ARM_TARGET_TIMEOUT,
        "EXTERNAL_ARM_TARGET_SOURCE": external_target.get("source"),
    }


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

def move_dual_arm_to_pose(arm_ctrl, target_q, duration=5.0, respect_stop=True):
    logger_mp.info(f"Moving dual arms to init pose over {duration:.2f}s: {target_q}")
    arm_ctrl.speed_gradual_max(duration)
    tau = np.zeros_like(target_q)
    deadline = time.time() + max(duration, 0.1)
    while time.time() < deadline and (not respect_stop or not STOP):
        arm_ctrl.ctrl_dual_arm(target_q, tau)
        time.sleep(0.004)



def move_h2_to_pose(arm_ctrl, target_arm_q, locked_joint_targets=None, duration=5.0, respect_stop=True):
    locked_joint_targets = locked_joint_targets or {}
    if not locked_joint_targets:
        move_dual_arm_to_pose(arm_ctrl, target_arm_q, duration, respect_stop)
        return

    logger_mp.info(
        f"Moving H2 arms and locked body joints to init pose over {duration:.2f}s; "
        f"locked joints: {[joint.name for joint in locked_joint_targets]}"
    )
    arm_ctrl.speed_gradual_max(duration)
    tau = np.zeros_like(target_arm_q)
    start_time = time.time()
    duration = max(duration, 0.1)
    deadline = start_time + duration
    start_arm_q = arm_ctrl.get_current_dual_arm_q()
    current_motor_q = arm_ctrl.get_current_motor_q()
    start_locked_targets = {
        joint: float(current_motor_q[joint]) for joint in locked_joint_targets
    }

    while time.time() < deadline and (not respect_stop or not STOP):
        alpha = min(1.0, (time.time() - start_time) / duration)
        arm_q = start_arm_q + (target_arm_q - start_arm_q) * alpha
        body_q = {
            joint: start_locked_targets[joint] + (target_q - start_locked_targets[joint]) * alpha
            for joint, target_q in locked_joint_targets.items()
        }
        arm_ctrl.set_locked_joint_targets(body_q)
        arm_ctrl.ctrl_dual_arm(arm_q, tau)
        time.sleep(0.004)

    arm_ctrl.set_locked_joint_targets(locked_joint_targets)
    arm_ctrl.ctrl_dual_arm(target_arm_q, tau)

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
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
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
    logger_mp.debug(f"args: {args}")

    # 先设置为空，避免无相机模式退出时找不到该变量
    img_client = None
    loco_wrapper = None
    motion_switcher = None
    init_arm_q = None
    init_locked_joint_targets = {}
    ik_replay_pusher = None

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
            img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
            camera_config = img_client.get_cam_config()
            logger_mp.debug(f"Camera config: {camera_config}")
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
        runtime_camera_names = ordered_unique(record_camera_names + (xr_camera_names if xr_quad_view else ['head_camera']))
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
                move_h2_to_pose(arm_ctrl, init_arm_q, h2_locked_targets, args.init_arm_pose_duration)
            else:
                move_dual_arm_to_pose(arm_ctrl, init_arm_q, args.init_arm_pose_duration)

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
        if args.record:
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
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
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

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        arm_ctrl.speed_gradual_max()

        image_frames = {camera_name: None for camera_name in record_camera_names}
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
                    if camera_cfg.get('enable_zmq') and (args.record or xr_need_local_img):
                        image_frames[camera_name] = img_client.get_camera_frame(camera_name)
                if xr_need_local_img:
                    if xr_quad_view:
                        tv_wrapper.render_to_xr(compose_xr_quad_view(image_frames, xr_camera_names, xr_display_shape))
                    else:
                        head_img = image_frames.get('head_camera')
                        if head_img is not None and head_img.bgr is not None:
                            tv_wrapper.render_to_xr(head_img.bgr)

            # record mode
            if args.record and RECORD_TOGGLE:
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
            if args.record:
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

            control_input_ready = tele_data.motion_data_ready or external_arm_target is not None
            if not tele_data.motion_data_ready and external_arm_target is None:
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
             
            # high level control
            if args.input_mode == "controller" and args.motion and tele_data.motion_data_ready:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    damp_code = loco_wrapper.Damp()
                    if damp_code not in (None, 0):
                        now = time.time()
                        if now - loco_last_warning_time >= 1.0:
                            logger_mp.warning(f"Loco {loco_wrapper.robot} Damp returned code={damp_code}")
                            loco_last_warning_time = now
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
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
            if not control_input_ready:
                sol_q = current_lr_arm_q.copy()
                sol_tauff = np.zeros_like(sol_q)
            elif external_arm_target is not None:
                sol_q = external_arm_target
                sol_tauff = np.zeros_like(sol_q)
            else:
                sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            if control_input_ready:
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
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
            if args.record:
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
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
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
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
