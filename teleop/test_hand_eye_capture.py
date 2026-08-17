import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from teleop.utils.hand_eye_capture import (
    CAPTURING,
    FOLLOW,
    HOLD,
    RESUMING,
    SAVING,
    HandEyeCaptureState,
    build_hand_eye_hud_status,
)
from teleop.utils.hand_eye_recorder import HandEyeRecorder
from teleop.utils.hand_eye_trajectory import (
    HandEyeTrajectoryRecorder,
    HandEyeTrajectoryReplay,
)


def pose(x=0.0, y=0.0, z=0.0):
    result = np.eye(4)
    result[:3, 3] = [x, y, z]
    return result


class HandEyeCaptureStateTest(unittest.TestCase):
    def test_controller_button_uses_rising_edge(self):
        state = HandEyeCaptureState()
        self.assertTrue(state.observe_button(True))
        self.assertTrue(state.consume_toggle())
        self.assertFalse(state.observe_button(True))
        self.assertFalse(state.consume_toggle())
        self.assertFalse(state.observe_button(False))
        self.assertTrue(state.observe_button(True))
        self.assertTrue(state.consume_toggle())

    def test_settling_requires_time_speed_and_span(self):
        state = HandEyeCaptureState(
            settle_seconds=0.5,
            max_joint_speed=0.02,
            max_joint_span=0.003,
        )
        q = np.zeros(14)
        state.begin_hold(q, now=0.0)
        self.assertFalse(state.update_settling(q, np.ones(14) * 0.03, now=0.1))
        self.assertFalse(state.update_settling(q, np.zeros(14), now=0.2))
        self.assertTrue(state.update_settling(q, np.zeros(14), now=0.7))
        self.assertEqual(state.state, CAPTURING)

    def test_hold_drift_is_fatal_and_blocks_capture(self):
        state = HandEyeCaptureState(max_hold_error=0.05)
        state.begin_hold(np.zeros(14), now=0.0)
        drifted_q = np.zeros(14)
        drifted_q[-1] = 0.051
        self.assertTrue(state.check_hold_drift(drifted_q))
        self.assertEqual(state.state, HOLD)
        self.assertTrue(state.fatal_error)
        self.assertIn("Hold drift", state.snapshot()["HAND_EYE_ERROR"])
        self.assertFalse(state.update_settling(drifted_q, np.zeros(14), now=1.0))

    def test_left_arm_motion_does_not_block_right_arm_settling(self):
        state = HandEyeCaptureState(
            settle_seconds=0.5,
            max_joint_speed=0.05,
            max_joint_span=0.003,
        )
        state.begin_hold(np.zeros(14), now=0.0)
        left_only_q = np.concatenate([np.ones(7), np.zeros(7)])
        left_only_dq = np.concatenate([np.ones(7), np.zeros(7)])
        self.assertFalse(state.check_hold_drift(left_only_q))
        self.assertFalse(state.update_settling(left_only_q, left_only_dq, now=0.1))
        self.assertTrue(state.update_settling(left_only_q, left_only_dq, now=0.6))
        self.assertEqual(state.state, CAPTURING)

    def test_rebase_first_frame_is_robot_anchor(self):
        state = HandEyeCaptureState()
        state.begin_hold(np.zeros(14), now=0.0)
        state.update_settling(np.zeros(14), np.zeros(14), now=0.1)
        state.update_settling(np.zeros(14), np.zeros(14), now=0.6)
        state.begin_saving()
        self.assertEqual(state.state, SAVING)
        state.finish_saving("episode_0001")
        self.assertEqual(state.state, HOLD)

        left_xr = pose(1.0, 2.0, 3.0)
        right_xr = pose(-1.0, 2.0, 3.0)
        left_robot = pose(0.2, 0.3, 0.4)
        right_robot = pose(0.2, -0.3, 0.4)
        anchors = state.preview_rebase(left_xr, right_xr, left_robot, right_robot)
        first_left, first_right = anchors.apply(left_xr, right_xr)
        np.testing.assert_allclose(first_left, left_robot)
        np.testing.assert_allclose(first_right, right_robot)

        state.commit_rebase(anchors)
        self.assertEqual(state.state, FOLLOW)
        moved_left, _ = state.apply_rebase(pose(1.1, 2.0, 3.0), right_xr)
        np.testing.assert_allclose(moved_left[:3, 3], [0.3, 0.3, 0.4])

    def test_initial_rebase_starts_from_current_robot_pose(self):
        state = HandEyeCaptureState()
        self.assertFalse(state.follow_enabled)
        left_xr = pose(1.0, 2.0, 3.0)
        right_xr = pose(-1.0, 2.0, 3.0)
        left_robot = pose(0.2, 0.3, 0.4)
        right_robot = pose(0.2, -0.3, 0.4)
        state.initialize_rebase(left_xr, right_xr, left_robot, right_robot)
        state.enable_follow()
        self.assertTrue(state.follow_enabled)
        first_left, first_right = state.apply_rebase(left_xr, right_xr)
        np.testing.assert_allclose(first_left, left_robot)
        np.testing.assert_allclose(first_right, right_robot)

    def test_smooth_resume_state_ignores_intentional_hold_departure(self):
        state = HandEyeCaptureState(max_hold_error=0.05)
        state.begin_hold(np.zeros(14), now=0.0)
        state.fail("sample ready")
        state.begin_resume()
        self.assertEqual(state.state, RESUMING)
        self.assertFalse(state.check_hold_drift(np.ones(14)))
        state.finish_resume()
        self.assertEqual(state.state, FOLLOW)

    def test_hud_status_tracks_capture_lifecycle(self):
        waiting = build_hand_eye_hud_status({}, started=False)
        self.assertEqual(waiting[0], "等待开始遥操")
        self.assertIn("按 A", waiting[1])

        waiting_follow = build_hand_eye_hud_status(
            {
                "HAND_EYE_STATE": FOLLOW,
                "HAND_EYE_FOLLOW_ENABLED": False,
            },
            started=True,
        )
        self.assertEqual(waiting_follow[0], "固定姿态已就绪")
        self.assertIn("B：开始绝对位姿跟随", waiting_follow[1])

        following = build_hand_eye_hud_status(
            {
                "HAND_EYE_STATE": FOLLOW,
                "HAND_EYE_SAVED_SAMPLES": 2,
            },
            started=True,
        )
        self.assertIn("已保存 2 条", following[0])
        self.assertIn("B：锁定并采样", following[1])

        capturing = build_hand_eye_hud_status(
            {
                "HAND_EYE_STATE": CAPTURING,
                "HAND_EYE_CAPTURED_FRAMES": 3,
                "HAND_EYE_BURST_FRAMES": 5,
            },
            started=True,
        )
        self.assertIn("3 / 5", capturing[0])

        hold = build_hand_eye_hud_status(
            {
                "HAND_EYE_STATE": HOLD,
                "HAND_EYE_SAVED_SAMPLES": 3,
            },
            started=True,
        )
        self.assertIn("第 3 条已保存", hold[0])
        self.assertIn("恢复绝对位姿跟随", hold[1])

        resuming = build_hand_eye_hud_status(
            {
                "HAND_EYE_STATE": RESUMING,
                "HAND_EYE_FOLLOW_ENABLED": True,
            },
            started=True,
        )
        self.assertIn("追赶手柄位置", resuming[0])


class HandEyeRecorderTest(unittest.TestCase):
    def test_preserves_jpeg_bytes_and_uint16_depth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = HandEyeRecorder(temp_dir)
            recorder.begin_sample()
            rgb_jpg = b"\xff\xd8hand-eye-test\xff\xd9"
            depth = np.arange(12, dtype=np.uint16).reshape(3, 4)
            metadata = {
                "frame_id": 42,
                "timestamp_ns": 123,
                "stream": "head_rgbd_camera",
                "color_shape": [1080, 1920],
                "depth_shape": [3, 4],
                "depth_dtype": "uint16",
            }
            self.assertEqual(
                recorder.add_frame(
                    rgb_jpg=rgb_jpg,
                    depth=depth,
                    rgbd_metadata=metadata,
                    right_arm_q=np.arange(7, dtype=float),
                    joint_timestamp_ns=124,
                    sample_timestamp_ns=125,
                ),
                1,
            )
            # Duplicate RGB-D frame IDs must not create duplicate samples.
            self.assertEqual(
                recorder.add_frame(
                    rgb_jpg=rgb_jpg,
                    depth=depth,
                    rgbd_metadata=metadata,
                    right_arm_q=np.arange(7, dtype=float),
                    joint_timestamp_ns=126,
                    sample_timestamp_ns=127,
                ),
                1,
            )
            recorder.submit_sample({"camera_serial": "TEST"})
            deadline = time.monotonic() + 3.0
            result = None
            while result is None and time.monotonic() < deadline:
                result = recorder.poll_result()
                time.sleep(0.01)
            recorder.close()
            self.assertIsNotNone(result)
            self.assertTrue(result["ok"])

            episode = Path(result["path"])
            payload = json.loads((episode / "data.json").read_text(encoding="utf-8"))
            frame = payload["data"][0]
            self.assertEqual((episode / frame["colors"]["head_rgb"]).read_bytes(), rgb_jpg)
            saved_depth = np.load(episode / frame["depths"]["head_depth"], allow_pickle=False)
            np.testing.assert_array_equal(saved_depth, depth)
            self.assertEqual(frame["states"]["right_arm"]["qpos"], list(np.arange(7, dtype=float)))
            self.assertEqual(frame["timestamps"]["rgbd_timestamp_ns"], 123)


class HandEyeTrajectoryTest(unittest.TestCase):
    def test_records_exact_frames_events_and_replays_sequentially(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            left_fixed = np.arange(7, dtype=float) * 0.01
            recorder = HandEyeTrajectoryRecorder(
                temp_dir,
                left_fixed_q=left_fixed,
                frequency=30,
            )
            base_time = time.monotonic_ns()
            commands = [
                np.zeros(7),
                np.ones(7) * 0.1,
                np.ones(7) * 0.2,
            ]
            measured = [
                np.zeros(7),
                np.ones(7) * 0.09,
                np.ones(7) * 0.19,
            ]
            recorder.add_frame(
                right_command_q=commands[0],
                right_measured_q=measured[0],
                monotonic_ns=base_time,
            )
            recorder.begin_capture_event(frame_index=0)
            recorder.add_frame(
                right_command_q=commands[1],
                right_measured_q=measured[1],
                monotonic_ns=base_time + 40_000_000,
            )
            recorder.mark_capture_saved("episode_0001", frame_index=1)
            recorder.finish_capture_event(frame_index=2)
            recorder.add_frame(
                right_command_q=commands[2],
                right_measured_q=measured[2],
                monotonic_ns=base_time + 80_000_000,
            )
            trajectory_dir = recorder.close()

            with np.load(trajectory_dir / "trajectory.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["right_command_q"], commands)
                np.testing.assert_array_equal(data["right_measured_q"], measured)
                np.testing.assert_array_equal(data["left_fixed_q"], left_fixed)
            events = json.loads((trajectory_dir / "events.json").read_text(encoding="utf-8"))
            self.assertTrue(events["completed"])
            self.assertEqual(events["events"][0]["capture_frame_index"], 1)
            self.assertEqual(events["events"][0]["resume_frame_index"], 2)

            replay = HandEyeTrajectoryReplay(trajectory_dir)
            replay.start(np.concatenate([left_fixed, measured[0]]))
            np.testing.assert_array_equal(replay.target_right_q, commands[0])
            self.assertIsNone(replay.advance())
            np.testing.assert_array_equal(replay.target_right_q, commands[1])
            event = replay.advance()
            self.assertEqual(event["episode"], "episode_0001")
            self.assertEqual(replay.state, replay.EVENT_HOLD)
            replay.resume_after_event()
            np.testing.assert_array_equal(replay.target_right_q, commands[2])
            replay.advance()
            self.assertEqual(replay.state, replay.COMPLETED)

    def test_replay_rejects_unmatched_start_pose(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = HandEyeTrajectoryRecorder(
                temp_dir,
                left_fixed_q=np.zeros(7),
                frequency=30,
            )
            recorder.add_frame(
                right_command_q=np.zeros(7),
                right_measured_q=np.zeros(7),
            )
            recorder.begin_capture_event(frame_index=0)
            recorder.mark_capture_saved("episode_0001", frame_index=0)
            recorder.finish_capture_event(frame_index=0)
            trajectory_dir = recorder.close()
            replay = HandEyeTrajectoryReplay(trajectory_dir, start_tolerance=0.01)
            with self.assertRaises(RuntimeError):
                replay.start(np.ones(14))
            self.assertEqual(replay.state, replay.ERROR)

    def test_replay_aborts_after_persistent_tracking_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = HandEyeTrajectoryRecorder(
                temp_dir,
                left_fixed_q=np.zeros(7),
                frequency=30,
            )
            for index in range(3):
                recorder.add_frame(
                    right_command_q=np.zeros(7),
                    right_measured_q=np.zeros(7),
                    monotonic_ns=time.monotonic_ns() + index * 30_000_000,
                )
            recorder.begin_capture_event(frame_index=2)
            recorder.mark_capture_saved("episode_0001", frame_index=2)
            recorder.finish_capture_event(frame_index=2)
            trajectory_dir = recorder.close()
            replay = HandEyeTrajectoryReplay(
                trajectory_dir,
                tracking_error_limit=0.05,
                tracking_error_frames=2,
            )
            replay.start(np.zeros(14))
            replay.check_tracking(np.ones(7))
            with self.assertRaises(RuntimeError):
                replay.check_tracking(np.ones(7))
            self.assertEqual(replay.state, replay.ERROR)

    def test_replay_rejects_trajectory_without_capture_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = HandEyeTrajectoryRecorder(
                temp_dir,
                left_fixed_q=np.zeros(7),
                frequency=30,
            )
            recorder.add_frame(
                right_command_q=np.zeros(7),
                right_measured_q=np.zeros(7),
            )
            trajectory_dir = recorder.close()
            with self.assertRaisesRegex(ValueError, "no capture events"):
                HandEyeTrajectoryReplay(trajectory_dir)


if __name__ == "__main__":
    unittest.main()
