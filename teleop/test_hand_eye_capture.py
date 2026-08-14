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
    SAVING,
    HandEyeCaptureState,
)
from teleop.utils.hand_eye_recorder import HandEyeRecorder


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


if __name__ == "__main__":
    unittest.main()
