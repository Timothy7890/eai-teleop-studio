import json
import tarfile
import time
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from teleop_web.server import PROJECT_ROOT, TeleopManager, ValidationError, build_command, episode_progress, validate_device, validate_task


class ValidationTests(unittest.TestCase):
    def test_sample_configuration_builds_expected_command(self):
        device = validate_device({
            "arm": "H2", "ee": "inspire_dfx", "input_mode": "hand",
            "img_server_ip": "192.168.123.5", "network_interface": "eth0",
        })
        task = validate_task({"name": "h2_pico_inspire_dfx_test", "description": "测试采集"})
        with patch.dict("os.environ", {"XR_TELEOP_PYTHON": "python"}):
            command = build_command(device, task, Path("/home/user/xr_teleoperate/data/datasets"))
        self.assertIn("--arm=H2", command)
        self.assertIn("--ee=inspire_dfx", command)
        self.assertIn("--record", command)
        self.assertIn("--ipc", command)
        self.assertIn("--task-name=h2_pico_inspire_dfx_test", command)
        self.assertIn("--task-desc=测试采集", command)

    def test_task_name_rejects_path_traversal_and_spaces(self):
        for name in ("../escape", "pick cup", "中文名", ""):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                validate_task({"name": name, "description": "描述"})

    def test_invalid_webrtc_ip_is_rejected(self):
        with self.assertRaises(ValidationError):
            validate_device({"webrtc_server_ip": "999.1.1.1"})

    def test_invalid_image_server_ip_is_rejected(self):
        with self.assertRaises(ValidationError):
            validate_device({"img_server_ip": "999.1.1.1"})

    def test_default_xr_view_is_head_camera(self):
        device = validate_device({"input_mode": "controller", "ee": "none"})
        task = validate_task({"name": "head_view_test", "description": "默认头部相机"})
        command = build_command(device, task, Path("/tmp/datasets"))
        self.assertEqual(device["xr_view"], "head")
        self.assertIn("--xr-view=head", command)

    def test_xr_view_quad_builds_expected_command(self):
        device = validate_device({"input_mode": "controller", "ee": "none", "xr_view": "quad"})
        task = validate_task({"name": "quad_view_test", "description": "四路相机视角"})
        command = build_command(device, task, Path("/tmp/datasets"))
        self.assertIn("--xr-view=quad", command)

    def test_invalid_xr_view_is_rejected(self):
        with self.assertRaises(ValidationError):
            validate_device({"xr_view": "wrist_only"})

    def test_h2_init_pose_defaults_to_project_relative_path(self):
        device = validate_device({"arm": "H2", "input_mode": "controller", "ee": "none"})
        self.assertEqual(device["init_arm_pose_file"], "config/h2_pose_init.json")

    def test_project_relative_init_pose_expands_for_runtime_command(self):
        device = validate_device({
            "arm": "H2",
            "input_mode": "controller",
            "ee": "none",
            "init_arm_pose_file": "config/h2_pose_init.json",
        })
        task = validate_task({
            "name": "relative_pose",
            "instruction": "Move to init pose",
            "description": "鐩稿璺緞",
        })
        command = build_command(device, task, Path("/tmp/datasets"))
        self.assertIn(f"--init-arm-pose-file={PROJECT_ROOT / 'config' / 'h2_pose_init.json'}", command)

    def test_controller_mode_rejects_end_effector(self):
        with self.assertRaises(ValidationError):
            validate_device({"input_mode": "controller", "ee": "inspire_dfx"})

    def test_controller_mode_omits_end_effector_argument(self):
        device = validate_device({"input_mode": "controller", "ee": "none"})
        task = validate_task({"name": "arm_only", "description": "仅采集机械臂"})
        command = build_command(device, task, Path("/tmp/datasets"))
        self.assertFalse(any(argument.startswith("--ee=") for argument in command))

    def test_hand_tracking_without_adapter_omits_end_effector_argument(self):
        device = validate_device({"input_mode": "hand", "ee": "none"})
        task = validate_task({"name": "arm_only_hand_tracking", "description": "手部追踪仅控制机械臂"})
        command = build_command(device, task, Path("/tmp/datasets"))
        self.assertFalse(any(argument.startswith("--ee=") for argument in command))

    def test_asymmetric_passive_left_inspire_right_builds_side_arguments(self):
        device = validate_device({"input_mode": "hand", "left_ee": "rubber", "right_ee": "inspire_dfx"})
        task = validate_task({"name": "rubber_left_inspire_right", "description": "左橡胶手右因时手"})
        command = build_command(device, task, Path("/tmp/datasets"))
        self.assertIn("--left-ee=rubber", command)
        self.assertIn("--right-ee=inspire_dfx", command)
        self.assertFalse(any(argument.startswith("--ee=") for argument in command))

    def test_asymmetric_two_active_end_effectors_is_rejected(self):
        with self.assertRaises(ValidationError):
            validate_device({"input_mode": "hand", "left_ee": "dex3", "right_ee": "inspire_dfx"})

    def test_asymmetric_two_passive_end_effectors_is_allowed(self):
        device = validate_device({"input_mode": "controller", "left_ee": "none", "right_ee": "rubber"})
        self.assertEqual(device["left_ee"], "none")
        self.assertEqual(device["right_ee"], "rubber")

    def test_removed_passive_aliases_are_rejected(self):
        for value in ("passive", "rubber_passive"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                validate_device({"input_mode": "hand", "left_ee": value, "right_ee": "inspire_dfx"})

    @patch("teleop_web.server.IpcBridge")
    @patch("teleop_web.server.subprocess.Popen")
    def test_teleop_process_runs_from_entrypoint_directory(self, popen, bridge_class):
        process = MagicMock()
        process.poll.return_value = None
        process.pid = 123
        process.stdout = []
        popen.return_value = process
        bridge_class.return_value.state.return_value = {"online": False}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = TeleopManager(root / "datasets", root / "console.json")
            manager.save_device({
                "name": "H2", "arm": "H2", "ee": "none",
                "input_mode": "controller", "img_server_ip": "192.168.123.5",
                "network_interface": "eth0",
            })
            state = manager.create_task({
                "name": "arm_test", "description": "机械臂测试", "target_episodes": 1,
            })
            manager.start_task(state["tasks"][0]["id"])
        self.assertEqual(Path(popen.call_args.kwargs["cwd"]), Path(__file__).resolve().parents[1] / "teleop")

    def test_episode_progress_continues_after_highest_existing_id(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            (task_dir / "episode_0001").mkdir()
            (task_dir / "episode_0010").mkdir()
            (task_dir / "episode_0001" / "data.json").write_text('{"data": [\n]\n}', encoding="utf-8")
            (task_dir / "episode_0010" / "data.json").write_text('{"data": [\n]\n}', encoding="utf-8")
            (task_dir / "notes").mkdir()
            progress = episode_progress(task_dir)
        self.assertEqual(progress["existing_episodes"], 2)
        self.assertEqual(progress["last_episode"], "episode_0010")
        self.assertEqual(progress["next_episode"], "episode_0011")
        self.assertTrue(progress["resuming"])

    def test_archive_task_creates_dated_targz_and_persists_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = TeleopManager(root / "datasets", root / "console.json")
            manager.save_device({
                "name": "H2", "arm": "H2", "ee": "none",
                "input_mode": "controller", "img_server_ip": "192.168.123.5",
                "network_interface": "eth0",
            })
            state = manager.create_task({
                "instruction": "pick up the red cup",
                "name": "pick_red_cup", "description": "拾取红色水杯", "target_episodes": 10,
            })
            episode_dir = root / "datasets" / "pick_red_cup" / "episode_0001"
            episode_dir.mkdir(parents=True)
            (episode_dir / "data.json").write_text('{"data": [\n]\n}', encoding="utf-8")

            archived = manager.archive_task(state["tasks"][0]["id"])
            archive_path = Path(archived["archive_job"]["archive_path"])
            for _ in range(50):
                persisted = manager.state()["tasks"][0]
                if not persisted["archive_status"]["running"]:
                    break
                time.sleep(0.05)
            persisted = manager.state()["tasks"][0]
            with tarfile.open(archive_path, "r:gz") as archive:
                members = archive.getnames()

        self.assertRegex(archive_path.name, r"^pick_red_cup_\d{8}_\d{6}\.tar\.gz$")
        self.assertIn("pick_red_cup/episode_0001/data.json", members)
        self.assertEqual(persisted["last_archive"], str(archive_path))
        self.assertTrue(persisted["archived_at"])

    def test_preview_task_summarizes_episode_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = TeleopManager(root / "datasets", root / "console.json")
            manager.save_device({
                "name": "H2", "arm": "H2", "ee": "none",
                "input_mode": "controller", "img_server_ip": "192.168.123.5",
                "network_interface": "eth0",
            })
            state = manager.create_task({
                "name": "pick_red_cup", "description": "拾取红色水杯", "target_episodes": 10,
            })
            episode_dir = root / "datasets" / "pick_red_cup" / "episode_0001"
            (episode_dir / "colors").mkdir(parents=True)
            (episode_dir / "colors" / "000000_color_0.jpg").write_bytes(b"fake-jpeg")
            (episode_dir / "data.json").write_text(json.dumps({
                "data": [{
                    "idx": 0,
                    "colors": {"color_0": "colors/000000_color_0.jpg"},
                    "states": {
                        "left_ee": {"type": "rubber", "qpos": []},
                        "right_ee": {"type": "inspire_dfx", "qpos": [0.1]},
                    },
                    "actions": {"right_ee": {"qpos": [0.2]}},
                }]
            }, ensure_ascii=False), encoding="utf-8")

            preview = manager.preview_task(state["tasks"][0]["id"])

        self.assertEqual(preview["episode_total"], 1)
        self.assertEqual(preview["episodes"][0]["frame_count"], 1)
        self.assertEqual(preview["episodes"][0]["left_ee_type"], "rubber")
        self.assertEqual(preview["episodes"][0]["right_ee_type"], "inspire_dfx")
        self.assertEqual(preview["episodes"][0]["preview_images"][0]["name"], "color_0")

    def test_preview_task_uses_latest_color_images_when_json_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = TeleopManager(root / "datasets", root / "console.json")
            manager.save_device({
                "name": "H2", "arm": "H2", "ee": "none",
                "input_mode": "controller", "network_interface": "eth0",
            })
            state = manager.create_task({
                "name": "pick_red_cup", "description": "拾取红色水杯", "target_episodes": 10,
            })
            episode_dir = root / "datasets" / "pick_red_cup" / "episode_0002"
            color_dir = episode_dir / "colors"
            color_dir.mkdir(parents=True)
            for index in range(4):
                (color_dir / f"000000_color_{index}.jpg").write_bytes(b"old")
                (color_dir / f"000123_color_{index}.jpg").write_bytes(b"new")
            (episode_dir / "data.json").write_text('{"data": [', encoding="utf-8")

            preview = manager.preview_task(state["tasks"][0]["id"])

        self.assertEqual([image["name"] for image in preview["episodes"][0]["preview_images"]], [
            "color_0", "color_1", "color_2", "color_3",
        ])
        self.assertTrue(all("000123" in image["url"] for image in preview["episodes"][0]["preview_images"]))

    def test_camera_preview_reports_record_color_mapping(self):
        class FakeRequester:
            def __init__(self, *_args, **_kwargs):
                pass

            def request(self):
                return {
                    "head_camera": {"enable_zmq": True, "enable_webrtc": True, "webrtc_port": 60001, "zmq_port": 55555, "binocular": False},
                    "torso_camera": {"enable_zmq": True, "enable_webrtc": True, "webrtc_port": 60002, "zmq_port": 55556},
                    "left_wrist_camera": {"enable_zmq": True, "enable_webrtc": True, "webrtc_port": 60003, "zmq_port": 55557},
                    "right_wrist_camera": {"enable_zmq": True, "enable_webrtc": True, "webrtc_port": 60004, "zmq_port": 55558},
                }

            def close(self):
                pass

        fake_package = types.ModuleType("teleimager")
        fake_client = types.ModuleType("teleimager.image_client")
        fake_client.ZMQ_Requester = FakeRequester
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {
            "teleimager": fake_package,
            "teleimager.image_client": fake_client,
        }):
            root = Path(directory)
            manager = TeleopManager(root / "datasets", root / "console.json")
            manager.save_device({
                "name": "H2", "arm": "H2", "ee": "none",
                "input_mode": "controller", "network_interface": "eth0",
            })
            preview = manager.camera_preview()

        self.assertEqual([camera["record_colors"] for camera in preview["cameras"]], [
            ["color_0"], ["color_1"], ["color_2"], ["color_3"],
        ])
        self.assertEqual(preview["cameras"][0]["serial_number"], "CP0BB53000FS")
        self.assertEqual(preview["cameras"][0]["usb_interface"], "1.4")
        self.assertEqual(preview["cameras"][0]["video_index"], 0)

    def test_device_and_task_registry_is_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = TeleopManager(root / "datasets", root / "console.json")
            state = manager.save_device({
                "name": "H2_01", "arm": "H2", "ee": "none",
                "input_mode": "controller", "img_server_ip": "192.168.123.5",
                "network_interface": "eth0",
            })
            manager.create_task({
                "name": "pick_red_cup",
                "description": "拾取红色水杯", "target_episodes": 10,
            })
            manager.save_device({
                "name": "H2_01_updated", "arm": "H2", "ee": "none",
                "input_mode": "controller", "img_server_ip": "192.168.123.6",
                "network_interface": "eth0",
            })
            reloaded = TeleopManager(root / "datasets", root / "console.json").state()
            task_file_payload = (root / "datasets" / "tasks.json").read_text(encoding="utf-8")
            config_file_payload = (root / "console.json").read_text(encoding="utf-8")
        self.assertEqual(reloaded["device"]["name"], "H2_01_updated")
        self.assertEqual(reloaded["device"]["config"]["img_server_ip"], "192.168.123.6")
        self.assertEqual(reloaded["tasks"][0]["id"], 1)
        self.assertEqual(reloaded["tasks"][0]["target_episodes"], 10)
        self.assertEqual(reloaded["tasks"][0]["progress_percent"], 0)
        self.assertIn('"name": "pick_red_cup"', task_file_payload)
        self.assertIn('"completed_episodes": 0', task_file_payload)
        self.assertIn('"status": "未开始"', task_file_payload)
        self.assertNotIn('"tasks"', config_file_payload)

    def test_daily_log_file_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = TeleopManager(root / "datasets", root / "console.json", log_dir=root / "logs")
            manager.logger.write("info", "unit test log", command="python teleop_hand_and_arm.py --record")
            log_file = manager.logger._path_for_today()
            self.assertEqual(log_file.parent, root / "logs" / "system")
            content = log_file.read_text(encoding="utf-8")
        self.assertIn("unit test log", content)
        self.assertIn("python teleop_hand_and_arm.py --record", content)


if __name__ == "__main__":
    unittest.main()
