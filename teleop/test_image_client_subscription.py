import logging
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


TELEIMAGER_SRC = Path(__file__).resolve().parent / "teleimager" / "src"
if str(TELEIMAGER_SRC) not in sys.path:
    sys.path.insert(0, str(TELEIMAGER_SRC))
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = Mock()
try:
    import logging_mp  # noqa: F401
except ModuleNotFoundError:
    sys.modules["logging_mp"] = logging

from teleimager import image_client
from teleimager.image_client import ImageClient, ZMQ_SubscriberManager


class _FakeSubscriber:
    def __init__(self):
        self.stop_count = 0
        self.recv_count = 0

    def recv(self):
        self.recv_count += 1
        return self

    def stop(self):
        self.stop_count += 1


class SubscriberLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.manager = object.__new__(ZMQ_SubscriberManager)
        self.manager._subscriber_threads = {}
        self.manager._lock = threading.Lock()
        self.manager._running = True
        self.created = []

        def create_subscriber(*args, **kwargs):
            subscriber = _FakeSubscriber()
            self.created.append(subscriber)
            return subscriber

        self.manager._create_subscriber_thread = create_subscriber

    def test_rgbd_subscriber_can_stop_and_restart(self):
        first = self.manager.subscribe_rgbd(
            "camera-host",
            55560,
            request_bgr=False,
        )
        again = self.manager.subscribe_rgbd(
            "camera-host",
            55560,
            request_bgr=False,
        )
        self.assertIs(first, again)
        self.assertEqual(len(self.created), 1)

        self.assertTrue(
            self.manager.unsubscribe_rgbd(
                "camera-host",
                55560,
                request_bgr=False,
            )
        )
        self.assertEqual(first.stop_count, 1)
        self.assertFalse(
            self.manager.unsubscribe_rgbd(
                "camera-host",
                55560,
                request_bgr=False,
            )
        )

        restarted = self.manager.subscribe_rgbd(
            "camera-host",
            55560,
            request_bgr=False,
        )
        self.assertIsNot(restarted, first)
        self.assertEqual(len(self.created), 2)

    def test_rgbd_unsubscribe_does_not_stop_other_decode_mode(self):
        raw_rgbd = self.manager.subscribe_rgbd(
            "camera-host",
            55560,
            request_bgr=False,
        )
        decoded_rgbd = self.manager.subscribe_rgbd(
            "camera-host",
            55560,
            request_bgr=True,
        )

        self.manager.unsubscribe_rgbd(
            "camera-host",
            55560,
            request_bgr=False,
        )

        self.assertEqual(raw_rgbd.stop_count, 1)
        self.assertEqual(decoded_rgbd.stop_count, 0)

    def test_image_client_can_skip_eager_jpeg_subscriptions(self):
        config = {
            "head_camera": {
                "enable_zmq": True,
                "enable_webrtc": True,
                "zmq_port": 55555,
                "data_format": "jpeg",
            },
            "head_rgbd_camera": {
                "enable_zmq": True,
                "zmq_port": 55560,
                "data_format": "rgbd",
            },
        }
        subscriber_manager = Mock()
        requester = Mock()
        requester.request.return_value = config

        with patch.object(
            image_client.ZMQ_SubscriberManager,
            "get_instance",
            return_value=subscriber_manager,
        ):
            with patch.object(image_client, "ZMQ_Requester", return_value=requester):
                ImageClient(
                    host="camera-host",
                    request_bgr=True,
                    auto_subscribe=False,
                )

        subscriber_manager.subscribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
