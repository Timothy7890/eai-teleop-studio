"""Minimal XR controller probe.

Serves a bare vuer scene with ONLY a MotionControllers subscription:
no HUD, no depth preview, no WebRTC video, no robot control.

Usage (robot does not move; the teleop program does not need to run):

    /home/robot/miniconda3/envs/teleop/bin/python teleop/xr_controller_probe.py

Then open the same URL form as teleop:

    https://<host-ip>:8013/?ws=wss://<host-ip>:8013

Direct port access without `?ws=` is not a valid Vuer session. The server
now redirects `/` to that form automatically. Enter VR and press any
controller button. The terminal prints per-second event counts:

    camera>0, controller>0  -> XR link fully works; the teleop scene is at fault
    camera>0, controller=0  -> headset sends head pose but no controllers
                               (hand-tracking mode / controllers asleep)
    camera=0                -> not in the immersive session at all
"""

import asyncio
import sys
import time
from pathlib import Path

from vuer import Vuer
from vuer.schemas import Hands, MotionControllers

# The teleop/ script dir contains a "televuer" project folder that shadows the
# installed package; point directly at its src layout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "televuer" / "src"))
from televuer.televuer import _install_asset_cache, _install_ws_query_redirect

CERT = str(Path(__file__).resolve().parent.parent / "config" / "cert.pem")
KEY = str(Path(__file__).resolve().parent.parent / "config" / "key.pem")

app = Vuer(host="0.0.0.0", port=8013, cert=CERT, key=KEY, queries=dict(grid=True))
# Same gzip + immutable-cache asset serving as the teleop server; without it
# the multi-MB JS bundle rarely survives the flaky headset WiFi link.
_install_ws_query_redirect(app.app)
_install_asset_cache(app.app, app.client_root)


from aiohttp import web as _web


@_web.middleware
async def _request_logger(request, handler):
    start = time.monotonic()
    try:
        response = await handler(request)
        status = getattr(response, "status", "?")
        return response
    except Exception as exc:
        status = f"EXC:{type(exc).__name__}"
        raise
    finally:
        print(
            f"[probe] {request.remote} {request.method} {request.path} "
            f"-> {status} ({(time.monotonic() - start) * 1000:.0f}ms)",
            flush=True,
        )


app.app.middlewares.insert(0, _request_logger)

counts = {"camera": 0, "controller": 0, "hand": 0}
first_seen = set()


def _note(kind, event):
    counts[kind] += 1
    if kind not in first_seen:
        first_seen.add(kind)
        value = getattr(event, "value", {})
        keys = list(value.keys()) if isinstance(value, dict) else type(value)
        print(f"[probe] first {kind} event, keys={keys}", flush=True)


@app.add_handler("CAMERA_MOVE")
async def on_cam(event, session):
    _note("camera", event)


@app.add_handler("CONTROLLER_MOVE")
async def on_ctrl(event, session):
    _note("controller", event)


@app.add_handler("HAND_MOVE")
async def on_hand(event, session):
    _note("hand", event)


@app.spawn(start=True)
async def main(session):
    session.upsert(
        MotionControllers(
            stream=True,
            key="motionControllers",
            left=True,
            right=True,
        ),
        to="bgChildren",
    )
    session.upsert(
        Hands(stream=True, key="hands", hideLeft=True, hideRight=True),
        to="bgChildren",
    )
    print("[probe] session started; MotionControllers + Hands subscribed.", flush=True)
    prev = dict(counts)
    while True:
        await asyncio.sleep(5.0)
        delta = {k: counts[k] - prev[k] for k in counts}
        prev = dict(counts)
        print(
            f"[probe] {time.strftime('%H:%M:%S')} last 5s: "
            f"camera={delta['camera']}, controller={delta['controller']}, "
            f"hand={delta['hand']} (totals {counts})",
            flush=True,
        )
