# for motion switcher
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
import time

# MotionSwitcher used to switch mode between debug mode and ai mode
class MotionSwitcher:
    def __init__(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(1.0)
        self.msc.Init()

    def Enter_Debug_Mode(self):
        try:
            status, result = self.msc.CheckMode()
            while result['name']:
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1)
            return status, result
        except Exception as e:
            return None, None
    
    def Exit_Debug_Mode(self):
        try:
            status, result = self.msc.SelectMode(nameOrAlias='ai')
            return status, result
        except Exception as e:
            return None, None

def _load_loco_client(arm="G1"):
    arm_name = (arm or "G1").upper()
    if arm_name == "H2":
        try:
            from unitree_sdk2py.h2.loco.h2_loco_client import LocoClient
        except ImportError as exc:
            raise RuntimeError(
                "H2 motion mode requires unitree_sdk2py.h2.loco.h2_loco_client."
            ) from exc
        return "H2", LocoClient

    # Keep the original behavior for non-H2 arms: use the G1 loco client.
    try:
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    except ImportError as exc:
        raise RuntimeError(
            "G1 motion mode requires unitree_sdk2py.g1.loco.g1_loco_client."
        ) from exc
    return "G1", LocoClient


class LocoClientWrapper:
    def __init__(self, arm="G1", command_timeout=0.05, status_timeout=1.0):
        self.robot, loco_client_cls = _load_loco_client(arm)
        self.command_timeout = command_timeout
        self.status_timeout = status_timeout
        self.client = loco_client_cls()
        self.client.SetTimeout(self.status_timeout)
        self.client.Init()

    def _call_code(self, method_name, *args, timeout=None):
        if not hasattr(self.client, method_name):
            return None
        self.client.SetTimeout(timeout if timeout is not None else self.command_timeout)
        result = getattr(self.client, method_name)(*args)
        if isinstance(result, tuple):
            return result[0] if len(result) > 0 else None
        return result

    def _call_reply(self, method_name, timeout=None):
        if not hasattr(self.client, method_name):
            return None, None
        self.client.SetTimeout(timeout if timeout is not None else self.status_timeout)
        result = getattr(self.client, method_name)()
        if isinstance(result, tuple):
            if len(result) == 0:
                return None, None
            if len(result) == 1:
                return result[0], None
            return result[0], result[1]
        return 0, result

    def GetStatusSummary(self):
        fsm_id_code, fsm_id = self._call_reply("GetFsmId")
        fsm_mode_code, fsm_mode = self._call_reply("GetFsmMode")
        summary = {
            "robot": self.robot,
            "fsm_id_code": fsm_id_code,
            "fsm_id": fsm_id,
            "fsm_mode_code": fsm_mode_code,
            "fsm_mode": fsm_mode,
        }

        if self.robot == "H2":
            arm_sdk_code, arm_sdk_enabled = self._call_reply("GetArmSdkStatus")
            summary["arm_sdk_code"] = arm_sdk_code
            summary["arm_sdk_enabled"] = arm_sdk_enabled

        return summary

    def EnsureArmSDKEnabled(self):
        if self.robot != "H2":
            return True, {"robot": self.robot, "skipped": True}

        code, enabled = self._call_reply("GetArmSdkStatus")
        info = {
            "initial_code": code,
            "initial_enabled": enabled,
            "enable_code": None,
            "final_code": code,
            "final_enabled": enabled,
        }
        if code != 0:
            return False, info
        if enabled:
            info["action"] = "already_enabled"
            return True, info

        info["enable_code"] = self._call_code(
            "SetArmSdkStatus", True, timeout=self.status_timeout
        )
        if info["enable_code"] != 0:
            return False, info

        for _ in range(10):
            time.sleep(0.2)
            final_code, final_enabled = self._call_reply("GetArmSdkStatus")
            info["final_code"] = final_code
            info["final_enabled"] = final_enabled
            if final_code == 0 and final_enabled:
                info["action"] = "enabled"
                return True, info

        info["action"] = "enable_timeout"
        return False, info

    def Enter_Damp_Mode(self):
        return self.Damp()

    def Damp(self):
        # FSM 1 is Damp on both G1 and H2 loco APIs.
        code = self._call_code("SetFsmId", 1, timeout=self.command_timeout)
        if code is not None:
            return code
        return self._call_code("Damp", timeout=self.command_timeout)

    def Move(self, vx, vy, vyaw):
        # Prefer SetVelocity because it returns the service code on H2/G1.
        code = self._call_code(
            "SetVelocity",
            float(vx),
            float(vy),
            float(vyaw),
            1.0,
            timeout=self.command_timeout,
        )
        if code is not None:
            return code
        return self._call_code(
            "Move",
            float(vx),
            float(vy),
            float(vyaw),
            False,
            timeout=self.command_timeout,
        )

if __name__ == '__main__':
    ChannelFactoryInitialize(1) # 0 for real robot, 1 for simulation
    ms = MotionSwitcher()
    status, result = ms.Enter_Debug_Mode()
    print("Enter debug mode:", status, result)
    time.sleep(5)
    status, result = ms.Exit_Debug_Mode()
    print("Exit debug mode:", status, result)
    time.sleep(2)
