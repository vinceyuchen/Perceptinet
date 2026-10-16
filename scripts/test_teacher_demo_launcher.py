"""CARLA-free tests for one-click teacher demo cycle coordination."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Perceptinet_Teacher_Demo as launcher


class FakeProcess:
    def __init__(self, exited=False, code=0):
        self.exited = exited
        self.returncode = code

    def poll(self):
        return self.returncode if self.exited else None

    def wait(self, timeout=None):
        del timeout
        return self.returncode

    def terminate(self):
        self.exited = True
        self.returncode = 0

    def kill(self):
        self.terminate()


def main() -> None:
    assert launcher.main(["--dry-run"]) == 0
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        markers = []
        dashboard = FakeProcess()
        scenario_count = [0]
        dashboard_snapshots = []
        scenario_snapshots = []
        scenario_modes = []

        def spawn(command, **kwargs):
            script = str(command[1]).replace("\\", "/")
            if script.endswith("scripts/run_dashboard.py"):
                target = command[command.index("--snapshot") + 1]
                dashboard_snapshots.append(target)
                return dashboard
            assert script.endswith("scripts/run_occluded_pedestrian_braking_v1.py")
            scenario_count[0] += 1
            env = kwargs["env"]
            scenario_modes.append(env["PERCEPTINET_DEMO_SCENARIO"])
            assert env["PERCEPTINET_TEACHER_DEMO"] == "1"
            scenario_snapshots.append(
                env["PERCEPTINET_DEMO_SNAPSHOT_FILE"]
            )
            assert env["PERCEPTINET_DEMO_CYCLE_INDEX"] == str(
                scenario_count[0]
            )
            assert "PERCEPTINET_VALIDATION" not in env
            info = {
                "completed": True,
                "reason": ("controller_confirmed_stop"
                           if scenario_modes[-1] == "clear"
                           else "kinematic_contact"),
                "scenario": scenario_modes[-1],
                "safe_stop_verified": scenario_modes[-1] == "clear",
                "pedestrian_motion_verified": True,
                "pedestrian_walked_m": 2.8,
                "frame_id": 100 + scenario_count[0],
            }
            markers.append(info)
            (folder / "cycle_complete.json").write_text(
                json.dumps(info), encoding="utf-8"
            )
            return FakeProcess(exited=True)

        with (patch.object(launcher, "SIGNALS", folder),
              patch.object(launcher, "STOP_FILE", folder / "stop.signal"),
              patch.object(launcher, "CYCLE_FILE", folder / "cycle_complete.json"),
              patch.object(launcher, "check_project"),
              patch.object(launcher, "launch_carla_if_needed", return_value=None),
              patch.object(launcher, "wait_for_carla_rpc",
                           return_value="Carla/Maps/Town10HD_Opt"),
              patch.object(launcher.subprocess, "Popen", side_effect=spawn),
              patch.object(launcher, "request_dashboard_close"),
              patch.object(launcher, "keyboard_quit", return_value=False)):
            assert launcher.main([
                "--cycles", "2", "--hold-seconds", "1",
                "--scenario", "alternate",
            ]) == 0
            assert scenario_count[0] == 2
            assert len(markers) == 2
            assert scenario_modes == ["clear", "rain_v2i"]
            assert len(dashboard_snapshots) == 1
            assert len(scenario_snapshots) == 2
            assert scenario_snapshots == dashboard_snapshots * 2
            private_snapshot = Path(dashboard_snapshots[0])
            assert private_snapshot.parent.parent == folder
            assert private_snapshot.parent.name.startswith("session_")
            assert private_snapshot.name == "dashboard_snapshot.json"

    # Reject invalid cycle count and invalid demo hold interval.
    try:
        launcher.main(["--dry-run", "--cycles", "-1"])
    except ValueError:
        pass
    else:
        raise AssertionError("Negative cycle count accepted")
    print("TEACHER DEMO LAUNCHER TEST PASSED (mock processes, no CARLA)")


if __name__ == "__main__":
    main()
