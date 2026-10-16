"""Regression tests: CARLA TCP readiness is not the same as world RPC readiness.

These tests use mocked CARLA clients and do NOT require the simulator,
GPU, YOLO, or an actual open TCP port.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Perceptinet_Teacher_Demo as demo


def main() -> None:
    # A CARLA client must return its actual map before it can be "ready".
    client = Mock()
    client.get_world.return_value.get_map.return_value.name = (
        "Carla/Maps/Town10HD_Opt"
    )
    fake_carla = SimpleNamespace(Client=Mock(return_value=client))
    with patch.dict(sys.modules, {"carla": fake_carla}):
        assert demo.carla_rpc_probe(4.0) == "Carla/Maps/Town10HD_Opt"
    client.set_timeout.assert_called_once_with(4.0)

    # Port may be open while Unreal is busy. Retry real RPC until ready.
    with (
        patch.object(
            demo, "carla_rpc_probe",
            side_effect=[RuntimeError("time-out of 5000ms"),
                         RuntimeError("time-out of 5000ms"),
                         "Carla/Maps/Town10HD_Opt"],
        ) as rpc,
        patch.object(demo, "keyboard_quit", return_value=False),
        patch.object(demo.time, "sleep"),
    ):
        assert demo.wait_for_carla_rpc(90.0) == "Carla/Maps/Town10HD_Opt"
        assert rpc.call_count == 3

    # Persistent RPC failure must NOT launch a demo or kill existing CARLA.
    with (
        patch.object(demo, "carla_rpc_probe",
                     side_effect=RuntimeError("time-out")) as rpc,
        patch.object(demo, "keyboard_quit", return_value=False),
        patch.object(demo.time, "monotonic",
                     side_effect=[0.0, 0.1, 1.1]),
        patch.object(demo.time, "sleep"),
    ):
        try:
            demo.wait_for_carla_rpc(1.0)
        except TimeoutError as exc:
            assert "world/map RPC" in str(exc)
            assert "will not terminate an existing CARLA" in str(exc)
        else:
            raise AssertionError("Unresponsive CARLA was incorrectly ready")
        assert rpc.call_count == 1

    # Even an already-open TCP port requires an RPC probe first.
    args = SimpleNamespace(carla_ready_timeout=8.0, carla_exe=None,
                           carla_start_timeout=20.0)
    with (
        patch.object(demo, "connected", return_value=True),
        patch.object(demo, "wait_for_carla_rpc",
                     return_value="Carla/Maps/Town10HD_Opt") as ready,
        patch.object(demo.subprocess, "Popen") as spawned,
    ):
        assert demo.launch_carla_if_needed(args) is None
        ready.assert_called_once_with(8.0)
        spawned.assert_not_called()

    # Dry-run must be executable even without a CARLA Python package.
    assert demo.main(["--dry-run"]) == 0
    print("CARLA RPC READINESS / TIMEOUT TEST PASSED")


if __name__ == "__main__":
    main()
