from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.state_validator import DashboardStateValidator


def build_snapshot():
    return {
        "schema_version": "dashboard-v1",
        "frame_id": 100,
        "sim_time_s": 5.0,
        "system_health": {
            "status": "HEALTHY",
            "issues": [],
        },
        "ego": {
            "speed_mps": 3.8,
            "yaw_deg": 0.0,
            "world_position": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.8,
            },
        },
        "hazard": {
            "observed_level": "WARNING",
            "control_level": "WARNING",
            "primary": {
                "class_name": "pedestrian",
                "actor_id": 7,
                "sources": ["RSU-01"],
                "distance_m": 10.0,
                "ttc_s": 2.8,
            },
        },
        "braking": {
            "state": "DRIVING",
            "active": True,
            "emergency": False,
            "deceleration_mps2": 1.5,
            "target_speed_mps": 3.725,
        },
        "fusion": {
            "object_count": 1,
            "objects": [
                {
                    "class_name": "pedestrian",
                    "actor_id": 7,
                    "sources": ["RSU-01"],
                }
            ],
        },
        "latency_ms": {
            "onboard_inference_ms": 11.0,
            "rsu_inference_ms": 12.0,
            "v2i_latency_ms": 25.0,
            "fusion_latency_ms": 0.03,
            "risk_latency_ms": 0.02,
            "control_latency_ms": 0.01,
            "end_to_end_latency_ms": 50.0,
        },
        "timeline": [],
    }


def main():
    validator = DashboardStateValidator(
        cooperative_expected=True
    )

    snapshot = build_snapshot()

    result = validator.validate(snapshot)

    assert result["errors"] == []

    next_snapshot = deepcopy(snapshot)
    next_snapshot["frame_id"] = 101
    next_snapshot["sim_time_s"] = 5.05

    result = validator.validate(
        next_snapshot,
        previous=snapshot,
    )

    assert result["errors"] == []

    broken = deepcopy(snapshot)
    broken["hazard"]["control_level"] = "NORMAL"
    broken["fusion"]["object_count"] = 3

    result = validator.validate(broken)

    assert any(
        "control hazard" in error
        for error in result["errors"]
    )
    assert any(
        "object_count" in error
        for error in result["errors"]
    )

    stopped = deepcopy(snapshot)
    stopped["frame_id"] = 102
    stopped["ego"]["speed_mps"] = 0.0
    stopped["hazard"]["observed_level"] = "NORMAL"
    stopped["hazard"]["control_level"] = "CRITICAL"
    stopped["braking"]["state"] = "STOPPED_HOLD"
    stopped["braking"]["active"] = True
    stopped["braking"]["emergency"] = True
    stopped["braking"]["deceleration_mps2"] = 6.0
    stopped["braking"]["target_speed_mps"] = 0.0

    result = validator.validate(stopped)

    assert result["errors"] == []

    print(
        "=== Perceptinet Dashboard State Validator Test ==="
    )
    print("Healthy cooperative state: PASS")
    print("Frame/time monotonic state: PASS")
    print("Contradiction detection: PASS")
    print("Stopped/latched CRITICAL state: PASS")
    print("DASHBOARD STATE VALIDATOR TEST PASSED")


if __name__ == "__main__":
    main()
