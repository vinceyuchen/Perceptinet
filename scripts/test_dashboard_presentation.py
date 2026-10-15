from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.presentation_state import derive_presentation_state


def base_snapshot():
    return {
        "schema_version": "dashboard-v1",
        "system_health": {
            "status": "HEALTHY",
            "issues": [],
        },
        "ego": {
            "speed_mps": 4.0,
        },
        "hazard": {
            "observed_level": "NORMAL",
            "control_level": "NORMAL",
            "primary": None,
        },
        "braking": {
            "state": "DRIVING",
            "active": False,
            "emergency": False,
            "deceleration_mps2": 0.0,
            "target_speed_mps": 4.0,
        },
    }


def main():
    normal = base_snapshot()

    state = derive_presentation_state(
        normal,
        stale=False,
    )

    assert state["data_status"] == "LIVE"
    assert state["display_health"] == "HEALTHY"
    assert state["vehicle_action"] == "DRIVING"
    assert state["footer_message"] == "ONBOARD PERCEPTION ACTIVE"

    warning = deepcopy(normal)
    warning["hazard"]["control_level"] = "WARNING"
    warning["hazard"]["primary"] = {
        "class_name": "pedestrian",
        "sources": ["RSU-01"],
    }
    warning["braking"]["active"] = True
    warning["braking"]["deceleration_mps2"] = 1.5
    warning["braking"]["target_speed_mps"] = 3.9

    state = derive_presentation_state(
        warning,
        stale=False,
    )

    assert state["cooperative_active"] is True
    assert state["perception_text"] == "COOPERATIVE ACTIVE"
    assert state["vehicle_action"] == "SLOWING"
    assert state["footer_message"] == "HAZARD AHEAD - VEHICLE SLOWING"

    critical = deepcopy(warning)
    critical["hazard"]["control_level"] = "CRITICAL"
    critical["braking"]["emergency"] = True

    state = derive_presentation_state(
        critical,
        stale=False,
    )

    assert state["vehicle_action"] == "EMERGENCY BRAKING"
    assert state["footer_message"] == "BRAKE NOW - IMMINENT COLLISION RISK"

    stopped = deepcopy(critical)
    stopped["ego"]["speed_mps"] = 0.0
    stopped["braking"]["state"] = "STOPPED_HOLD"
    stopped["braking"]["target_speed_mps"] = 0.0

    state = derive_presentation_state(
        stopped,
        stale=False,
    )

    assert state["stopped"] is True
    assert state["vehicle_action"] == "STOPPED"
    assert state["footer_message"] == "VEHICLE STOPPED - PEDESTRIAN AHEAD"

    stale = derive_presentation_state(
        warning,
        stale=True,
    )

    assert stale["data_status"] == "DATA STALE"
    assert stale["display_health"] == "DATA LIMITED"
    assert stale["perception_text"] == "COOPERATIVE LIMITED"
    assert stale["footer_message"] == "COOPERATIVE DATA STALE"

    degraded = deepcopy(warning)
    degraded["system_health"] = {
        "status": "DEGRADED",
        "issues": ["V2I_HIGH_LATENCY"],
    }

    state = derive_presentation_state(
        degraded,
        stale=False,
    )

    assert state["display_health"] == "DEGRADED"

    print("=== Perceptinet Dashboard Presentation Policy Test ===")
    print("LIVE / DATA STALE: PASS")
    print("HEALTHY / DATA LIMITED / DEGRADED: PASS")
    print("DRIVING / SLOWING / EMERGENCY / STOPPED: PASS")
    print("Onboard / Cooperative source semantics: PASS")
    print("Hazard footer priority: PASS")
    print("DASHBOARD PRESENTATION POLICY TEST PASSED")


if __name__ == "__main__":
    main()
