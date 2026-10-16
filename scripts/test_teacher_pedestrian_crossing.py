"""No-CARLA tests for physically animated Scenario 2 crossing triggers.

Tests crossing geometry and explicitly separate kinematic envelope contact
from real CARLA collision sensor evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.pedestrian_crossing import (
    PedestrianCrossing, kinematic_footprint_overlap, relative_position,
)
from dashboard.presentation_state import derive_presentation_state
from dashboard.driver_mode import DriverModeRenderer, SEVERE_RED


def main() -> None:
    fwd = (1.0, 0.0)
    crossing = PedestrianCrossing(
        trigger_distance_m=8.5, walking_speed_mps=1.05
    )
    initial = crossing.step((0, 0), (11.0, 1.9), fwd)
    assert not initial.walking
    assert initial.speed_mps == 0

    active = crossing.step((0, 0), (8.2, 1.9), fwd)
    assert active.walking
    assert abs(active.speed_mps - 1.05) < 1e-6
    assert active.direction_xy == (0.0, -1.0)
    assert abs(active.lateral_m - 1.9) < 1e-6
    assert crossing.triggered

    crossing.step((0, 0), (3.5, 0.0), fwd)
    complete = crossing.step((0, 0), (2.0, -2.6), fwd)
    assert not complete.walking
    assert crossing.finished
    assert not crossing.step((0, 0), (1.0, -1.0), fwd).walking

    other_side = PedestrianCrossing(trigger_distance_m=8.5)
    result = other_side.step((0, 0), (8, -1.8), fwd)
    assert result.direction_xy == (0.0, 1.0)

    assert relative_position((0, 0), (7, -3), fwd) == (7.0, -3.0)
    kwargs = dict(
        ego_half_length_m=2.2,
        ego_half_width_m=0.95,
        pedestrian_radius_m=0.30,
    )
    assert not kinematic_footprint_overlap(
        (0, 0), (4, 1), fwd, **kwargs
    )
    assert kinematic_footprint_overlap(
        (0, 0), (2.1, 0.5), fwd, **kwargs
    )
    assert not kinematic_footprint_overlap(
        (0, 0), (2.7, 1.4), fwd, **kwargs
    )

    snapshot = {
        "ego": {"speed_mps": 0.0},
        "braking": {"target_speed_mps": 4.0, "active": False},
        "hazard": {"control_level": "NORMAL", "primary": None},
        "system_health": {"status": "DATA LIMITED"},
        "experiment": {
            "scenario": "rain_v2i", "kinematic_contact_observed": True,
        },
    }
    assert derive_presentation_state(snapshot)["vehicle_action"] == "CONTACT (SIM)"
    renderer = DriverModeRenderer()
    assert renderer._unified_status_color(snapshot, stale=False) == SEVERE_RED
    assert renderer._unified_status_color(snapshot, stale=True) != SEVERE_RED
    snapshot["experiment"] = {
        "scenario": "rain_v2i",
        "carla_collision_events": [
            {"other_type": "walker.pedestrian.0001", "frame": 100}
        ],
    }
    assert derive_presentation_state(snapshot)["vehicle_action"] == "CARLA CONTACT"
    assert renderer._unified_status_color(snapshot, stale=False) == SEVERE_RED

    print("SCENARIO 2 MOVING PEDESTRIAN / CONTACT TEST PASSED")


if __name__ == "__main__":
    main()
