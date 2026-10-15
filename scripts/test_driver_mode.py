import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.driver_mode import DriverModeRenderer


def main():
    snapshot = {
        "schema_version": "dashboard-v1",
        "frame_id": 443326,
        "sim_time_s": 100.0,
        "system_health": {
            "status": "HEALTHY",
            "issues": [],
        },
        "ego": {
            "speed_mps": 4.0,
            "yaw_deg": 0.16,
            "world_position": {
                "x": -64.0,
                "y": 24.4,
                "z": 0.8,
            },
        },
        "hazard": {
            "observed_level": "WARNING",
            "control_level": "WARNING",
            "primary": {
                "class_name": "pedestrian",
                "actor_id": 46,
                "sources": ["RSU-01"],
                "distance_m": 12.18,
                "longitudinal_m": 12.10,
                "lateral_m": 0.70,
                "ttc_s": 3.0,
                "hazard_level": "WARNING",
                "world_position": {
                    "x": -44.65,
                    "y": 26.63,
                    "z": 0.95,
                },
            },
        },
        "braking": {
            "state": "DRIVING",
            "active": True,
            "emergency": False,
            "deceleration_mps2": 0.5,
            "target_speed_mps": 3.95,
            "estimated_stopping_distance_m": 10.5,
        },
        "fusion": {
            "object_count": 3,
            "objects": [],
        },
        "latency_ms": {
            "v2i_latency_ms": 24.6,
            "end_to_end_latency_ms": 43.0,
        },
        "metrics": {},
        "timeline": [],
    }

    renderer = DriverModeRenderer(
        width=1280,
        height=720,
    )

    frame = renderer.render(snapshot)

    assert frame.shape == (720, 1280, 3)
    assert frame.dtype.name == "uint8"
    assert frame.mean() > 0.0

    critical = dict(snapshot)
    critical["hazard"] = dict(snapshot["hazard"])
    critical["hazard"]["control_level"] = "CRITICAL"
    critical["braking"] = dict(snapshot["braking"])
    critical["braking"]["emergency"] = True

    critical_frame = renderer.render(
        critical,
        stale=True,
    )

    assert critical_frame.shape == (720, 1280, 3)

    stopped = dict(snapshot)
    stopped["ego"] = dict(snapshot["ego"])
    stopped["ego"]["speed_mps"] = 0.0
    stopped["hazard"] = dict(snapshot["hazard"])
    stopped["hazard"]["control_level"] = "NORMAL"
    stopped["hazard"]["primary"] = dict(
        snapshot["hazard"]["primary"]
    )
    stopped["hazard"]["primary"]["ttc_s"] = None
    stopped["braking"] = dict(snapshot["braking"])
    stopped["braking"]["state"] = "DRIVING"
    stopped["braking"]["active"] = False
    stopped["braking"]["emergency"] = False
    stopped["braking"]["target_speed_mps"] = 0.0

    assert renderer._is_stopped(stopped) is True
    assert renderer._vehicle_action(stopped) == "STOPPED"

    stopped_frame = renderer.render(stopped)

    assert stopped_frame.shape == (720, 1280, 3)

    # v6.2: a real-road geometry payload changes the map rendering.
    road_context = copy.deepcopy(snapshot)
    road_context["road_context"] = {
        "version": 1,
        "source": "carla_waypoints",
        "coordinate_frame": "ego_relative_lateral_forward_m",
        "segments": [
            {"points": [[-1.5, 0], [-1.5, 30]], "width_m": 3.5,
             "junction": False},
            {"points": [[-15, 10], [15, 10]], "width_m": 3.5,
             "junction": True},
        ],
    }
    road_context["fusion"]["objects"] = [
        {
            "actor_id": 99, "class_name": "car",
            "world_position": dict(snapshot["ego"]["world_position"]),
            "sources": ["RSU-01"],
        },
        {
            "actor_id": 46, "class_name": "pedestrian",
            "world_position": {
                "x": -52.0, "y": 26.5, "z": 0.9,
            },
            "sources": ["RSU-01"],
        },
    ]
    geometry_frame = renderer.render(road_context)
    assert geometry_frame.shape == frame.shape
    assert (geometry_frame != frame).any()

    # Roadway must be a single filled union, not individually outlined
    # round-capped segments that create scalloped junction blobs.
    import numpy as np
    map_only = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert renderer._draw_context_roads(
        map_only, road_context["road_context"]
    )
    assert tuple(map_only[488, 430]) == (52, 50, 53)
    assert tuple(map_only[488, 500]) == (52, 50, 53)
    assert tuple(map_only[488, 200]) == (0, 0, 0)

    # A saved frame can be viewed in normal colors but must say PREVIEW,
    # not LIVE, when explicitly marked as recorded.
    recorded = copy.deepcopy(road_context)
    recorded["_preview_recorded"] = True
    assert renderer.render(recorded).shape == frame.shape
    assert (renderer.render(recorded) != geometry_frame).any()

    # A NORMAL-only zero-distance car is not an actionable Driver hazard.
    normal = copy.deepcopy(road_context)
    normal["hazard"]["control_level"] = "NORMAL"
    normal["hazard"]["primary"] = {
        "actor_id": 99, "class_name": "car",
        "hazard_level": "NORMAL",
        "longitudinal_m": 0.0, "lateral_m": 0.0,
        "distance_m": 0.0, "ttc_s": None,
    }
    assert renderer._active_primary(normal) is None
    assert renderer.render(normal).shape == frame.shape
    assert renderer.render(normal, stale=True).shape == frame.shape

    print("=== Perceptinet Driver Mode v1 Test ===")
    print("Resolution: 1280x720")
    print("Hazard card: WARNING")
    print("Cooperative source: RSU-01")
    print("Emergency state render: OK")
    print("Safe-stop state render: STOPPED")
    print("Merged CARLA road mask and recorded preview status: OK")
    print("Road-aware geometry and zero-distance NORMAL target: OK")
    print("DRIVER MODE TEST PASSED")


if __name__ == "__main__":
    main()
