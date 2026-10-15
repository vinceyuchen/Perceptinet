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

    # v6.6 is fixed, wider and strictly monotonic in the forward axis.
    assert renderer._road_projection(0.0, 0.0) == (430, 616)
    assert renderer._road_projection(-24.0, 0.0)[0] == 178
    assert renderer._road_projection(24.0, 0.0)[0] == 682
    assert renderer._road_projection(0.0, 50.0)[1] == 141
    road_ys = [
        renderer._road_projection(0.0, distance)[1]
        for distance in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
    ]
    assert all(first > second for first, second in zip(road_ys, road_ys[1:]))

    frame = renderer.render(snapshot)

    assert frame.shape == (720, 1280, 3)
    assert frame.dtype.name == "uint8"
    assert frame.mean() > 0.0

    # v6.8 is intentionally grayscale: no blue/green/orange pixels.
    import numpy as np
    assert np.array_equal(frame[:, :, 0], frame[:, :, 1])
    assert np.array_equal(frame[:, :, 1], frame[:, :, 2])
    assert 15 <= frame[300, 80, 0] < 90   # Dark Road Map glass
    assert 15 <= frame[320, 1050, 0] < 90  # Dark response card
    assert np.max(frame[50:84, 52:300]) >= 210  # Bright ROAD MAP title

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
        "segments": (
            [
                {"points": [[-1.5, float(fwd)],
                            [-1.5, float(fwd + 3)]],
                 "width_m": 3.5, "junction": False}
                for fwd in range(-6, 33, 3)
            ] + [
                {"points": [[float(lat), 10.0],
                            [float(lat + 3), 10.0]],
                 "width_m": 3.5, "junction": True}
                for lat in range(-15, 15, 3)
            ]
        ),
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

    # v6.5: only smoothed, fixed-width schematic road boundaries.
    # At a synthetic crossroad, the centre of both roads remains clear.
    map_only = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert renderer._draw_context_roads(
        map_only, road_context["road_context"]
    )
    assert not np.any(map_only[488, 430])  # Intersection interior
    assert not np.any(map_only[488, 500])  # Cross street interior
    assert not np.any(map_only[488, 200])  # Outside street
    assert np.any(map_only[257:286, 346:375])  # Left fixed-width edge
    assert np.any(map_only[257:286, 455:481])  # Right fixed-width edge
    assert np.any(map_only[390:417, 291:322])  # Cross-road edge

    # Fallback without CARLA data is also only two parallel road lines.
    fallback_only = np.zeros_like(map_only)
    renderer._draw_fallback_road(fallback_only)
    assert not np.any(fallback_only[390, 430])
    left = renderer._road_projection(-5.4, 10.0)[0]
    right = renderer._road_projection(5.4, 10.0)[0]
    assert np.any(fallback_only[390, left])
    assert np.any(fallback_only[390, right])

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
    print("v6.9 dark grayscale glass and CARLA road layout: OK")
    print("Road-aware geometry and zero-distance NORMAL target: OK")
    print("DRIVER MODE TEST PASSED")


if __name__ == "__main__":
    main()
