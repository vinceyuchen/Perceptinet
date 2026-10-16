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

    # Semantic green/yellow/red is now intentionally used for states,
    # while the background and road geometry remain graphite grayscale.
    import numpy as np
    from unittest.mock import patch
    assert np.array_equal(frame[300, 80], np.repeat(frame[300, 80, 0], 3))
    assert np.array_equal(frame[320, 1050], np.repeat(frame[320, 1050, 0], 3))
    assert 15 <= frame[300, 80, 0] < 90   # Dark Road Map glass
    assert 15 <= frame[320, 1050, 0] < 90  # Dark response card
    assert np.max(frame[50:84, 52:300]) >= 210  # Bright ROAD MAP title

    safe = renderer._hazard_color("NORMAL")
    caution = renderer._hazard_color("CAUTION")
    warning = renderer._hazard_color("WARNING")
    critical_color = renderer._hazard_color("CRITICAL")
    # BGR channel ranking confirms green -> yellow -> red.
    assert safe[1] > safe[2] and safe[1] > safe[0]
    assert caution[1] > caution[0] and caution[2] > caution[0]
    assert warning[2] > warning[1] and warning[2] > warning[0]
    assert critical_color[2] >= critical_color[0]
    assert renderer._vehicle_state_color("DRIVING", stale=False) == safe
    assert renderer._vehicle_state_color("STOPPED", stale=False) == safe
    assert renderer._vehicle_state_color("SLOWING", stale=False) == caution
    assert renderer._vehicle_state_color("EMERGENCY BRAKING", stale=False) == critical_color
    assert renderer._vehicle_state_color("EMERGENCY BRAKING", stale=True) == (178, 178, 178)

    # Label-free road map: no EGO/CAR/PEDESTRIAN text beside markers.
    with patch.object(renderer, "_put_text", wraps=renderer._put_text) as text_spy:
        road_frame = renderer._base_canvas.copy()
        renderer._draw_road(road_frame, snapshot, stale=False)
        map_labels = [call.args[1] for call in text_spy.call_args_list]
    assert "ROAD MAP" in map_labels
    assert not any(label in map_labels for label in (
        "EGO", "CAR", "TRUCK", "BUS", "MOTORCYCLE", "BICYCLE",
        "PEDESTRIAN", "RSU-01", "NO TRACKED OBJECTS"
    )), map_labels

    # Exact source class chooses the corresponding uploaded SVG icon.
    with patch("dashboard.driver_mode.draw_icon", wraps=__import__(
        "dashboard.object_icon_assets", fromlist=["draw_icon"]
    ).draw_icon) as icon_spy:
        hazard_frame = renderer._base_canvas.copy()
        renderer._draw_hazard(hazard_frame, snapshot, stale=False)
        assert any(call.args[2] == "PEDESTRIAN"
                   for call in icon_spy.call_args_list)
        icon_spy.reset_mock()
        no_primary = copy.deepcopy(snapshot)
        no_primary["hazard"]["primary"]["longitudinal_m"] = 0.0
        renderer._draw_hazard(renderer._base_canvas.copy(), no_primary, stale=False)
        assert any(c.args[2] == "UNCONFIRMED"
                   for c in icon_spy.call_args_list)

        icon_spy.reset_mock()
        normal_heading = copy.deepcopy(snapshot)
        normal_heading["hazard"]["control_level"] = "NORMAL"
        renderer._draw_hazard(
            renderer._base_canvas.copy(), normal_heading, stale=False
        )
        assert any(c.args[2] == "SAFE"
                   for c in icon_spy.call_args_list)

    # Target card has only one main heading. No extra risk subtitle.
    for level, expected in (
        ("NORMAL", "SAFE"),
        ("CAUTION", "PEDESTRIAN"),
        ("WARNING", "PEDESTRIAN"),
        ("CRITICAL", "PEDESTRIAN"),
    ):
        case = copy.deepcopy(snapshot)
        case["hazard"]["control_level"] = level
        case["hazard"]["primary"]["hazard_level"] = level
        if level == "CRITICAL":
            case["braking"]["emergency"] = True
        with patch.object(renderer, "_put_text",
                          wraps=renderer._put_text) as text_spy:
            renderer._draw_hazard(renderer._base_canvas.copy(), case, stale=False)
            names = [call.args[1] for call in text_spy.call_args_list]
        assert expected in names
        assert "TARGET" not in names
        assert "ROAD CLEAR" not in names
        assert "UNCONFIRMED" not in names
        assert level not in names
        assert "DISTANCE" in names and "TTC" in names

    # The two cards share an IDENTICAL accent from the same risk state.
    for level, expected_color in (
        ("NORMAL", safe),
        ("CAUTION", caution),
        ("WARNING", warning),
        ("CRITICAL", critical_color),
    ):
        case = copy.deepcopy(snapshot)
        case["hazard"]["control_level"] = level
        case["hazard"]["primary"]["hazard_level"] = level
        if level == "CRITICAL":
            case["braking"]["emergency"] = True
        assert renderer._unified_status_color(case, stale=False) == expected_color
        action = renderer._vehicle_action(case)
        with patch.object(renderer, "_put_text",
                          wraps=renderer._put_text) as write_spy:
            renderer._draw_hazard(renderer._base_canvas.copy(), case, stale=False)
            hazard_colors = [
                c.kwargs["color"] for c in write_spy.call_args_list
                if c.args[1] in ("PEDESTRIAN", "SAFE")
            ]
            assert hazard_colors == [expected_color]
        with patch.object(renderer, "_put_text",
                          wraps=renderer._put_text) as write_spy:
            renderer._draw_vehicle(
                renderer._base_canvas.copy(), case, stale=False,
                presentation={"vehicle_action": action}
            )
            vehicle_colors = [
                c.kwargs["color"] for c in write_spy.call_args_list
                if c.args[1] == action
            ]
            assert vehicle_colors == [expected_color]
    assert renderer._unified_status_color(snapshot, stale=True) == (178, 178, 178)

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
    print("v7.3 SAFE/UNCONFIRMED icons, one-line heading, unified severity: OK")
    print("Road-aware geometry and zero-distance NORMAL target: OK")
    print("DRIVER MODE TEST PASSED")


if __name__ == "__main__":
    main()
