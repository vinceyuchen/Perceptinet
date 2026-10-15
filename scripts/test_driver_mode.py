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

    print("=== Perceptinet Driver Mode v1 Test ===")
    print("Resolution: 1280x720")
    print("Hazard card: WARNING")
    print("Cooperative source: RSU-01")
    print("Emergency state render: OK")
    print("DRIVER MODE TEST PASSED")


if __name__ == "__main__":
    main()
