import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.pro_mode import BLUE, ProModeRenderer


def main():
    snapshot = {
        "schema_version": "dashboard-v1",
        "frame_id": 500001,
        "sim_time_s": 120.5,
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
                "actor_id": 41,
                "sources": ["RSU-01"],
                "distance_m": 12.0,
                "longitudinal_m": 12.0,
                "lateral_m": 1.2,
                "ttc_s": 3.1,
                "hazard_level": "WARNING",
                "world_position": {
                    "x": 12.0,
                    "y": 1.2,
                    "z": 0.95,
                },
            },
        },
        "braking": {
            "state": "DRIVING",
            "active": True,
            "emergency": False,
            "deceleration_mps2": 0.5,
            "target_speed_mps": 3.75,
            "estimated_stopping_distance_m": 14.2,
        },
        "fusion": {
            "object_count": 3,
            "objects": [
                {
                    "class_name": "pedestrian",
                    "confidence": 0.78,
                    "actor_id": 41,
                    "world_position": {
                        "x": 12.0,
                        "y": 1.2,
                        "z": 0.95,
                    },
                    "sources": ["RSU-01"],
                    "source_count": 1,
                },
                {
                    "class_name": "car",
                    "confidence": 0.84,
                    "actor_id": 10,
                    "world_position": {
                        "x": 20.0,
                        "y": -2.0,
                        "z": 0.8,
                    },
                    "sources": ["onboard", "RSU-01"],
                    "source_count": 2,
                },
                {
                    "class_name": "bus",
                    "confidence": 0.73,
                    "actor_id": 11,
                    "world_position": {
                        "x": 27.0,
                        "y": 3.0,
                        "z": 0.8,
                    },
                    "sources": ["onboard"],
                    "source_count": 1,
                },
            ],
        },
        "latency_ms": {
            "onboard_inference_ms": 10.8,
            "rsu_inference_ms": 11.2,
            "v2i_latency_ms": 24.6,
            "fusion_latency_ms": 0.45,
            "risk_latency_ms": 0.18,
            "control_latency_ms": 0.12,
            "end_to_end_latency_ms": 36.7,
        },
        "metrics": {
            "detection": {},
            "braking": {},
        },
        "timeline": [
            {
                "event_type": "stable_detection",
                "sim_time_s": 118.2,
                "frame_id": 499950,
                "source_id": "RSU-01",
            },
            {
                "event_type": "hazard_transition",
                "sim_time_s": 119.0,
                "frame_id": 499966,
                "old_level": "CAUTION",
                "new_level": "WARNING",
            },
            {
                "event_type": "brake_trigger",
                "sim_time_s": 119.1,
                "frame_id": 499968,
                "distance_m": 12.2,
                "ttc_s": 3.0,
            },
        ],
    }

    renderer = ProModeRenderer()

    sources = renderer.available_sources(snapshot)

    assert sources == ["ALL", "onboard", "RSU-01"]
    assert renderer.cycle_source(snapshot) == "onboard"
    assert renderer.cycle_source(snapshot) == "RSU-01"

    assert renderer._timeline_detail(
        snapshot["timeline"][1]
    ) == "CAUTION -> WARNING"

    first_source = renderer._first_detection_source(snapshot)
    assert first_source == "RSU-01"

    forward_max, lateral_half = renderer._fused_view_limits(snapshot)
    assert forward_max == 30.0
    assert lateral_half >= 4.0

    normal_primary = renderer._object_color(
        snapshot["fusion"]["objects"][0],
        is_primary=True,
        hazard_level="NORMAL",
    )
    assert normal_primary == BLUE

    frame = renderer.render(snapshot)

    assert frame.shape == (900, 1600, 3)
    assert frame.dtype.name == "uint8"
    assert frame.mean() > 0.0

    print("=== Perceptinet Pro Mode v1 Test ===")
    print("Fused World Model: OK")
    print("Source inspector: ALL / onboard / RSU-01")
    print("Active contributor: RSU-01")
    print("Latency panel: OK")
    print("Event timeline old/new levels: OK")
    print("First detection source: RSU-01")
    print("Dynamic fused-world scale: OK")
    print("NORMAL primary hazard color: non-red")
    print("PRO MODE TEST PASSED")


if __name__ == "__main__":
    main()
