import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.app import PerceptinetDashboardApp
from dashboard.ui_theme import draw_text, rounded_rect

import numpy as np


def build_snapshot():
    return {
        "schema_version": "dashboard-v1",
        "frame_id": 600001,
        "sim_time_s": 125.0,
        "system_health": {
            "status": "HEALTHY",
            "issues": [],
        },
        "ego": {
            "speed_mps": 3.5,
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
                "distance_m": 11.0,
                "longitudinal_m": 11.0,
                "lateral_m": 1.5,
                "ttc_s": 2.9,
                "hazard_level": "WARNING",
                "world_position": {
                    "x": 11.0,
                    "y": 1.5,
                    "z": 0.95,
                },
            },
        },
        "braking": {
            "state": "SLOWING",
            "active": True,
            "emergency": False,
            "deceleration_mps2": 0.5,
            "target_speed_mps": 3.45,
            "estimated_stopping_distance_m": 13.0,
        },
        "fusion": {
            "object_count": 2,
            "objects": [
                {
                    "class_name": "pedestrian",
                    "confidence": 0.76,
                    "actor_id": 41,
                    "world_position": {
                        "x": 11.0,
                        "y": 1.5,
                        "z": 0.95,
                    },
                    "sources": ["RSU-01"],
                    "source_count": 1,
                },
                {
                    "class_name": "car",
                    "confidence": 0.83,
                    "actor_id": 10,
                    "world_position": {
                        "x": 20.0,
                        "y": -2.0,
                        "z": 0.8,
                    },
                    "sources": ["onboard", "RSU-01"],
                    "source_count": 2,
                },
            ],
        },
        "latency_ms": {
            "onboard_inference_ms": 10.5,
            "rsu_inference_ms": 11.0,
            "v2i_latency_ms": 25.0,
            "fusion_latency_ms": 0.4,
            "risk_latency_ms": 0.2,
            "control_latency_ms": 0.1,
            "end_to_end_latency_ms": 36.2,
        },
        "metrics": {
            "detection": {},
            "braking": {},
        },
        "timeline": [
            {
                "event_type": "stable_detection",
                "sim_time_s": 122.0,
                "frame_id": 599950,
                "source_id": "RSU-01",
            },
            {
                "event_type": "hazard_transition",
                "sim_time_s": 124.0,
                "frame_id": 599990,
                "previous_level": "CAUTION",
                "new_level": "WARNING",
            },
        ],
    }


def main():
    snapshot = build_snapshot()

    theme_canvas = np.zeros((100, 320, 3), dtype=np.uint8)
    rounded_rect(
        theme_canvas,
        (5, 5),
        (315, 95),
        fill=(24, 24, 24),
        border=(55, 55, 55),
        radius=10,
    )
    draw_text(
        theme_canvas,
        "PERCEPTINET",
        (20, 58),
        scale=0.7,
        thickness=2,
    )
    assert theme_canvas.mean() > 0.0

    app = PerceptinetDashboardApp(
        width=1600,
        height=900,
    )

    assert app.mode == app.DRIVER_MODE

    driver_frame = app.render(snapshot)
    assert driver_frame.shape == (900, 1600, 3)

    dx1, dy1, dx2, dy2 = app.driver_button
    px1, py1, px2, py2 = app.pro_button

    assert app.handle_click(
        (px1 + px2) // 2,
        (py1 + py2) // 2,
    ) == app.PRO_MODE

    pro_frame = app.render(snapshot)
    assert pro_frame.shape == (900, 1600, 3)

    assert app.handle_click(
        (dx1 + dx2) // 2,
        (dy1 + dy2) // 2,
    ) == app.DRIVER_MODE

    assert app.toggle_mode() == app.PRO_MODE
    assert app.toggle_mode() == app.DRIVER_MODE

    print("=== Perceptinet Unified Dashboard Test ===")
    print("Sharp typography / rounded panels: OK")
    print("Driver Mode render: OK")
    print("Pro Mode render: OK")
    print("Driver button: OK")
    print("Pro button: OK")
    print("Mode switching: OK")
    print("UNIFIED DASHBOARD TEST PASSED")


if __name__ == "__main__":
    main()
