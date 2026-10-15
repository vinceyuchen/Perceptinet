from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.app import PerceptinetDashboardApp


def build_snapshot():
    return {
        "schema_version": "dashboard-v1",
        "frame_id": 700321,
        "sim_time_s": 215.45,
        "system_health": {
            "status": "HEALTHY",
            "issues": [],
        },
        "ego": {
            "speed_mps": 3.4,
            "yaw_deg": 0.16,
            "world_position": {
                "x": -64.64,
                "y": 24.47,
                "z": 0.04,
            },
        },
        "hazard": {
            "observed_level": "WARNING",
            "control_level": "WARNING",
            "primary": {
                "class_name": "pedestrian",
                "actor_id": 46,
                "track_id": "pedestrian:actor:46",
                "sources": ["RSU-01"],
                "distance_m": 10.8,
                "longitudinal_m": 10.6,
                "lateral_m": 2.1,
                "ttc_s": 2.9,
                "hazard_level": "WARNING",
                "world_position": {
                    "x": -44.65,
                    "y": 26.63,
                    "z": 0.96,
                },
            },
        },
        "braking": {
            "state": "SLOWING",
            "active": True,
            "emergency": False,
            "deceleration_mps2": 0.5,
            "target_speed_mps": 3.35,
            "estimated_stopping_distance_m": 12.0,
        },
        "fusion": {
            "object_count": 3,
            "objects": [
                {
                    "class_name": "pedestrian",
                    "confidence": 0.81,
                    "actor_id": 46,
                    "track_id": "pedestrian:actor:46",
                    "world_position": {
                        "x": -44.65,
                        "y": 26.63,
                        "z": 0.96,
                    },
                    "display_world_position": {
                        "x": -44.65,
                        "y": 26.63,
                        "z": 0.96,
                    },
                    "sources": ["RSU-01"],
                    "display_sources": ["RSU-01"],
                    "source_count": 1,
                },
                {
                    "class_name": "truck",
                    "confidence": 0.76,
                    "actor_id": 45,
                    "track_id": "truck:actor:45",
                    "world_position": {
                        "x": -47.65,
                        "y": 27.22,
                        "z": 0.03,
                    },
                    "display_world_position": {
                        "x": -47.65,
                        "y": 27.22,
                        "z": 0.03,
                    },
                    "sources": ["onboard", "RSU-01"],
                    "display_sources": ["onboard", "RSU-01"],
                    "source_count": 2,
                },
                {
                    "class_name": "car",
                    "confidence": 0.72,
                    "actor_id": 99,
                    "track_id": "car:actor:99",
                    "world_position": {
                        "x": -52.0,
                        "y": 22.7,
                        "z": 0.04,
                    },
                    "display_world_position": {
                        "x": -52.0,
                        "y": 22.7,
                        "z": 0.04,
                    },
                    "sources": ["onboard"],
                    "display_sources": ["onboard"],
                    "source_count": 1,
                },
            ],
        },
        "latency_ms": {
            "onboard_inference_ms": 11.4,
            "rsu_inference_ms": 12.1,
            "v2i_latency_ms": 24.7,
            "fusion_latency_ms": 0.04,
            "risk_latency_ms": 0.03,
            "control_latency_ms": 0.01,
            "end_to_end_latency_ms": 48.3,
        },
        "metrics": {},
        "timeline": [
            {
                "event_type": "stable_detection",
                "sim_time_s": 212.20,
                "frame_id": 700256,
                "source_id": "RSU-01",
            },
            {
                "event_type": "hazard_transition",
                "sim_time_s": 214.10,
                "frame_id": 700294,
                "old_level": "CAUTION",
                "new_level": "WARNING",
            },
            {
                "event_type": "brake_trigger",
                "sim_time_s": 214.15,
                "frame_id": 700295,
                "distance_m": 12.18,
                "ttc_s": 3.0,
            },
        ],
    }


def main():
    output_dir = (
        REPO_ROOT
        / "results"
        / "dashboard_polish_preview"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot = build_snapshot()

    app = PerceptinetDashboardApp(
        width=1600,
        height=900,
    )

    app.set_mode(app.DRIVER_MODE)
    driver = app.render(
        snapshot,
        stale=False,
    )

    app.set_mode(app.PRO_MODE)
    pro = app.render(
        snapshot,
        stale=False,
    )

    app.set_mode(app.DRIVER_MODE)
    stale = app.render(
        snapshot,
        stale=True,
    )

    paths = {
        "driver": output_dir / "driver_mode_polish.png",
        "pro": output_dir / "pro_mode_polish.png",
        "stale": output_dir / "driver_mode_stale.png",
    }

    cv2.imwrite(
        str(paths["driver"]),
        driver,
    )
    cv2.imwrite(
        str(paths["pro"]),
        pro,
    )
    cv2.imwrite(
        str(paths["stale"]),
        stale,
    )

    print("=== Perceptinet Dashboard Final Polish Preview ===")
    print("Driver:", paths["driver"])
    print("Pro:", paths["pro"])
    print("Stale:", paths["stale"])
    print("DASHBOARD POLISH PREVIEW READY")


if __name__ == "__main__":
    main()
