import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.data_layer import DashboardDataLayer


def main():
    layer = DashboardDataLayer(max_timeline_events=10)

    ego_state = {
        "world_position": {
            "x": -55.0,
            "y": 26.0,
            "z": 0.8,
        },
        "yaw_deg": 0.0,
        "speed_mps": 3.5,
    }

    fused_world = {
        "object_count": 2,
        "objects": [
            {
                "class_name": "pedestrian",
                "actor_id": 26,
                "confidence": 0.74,
                "world_position": {
                    "x": -44.65,
                    "y": 26.63,
                    "z": 0.95,
                },
                "sources": ["RSU-01"],
                "source_count": 1,
            },
            {
                "class_name": "car",
                "actor_id": 10,
                "confidence": 0.82,
                "world_position": {
                    "x": -50.0,
                    "y": 22.0,
                    "z": 0.8,
                },
                "sources": ["onboard", "RSU-01"],
                "source_count": 2,
            },
        ],
    }

    risk_result = {
        "highest_hazard": "WARNING",
        "risks": [
            {
                "class_name": "pedestrian",
                "actor_id": 26,
                "sources": ["RSU-01"],
                "distance_m": 10.35,
                "longitudinal_m": 10.33,
                "lateral_m": 0.63,
                "closing_speed_mps": 3.5,
                "ttc_s": 2.95,
                "in_path": True,
                "hazard_level": "WARNING",
                "world_position": {
                    "x": -44.65,
                    "y": 26.63,
                    "z": 0.95,
                },
            }
        ],
    }

    braking_command = {
        "hazard_level": "WARNING",
        "speed_before_mps": 3.5,
        "target_speed_mps": 3.475,
        "deceleration_mps2": 0.5,
        "braking_active": True,
        "emergency_braking": False,
        "estimated_stopping_distance_m": 12.25,
    }

    metrics_summary = {
        "latency": {
            "v2i_latency_ms": {
                "count": 219,
                "mean": 24.70,
                "p50": 24.45,
                "p95": 29.47,
                "min": 20.04,
                "max": 29.88,
            },
            "end_to_end_latency_ms": {
                "count": 219,
                "mean": 37.2,
                "p50": 36.8,
                "p95": 42.5,
                "min": 31.0,
                "max": 45.0,
            },
        },
        "detection": {
            "earliest_rsu_source": "RSU-01",
            "rsu_detection_lead_s": 2.95,
        },
        "braking": {
            "stop": {
                "actual_braking_distance_m": 8.72,
                "final_clearance_m": 4.18,
            }
        },
    }

    raw_events = [
        {
            "event_type": "hazard_transition",
            "sim_time_s": 1.0,
            "new_level": "CAUTION",
        },
        {
            "event_type": "hazard_transition",
            "sim_time_s": 1.1,
            "new_level": "NORMAL",
        },
        {
            "event_type": "stable_detection",
            "sim_time_s": 1.2,
            "source": "RSU-01",
        },
        {
            "event_type": "hazard_transition",
            "sim_time_s": 2.0,
            "new_level": "WARNING",
        },
        {
            "event_type": "brake_trigger",
            "sim_time_s": 2.0,
            "distance_m": 12.18,
        },
        {
            "event_type": "hazard_transition",
            "sim_time_s": 4.0,
            "new_level": "CRITICAL",
        },
        {
            "event_type": "vehicle_stopped",
            "sim_time_s": 5.0,
            "final_clearance_m": 4.18,
        },
    ]

    snapshot = layer.build_snapshot(
        frame_id=305400,
        sim_time_s=1767.67,
        ego_state=ego_state,
        fused_world=fused_world,
        risk_result=risk_result,
        braking_command=braking_command,
        braking_state="DRIVING",
        control_hazard="WARNING",
        metrics_summary=metrics_summary,
        metrics_events=raw_events,
        live_latencies={
            "onboard_inference_ms": 9.4,
            "rsu_inference_ms": 10.8,
            "v2i_latency_ms": 24.6,
            "fusion_latency_ms": 0.12,
            "risk_latency_ms": 0.08,
            "control_latency_ms": 0.01,
            "end_to_end_latency_ms": 35.5,
        },
    )

    assert snapshot["schema_version"] == "dashboard-v1"
    assert "road_context" not in snapshot  # Older snapshots stay unchanged
    assert snapshot["system_health"]["status"] == "HEALTHY"
    assert snapshot["ego"]["speed_mps"] == 3.5
    assert snapshot["hazard"]["control_level"] == "WARNING"
    assert snapshot["hazard"]["primary"]["class_name"] == "pedestrian"
    assert snapshot["hazard"]["primary"]["sources"] == ["RSU-01"]
    assert snapshot["braking"]["active"] is True
    assert snapshot["fusion"]["object_count"] == 2
    assert snapshot["latency_ms"]["v2i_latency_ms"] == 24.6
    assert len(snapshot["timeline"]) == 5

    timeline_types = [
        event["event_type"]
        for event in snapshot["timeline"]
    ]

    assert timeline_types == [
        "stable_detection",
        "hazard_transition",
        "brake_trigger",
        "hazard_transition",
        "vehicle_stopped",
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary_path = tmp_path / "metrics_summary.json"
        events_path = tmp_path / "metrics_events.csv"

        summary_path.write_text(
            json.dumps(metrics_summary),
            encoding="utf-8",
        )

        events_path.write_text(
            "event_type,sim_time_s,new_level,source\n"
            "stable_detection,1.2,,RSU-01\n"
            "hazard_transition,2.0,WARNING,\n",
            encoding="utf-8",
        )

        file_snapshot = layer.build_snapshot_from_files(
            summary_path=summary_path,
            events_path=events_path,
            frame_id=1,
            sim_time_s=1.0,
            ego_state=ego_state,
            fused_world=fused_world,
            risk_result=risk_result,
            braking_command=braking_command,
            braking_state="DRIVING",
            road_context={
                "version": 1,
                "source": "carla_waypoints",
                "coordinate_frame": "ego_relative_lateral_forward_m",
                "segments": [],
            },
        )

        assert file_snapshot["road_context"]["source"] == "carla_waypoints"
        assert len(file_snapshot["timeline"]) == 2
        assert (
            file_snapshot["latency_ms"]["v2i_latency_ms"]
            == 24.70
        )

        snapshot_path = tmp_path / "dashboard_snapshot.json"

        layer.write_snapshot_json(
            snapshot,
            snapshot_path,
        )

        published_snapshot = json.loads(
            snapshot_path.read_text(encoding="utf-8")
        )

        assert (
            published_snapshot["schema_version"]
            == "dashboard-v1"
        )
        assert (
            published_snapshot["hazard"]["control_level"]
            == "WARNING"
        )

    print("=== Perceptinet Dashboard Data Layer v1 Test ===")
    print(
        "Driver core:",
        f'speed={snapshot["ego"]["speed_mps"]:.2f}m/s',
        f'hazard={snapshot["hazard"]["control_level"]}',
        f'TTC={snapshot["hazard"]["primary"]["ttc_s"]:.2f}s',
    )
    print(
        "Pro core:",
        f'objects={snapshot["fusion"]["object_count"]}',
        f'v2i={snapshot["latency_ms"]["v2i_latency_ms"]:.2f}ms',
        f'e2e={snapshot["latency_ms"]["end_to_end_latency_ms"]:.2f}ms',
    )
    print(
        "Filtered timeline events:",
        len(snapshot["timeline"]),
    )
    print("DASHBOARD DATA TEST PASSED")


if __name__ == "__main__":
    main()
