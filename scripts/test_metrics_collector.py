import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.collector import MetricsCollector


def packet(
    source,
    frame_id,
    timestamp,
    detected,
    inference_ms,
    v2i_ms=None,
):
    data = {
        "source": source,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "inference_ms": inference_ms,
        "detections": (
            [
                {
                    "class_name": "pedestrian",
                    "confidence": 0.75,
                }
            ]
            if detected
            else []
        ),
    }

    if v2i_ms is not None:
        data["v2i"] = {
            "source_id": "RSU-01",
            "latency_ms": v2i_ms,
        }

    return data


def main():
    collector = MetricsCollector(stable_detection_frames=3)

    # One-frame Onboard flicker must NOT count as a stable detection.
    collector.record_perception_packet(
        packet("onboard", 10, 0.50, True, 9.0)
    )
    collector.record_perception_packet(
        packet("onboard", 11, 0.55, False, 9.5)
    )

    # RSU obtains a stable three-frame detection first.
    for frame_id, timestamp in [
        (20, 1.00),
        (21, 1.05),
        (22, 1.10),
    ]:
        collector.record_perception_packet(
            packet(
                "rsu_01",
                frame_id,
                timestamp,
                True,
                10.0,
                25.0,
            )
        )

    # Onboard later obtains its own stable detection.
    for frame_id, timestamp in [
        (30, 1.80),
        (31, 1.85),
        (32, 1.90),
    ]:
        collector.record_perception_packet(
            packet(
                "onboard",
                frame_id,
                timestamp,
                True,
                8.0,
            )
        )

    collector.record_pipeline_timing(
        fusion_ms=0.40,
        risk_ms=0.20,
        control_ms=0.10,
        end_to_end_ms=35.70,
    )

    collector.record_hazard_transition(
        old_level="CAUTION",
        new_level="WARNING",
        sim_time_s=3.00,
        frame_id=60,
    )

    collector.record_brake_trigger(
        sim_time_s=3.00,
        frame_id=60,
        distance_m=12.18,
        ttc_s=3.00,
        speed_mps=4.00,
        hazard_level="WARNING",
    )

    collector.record_stop(
        sim_time_s=5.20,
        frame_id=104,
        actual_braking_distance_m=8.84,
        final_clearance_m=4.08,
        hazard_level="CRITICAL",
    )

    summary = collector.summary()

    detection = summary["detection"]
    braking = summary["braking"]
    latency = summary["latency"]

    assert detection["earliest_rsu_source"] == "RSU-01"
    assert abs(
        detection["earliest_rsu_stable_time_s"] - 1.00
    ) < 1e-9
    assert abs(
        detection["onboard_stable_time_s"] - 1.80
    ) < 1e-9
    assert abs(
        detection["rsu_detection_lead_s"] - 0.80
    ) < 1e-9

    assert latency["v2i_latency_ms"]["count"] == 3
    assert abs(
        latency["v2i_latency_ms"]["mean"] - 25.0
    ) < 1e-9

    assert braking["trigger"]["hazard_level"] == "WARNING"
    assert braking["stop"]["hazard_level"] == "CRITICAL"
    assert abs(
        braking["stop"]["final_clearance_m"] - 4.08
    ) < 1e-9

    print("=== Perceptinet Metrics v1 Test ===")
    print(
        "Stable RSU detection:",
        f'{detection["earliest_rsu_stable_time_s"]:.2f}s',
    )
    print(
        "Stable Onboard detection:",
        f'{detection["onboard_stable_time_s"]:.2f}s',
    )
    print(
        "RSU detection lead:",
        f'{detection["rsu_detection_lead_s"]:.2f}s',
    )
    print(
        "Mean V2I latency:",
        f'{latency["v2i_latency_ms"]["mean"]:.2f}ms',
    )
    print(
        "Brake trigger:",
        f'{braking["trigger"]["distance_m"]:.2f}m',
        f'TTC={braking["trigger"]["ttc_s"]:.2f}s',
    )
    print(
        "Safe stop:",
        f'{braking["stop"]["actual_braking_distance_m"]:.2f}m',
        f'clearance={braking["stop"]["final_clearance_m"]:.2f}m',
    )
    print("METRICS TEST PASSED")


if __name__ == "__main__":
    main()
