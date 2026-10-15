from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.comparison import (
    build_suite_summary,
    compare_pair,
)
from validation.runtime import (
    ValidationConfig,
    ValidationRunTracker,
)


def make_config(
    mode: str,
) -> ValidationConfig:
    return ValidationConfig(
        enabled=True,
        mode=mode,
        run_id=f"{mode}_test",
        headless=True,
        auto_exit=True,
        max_sim_seconds=12.0,
        near_miss_threshold_m=2.5,
        occlusion_release_distance_m=1.0,
        results_root=REPO_ROOT / "results" / "validation_test",
    )


def pipeline_metrics(
    *,
    rsu_lead_s=None,
):
    return {
        "detection": {
            "rsu_detection_lead_s": rsu_lead_s,
        },
        "latency": {
            "v2i_latency_ms": {
                "mean": 25.0,
                "p95": 29.0,
            },
            "end_to_end_latency_ms": {
                "mean": 45.0,
                "p95": 52.0,
            },
        },
    }


def build_baseline():
    tracker = ValidationRunTracker(
        make_config("baseline")
    )

    for frame in range(10):
        tracker.observe_frame(
            frame_id=frame,
            sim_time_s=frame * 0.05,
            ego_to_pedestrian_m=10.0 - frame * 0.9,
            truck_ahead_m=8.0 - frame * 0.8,
            onboard_detected=False,
            rsu_detected=True,
            ego_travelled_m=frame * 0.2,
            ego_speed_mps=4.0,
        )

    tracker.mark_system_stop(
        final_clearance_m=3.2,
        actual_braking_distance_m=1.4,
    )

    tracker.min_clearance_m = 3.2
    tracker.ego_travelled_m = 8.2

    return tracker.build_summary(
        pipeline_metrics=pipeline_metrics(),
    )


def build_cooperative():
    tracker = ValidationRunTracker(
        make_config("cooperative")
    )

    for frame in range(10):
        tracker.observe_frame(
            frame_id=frame,
            sim_time_s=frame * 0.05,
            ego_to_pedestrian_m=12.0 - frame * 0.6,
            truck_ahead_m=8.0 - frame * 0.8,
            onboard_detected=(frame >= 8),
            rsu_detected=(frame >= 1),
            ego_travelled_m=frame * 0.15,
            ego_speed_mps=4.0,
        )

    tracker.mark_brake_trigger(
        distance_m=10.5,
        ttc_s=2.8,
        speed_mps=4.0,
        hazard_level="WARNING",
    )
    tracker.mark_system_stop(
        final_clearance_m=5.4,
        actual_braking_distance_m=4.6,
    )

    tracker.min_clearance_m = 5.4
    tracker.ego_travelled_m = 6.8

    return tracker.build_summary(
        pipeline_metrics=pipeline_metrics(
            rsu_lead_s=0.8,
        ),
    )


def main():
    baseline = build_baseline()
    cooperative = build_cooperative()

    assert baseline["outcome"] == "LATE_LOCAL_STOP"
    assert cooperative["outcome"] == "SAFE_STOP"

    assert (
        cooperative["detection"][
            "rsu_occluded_detection_rate"
        ]
        >
        cooperative["detection"][
            "onboard_occluded_detection_rate"
        ]
    )

    pair = compare_pair(
        baseline,
        cooperative,
        pair_id="trial_01",
    )

    assert pair["pair_pass"] is True
    assert pair["clearance_gain_m"] > 0.0
    assert pair["travel_distance_saved_m"] > 0.0
    assert pair["cooperative_rsu_detection_lead_s"] == 0.8
    assert pair["baseline_failure_demonstrated"] is False

    suite = build_suite_summary([pair])

    assert suite["all_pairs_passed"] is True
    assert suite["passed_pairs"] == 1

    print("=== Perceptinet Final Validation v1 Test ===")
    print("Baseline outcome:", baseline["outcome"])
    print("Cooperative outcome:", cooperative["outcome"])
    print(
        "Clearance gain:",
        f"{pair['clearance_gain_m']:.2f} m",
    )
    print(
        "RSU detection lead:",
        f"{pair['cooperative_rsu_detection_lead_s']:.2f} s",
    )
    print("FINAL VALIDATION TEST PASSED")


if __name__ == "__main__":
    main()
