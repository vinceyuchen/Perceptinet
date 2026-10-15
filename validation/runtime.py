from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)

    if value is None:
        return bool(default)

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


@dataclass(frozen=True)
class ValidationConfig:
    """
    Final-system validation configuration.

    These thresholds are prototype/demo evaluation parameters only.
    They are not production automotive safety limits.
    """

    enabled: bool
    mode: str
    run_id: str
    headless: bool
    auto_exit: bool
    max_sim_seconds: float
    near_miss_threshold_m: float
    occlusion_release_distance_m: float
    results_root: Path

    @property
    def cooperative_enabled(self) -> bool:
        # Outside validation mode, preserve the original cooperative demo.
        if not self.enabled:
            return True

        return self.mode == "cooperative"

    @property
    def run_dir(self) -> Path:
        return self.results_root / self.run_id

    @classmethod
    def from_environment(
        cls,
        repo_root: Path,
    ) -> "ValidationConfig":
        enabled = _env_bool(
            "PERCEPTINET_VALIDATION",
            False,
        )

        mode = os.environ.get(
            "PERCEPTINET_VALIDATION_MODE",
            "cooperative",
        ).strip().lower()

        if mode not in {"baseline", "cooperative"}:
            raise ValueError(
                "PERCEPTINET_VALIDATION_MODE must be "
                "'baseline' or 'cooperative'"
            )

        run_id = os.environ.get(
            "PERCEPTINET_VALIDATION_RUN_ID",
            f"{mode}_manual",
        ).strip()

        if not run_id:
            run_id = f"{mode}_manual"

        root_value = os.environ.get(
            "PERCEPTINET_VALIDATION_RESULTS_ROOT"
        )

        results_root = (
            Path(root_value)
            if root_value
            else repo_root / "results" / "validation"
        )

        return cls(
            enabled=enabled,
            mode=mode,
            run_id=run_id,
            headless=_env_bool(
                "PERCEPTINET_VALIDATION_HEADLESS",
                enabled,
            ),
            auto_exit=_env_bool(
                "PERCEPTINET_VALIDATION_AUTO_EXIT",
                enabled,
            ),
            max_sim_seconds=float(
                os.environ.get(
                    "PERCEPTINET_VALIDATION_MAX_SIM_SECONDS",
                    "12.0",
                )
            ),
            near_miss_threshold_m=float(
                os.environ.get(
                    "PERCEPTINET_VALIDATION_NEAR_MISS_M",
                    "2.5",
                )
            ),
            occlusion_release_distance_m=float(
                os.environ.get(
                    "PERCEPTINET_VALIDATION_OCCLUSION_RELEASE_M",
                    "1.0",
                )
            ),
            results_root=results_root,
        )


class ValidationRunTracker:
    """
    Evaluation-only run tracker.

    Raw CARLA actor distance is used here only as evaluation ground truth and
    as a simulation guard. It is deliberately not fed into the RiskEngine or
    BrakingController decision path.
    """

    def __init__(self, config: ValidationConfig):
        self.config = config

        self.start_sim_time_s: Optional[float] = None
        self.last_sim_time_s: Optional[float] = None
        self.last_frame_id: Optional[int] = None

        self.total_frames = 0
        self.occluded_frames = 0

        self.onboard_detection_frames = 0
        self.rsu_detection_frames = 0
        self.onboard_occluded_detection_frames = 0
        self.rsu_occluded_detection_frames = 0

        self.min_clearance_m: Optional[float] = None
        self.ego_travelled_m = 0.0
        self.final_speed_mps: Optional[float] = None

        self.motion_guard_stop = False
        self.motion_guard_distance_m: Optional[float] = None

        self.brake_triggered = False
        self.brake_trigger_distance_m: Optional[float] = None
        self.brake_trigger_ttc_s: Optional[float] = None
        self.brake_trigger_speed_mps: Optional[float] = None
        self.brake_trigger_hazard: Optional[str] = None

        self.system_stop = False
        self.final_clearance_m: Optional[float] = None
        self.actual_braking_distance_m: Optional[float] = None

        self.timed_out = False

    @staticmethod
    def _rate(
        numerator: int,
        denominator: int,
    ) -> Optional[float]:
        if denominator <= 0:
            return None

        return float(numerator) / float(denominator)

    def observe_frame(
        self,
        *,
        frame_id: int,
        sim_time_s: float,
        ego_to_pedestrian_m: float,
        truck_ahead_m: float,
        onboard_detected: bool,
        rsu_detected: bool,
        ego_travelled_m: float,
        ego_speed_mps: float,
    ) -> None:
        sim_time = float(sim_time_s)

        if self.start_sim_time_s is None:
            self.start_sim_time_s = sim_time

        self.last_sim_time_s = sim_time
        self.last_frame_id = int(frame_id)

        self.total_frames += 1
        self.ego_travelled_m = float(ego_travelled_m)
        self.final_speed_mps = float(ego_speed_mps)

        distance = _number(ego_to_pedestrian_m)

        if distance is not None:
            if (
                self.min_clearance_m is None
                or distance < self.min_clearance_m
            ):
                self.min_clearance_m = distance

        if onboard_detected:
            self.onboard_detection_frames += 1

        if rsu_detected:
            self.rsu_detection_frames += 1

        occluded = (
            float(truck_ahead_m)
            > self.config.occlusion_release_distance_m
        )

        if occluded:
            self.occluded_frames += 1

            if onboard_detected:
                self.onboard_occluded_detection_frames += 1

            if rsu_detected:
                self.rsu_occluded_detection_frames += 1

    def mark_motion_guard(
        self,
        *,
        distance_m: Any,
    ) -> None:
        self.motion_guard_stop = True
        self.motion_guard_distance_m = _number(distance_m)

    def mark_brake_trigger(
        self,
        *,
        distance_m: Any,
        ttc_s: Any,
        speed_mps: Any,
        hazard_level: str,
    ) -> None:
        if self.brake_triggered:
            return

        self.brake_triggered = True
        self.brake_trigger_distance_m = _number(distance_m)
        self.brake_trigger_ttc_s = _number(ttc_s)
        self.brake_trigger_speed_mps = _number(speed_mps)
        self.brake_trigger_hazard = str(hazard_level)

    def mark_system_stop(
        self,
        *,
        final_clearance_m: Any,
        actual_braking_distance_m: Any,
    ) -> None:
        self.system_stop = True
        self.final_clearance_m = _number(final_clearance_m)
        self.actual_braking_distance_m = _number(
            actual_braking_distance_m
        )
        self.final_speed_mps = 0.0

    def mark_timeout(self) -> None:
        self.timed_out = True

    def reached_time_limit(
        self,
        sim_time_s: float,
    ) -> bool:
        if self.start_sim_time_s is None:
            return False

        return (
            float(sim_time_s) - self.start_sim_time_s
            >= self.config.max_sim_seconds
        )

    def _outcome(self) -> str:
        threshold = self.config.near_miss_threshold_m

        if self.config.mode == "cooperative":
            if (
                self.system_stop
                and self.final_clearance_m is not None
                and self.final_clearance_m > threshold
            ):
                return "SAFE_STOP"

            if self.system_stop:
                return "STOPPED_LOW_CLEARANCE"

            if self.motion_guard_stop:
                return "MOTION_GUARD_INTERVENTION"

            if self.timed_out:
                return "TIMEOUT"

            return "INCOMPLETE"

        # Baseline is the control/failure-mode run. A small minimum clearance
        # demonstrates the blind-spot problem rather than a validation success.
        if (
            self.min_clearance_m is not None
            and self.min_clearance_m <= threshold
        ):
            return "NEAR_MISS"

        if self.system_stop:
            return "LATE_LOCAL_STOP"

        if self.motion_guard_stop:
            return "MOTION_GUARD_INTERVENTION"

        if self.timed_out:
            return "TIMEOUT"

        return "INCOMPLETE"

    def build_summary(
        self,
        *,
        pipeline_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        elapsed_s = None

        if (
            self.start_sim_time_s is not None
            and self.last_sim_time_s is not None
        ):
            elapsed_s = (
                self.last_sim_time_s
                - self.start_sim_time_s
            )

        threshold = self.config.near_miss_threshold_m

        cooperative_safe_stop = (
            self.config.mode == "cooperative"
            and self.system_stop
            and self.final_clearance_m is not None
            and self.final_clearance_m > threshold
        )

        baseline_failure_demonstrated = (
            self.config.mode == "baseline"
            and self.min_clearance_m is not None
            and self.min_clearance_m <= threshold
        )

        return {
            "schema_version": "final-validation-v1",
            "run_id": self.config.run_id,
            "mode": self.config.mode,
            "cooperative_enabled": (
                self.config.cooperative_enabled
            ),
            "outcome": self._outcome(),
            "elapsed_sim_s": elapsed_s,
            "last_frame_id": self.last_frame_id,
            "detection": {
                "total_frames": self.total_frames,
                "occluded_frames": self.occluded_frames,
                "onboard_detection_frames":
                    self.onboard_detection_frames,
                "rsu_detection_frames":
                    self.rsu_detection_frames,
                "onboard_occluded_detection_frames":
                    self.onboard_occluded_detection_frames,
                "rsu_occluded_detection_frames":
                    self.rsu_occluded_detection_frames,
                "onboard_detection_rate": self._rate(
                    self.onboard_detection_frames,
                    self.total_frames,
                ),
                "rsu_detection_rate": self._rate(
                    self.rsu_detection_frames,
                    self.total_frames,
                ),
                "onboard_occluded_detection_rate": self._rate(
                    self.onboard_occluded_detection_frames,
                    self.occluded_frames,
                ),
                "rsu_occluded_detection_rate": self._rate(
                    self.rsu_occluded_detection_frames,
                    self.occluded_frames,
                ),
            },
            "safety": {
                "near_miss_threshold_m": threshold,
                "minimum_clearance_m": self.min_clearance_m,
                "final_clearance_m": self.final_clearance_m,
                "ego_travelled_m": self.ego_travelled_m,
                "final_speed_mps": self.final_speed_mps,
                "motion_guard_stop": self.motion_guard_stop,
                "motion_guard_distance_m":
                    self.motion_guard_distance_m,
                "brake_triggered": self.brake_triggered,
                "brake_trigger_distance_m":
                    self.brake_trigger_distance_m,
                "brake_trigger_ttc_s":
                    self.brake_trigger_ttc_s,
                "brake_trigger_speed_mps":
                    self.brake_trigger_speed_mps,
                "brake_trigger_hazard":
                    self.brake_trigger_hazard,
                "system_stop": self.system_stop,
                "actual_braking_distance_m":
                    self.actual_braking_distance_m,
                "cooperative_safe_stop":
                    cooperative_safe_stop,
                "baseline_failure_demonstrated":
                    baseline_failure_demonstrated,
            },
            "pipeline_metrics": pipeline_metrics or {},
            "evaluation_notes": {
                "ground_truth_usage":
                    "CARLA actor distance is evaluation-only and "
                    "does not drive RiskEngine/BrakingController.",
                "occlusion_window":
                    "Frame counted as occluded while truck centre "
                    f"is > {self.config.occlusion_release_distance_m:.2f} m "
                    "ahead of Ego along the frozen scenario path.",
                "threshold_note":
                    "Prototype/demo threshold only; not a validated "
                    "automotive safety limit.",
            },
        }

    def export_summary_json(
        self,
        path: Any,
        *,
        pipeline_metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                self.build_summary(
                    pipeline_metrics=pipeline_metrics,
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        return output_path
