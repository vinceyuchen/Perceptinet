from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


HAZARD_PRIORITY = {
    "NORMAL": 0,
    "CAUTION": 1,
    "WARNING": 2,
    "CRITICAL": 3,
}

VALID_HAZARDS = set(HAZARD_PRIORITY)
VALID_HEALTH = {"HEALTHY", "DEGRADED"}


def _number(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    if value is None:
        return default

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(number):
        return default

    return number


class DashboardStateValidator:
    """
    Semantic consistency checks for dashboard-v1 snapshots.

    This validates that the UI data layer is internally coherent. It does not
    judge visual styling and it does not replace RiskEngine or controller
    tests.
    """

    def __init__(
        self,
        *,
        cooperative_expected: Optional[bool] = None,
    ):
        self.cooperative_expected = cooperative_expected

    def validate(
        self,
        snapshot: Dict[str, Any],
        *,
        previous: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        errors: List[str] = []
        warnings: List[str] = []

        if snapshot.get("schema_version") != "dashboard-v1":
            errors.append("schema_version must be dashboard-v1")

        frame_id = snapshot.get("frame_id")
        sim_time_s = _number(snapshot.get("sim_time_s"))

        if frame_id is None:
            errors.append("frame_id missing")

        if sim_time_s is None:
            errors.append("sim_time_s missing/invalid")

        if previous is not None:
            previous_frame = previous.get("frame_id")
            previous_time = _number(previous.get("sim_time_s"))

            if (
                frame_id is not None
                and previous_frame is not None
                and int(frame_id) <= int(previous_frame)
            ):
                errors.append(
                    "frame_id did not increase"
                )

            if (
                sim_time_s is not None
                and previous_time is not None
                and sim_time_s < previous_time
            ):
                errors.append(
                    "sim_time_s moved backwards"
                )

        health = snapshot.get("system_health") or {}
        health_status = str(
            health.get("status", "")
        ).upper()
        health_issues = list(
            health.get("issues") or []
        )

        if health_status not in VALID_HEALTH:
            errors.append(
                f"invalid system health: {health_status}"
            )

        if (
            health_status == "HEALTHY"
            and health_issues
        ):
            errors.append(
                "HEALTHY snapshot has health issues"
            )

        if (
            health_status == "DEGRADED"
            and not health_issues
        ):
            errors.append(
                "DEGRADED snapshot has no issue reason"
            )

        hazard = snapshot.get("hazard") or {}
        observed = str(
            hazard.get("observed_level", "NORMAL")
        ).upper()
        control = str(
            hazard.get("control_level", "NORMAL")
        ).upper()

        if observed not in VALID_HAZARDS:
            errors.append(
                f"invalid observed hazard: {observed}"
            )

        if control not in VALID_HAZARDS:
            errors.append(
                f"invalid control hazard: {control}"
            )

        if (
            observed in VALID_HAZARDS
            and control in VALID_HAZARDS
            and HAZARD_PRIORITY[control]
            < HAZARD_PRIORITY[observed]
        ):
            errors.append(
                "control hazard is lower than observed hazard"
            )

        fusion = snapshot.get("fusion") or {}
        objects = list(
            fusion.get("objects") or []
        )

        try:
            object_count = int(
                fusion.get("object_count", len(objects))
            )
        except (TypeError, ValueError):
            object_count = -1

        if object_count != len(objects):
            errors.append(
                "fusion.object_count does not match objects length"
            )

        primary = hazard.get("primary")

        if primary:
            actor_id = primary.get("actor_id")
            sources = [
                str(source)
                for source in (
                    primary.get("sources") or []
                )
            ]

            if not sources:
                warnings.append(
                    "primary hazard has no source"
                )

            if actor_id is not None:
                fused_actor_ids = {
                    obj.get("actor_id")
                    for obj in objects
                }

                if actor_id not in fused_actor_ids:
                    errors.append(
                        "primary hazard actor missing from fused world"
                    )

            has_rsu_source = any(
                source.upper().startswith("RSU")
                for source in sources
            )

            if (
                has_rsu_source
                and self.cooperative_expected is not False
            ):
                v2i = _number(
                    (snapshot.get("latency_ms") or {}).get(
                        "v2i_latency_ms"
                    )
                )

                if v2i is None:
                    errors.append(
                        "RSU primary hazard has no V2I latency"
                    )

        braking = snapshot.get("braking") or {}
        ego = snapshot.get("ego") or {}

        speed = _number(
            ego.get("speed_mps"),
            0.0,
        )
        target = _number(
            braking.get("target_speed_mps")
        )
        decel = _number(
            braking.get("deceleration_mps2"),
            0.0,
        )
        active = bool(
            braking.get("active")
        )
        emergency = bool(
            braking.get("emergency")
        )
        state = str(
            braking.get("state", "")
        ).upper()

        if speed is None or speed < -1e-6:
            errors.append("invalid ego speed")

        if target is not None and target < -1e-6:
            errors.append("invalid target speed")

        if decel is None or decel < -1e-6:
            errors.append("invalid deceleration")

        if (
            active
            and speed is not None
            and target is not None
            and target > speed + 0.05
        ):
            errors.append(
                "active braking increases target speed"
            )

        if (
            active
            and control in {"WARNING", "CRITICAL"}
            and (decel or 0.0) <= 0.0
        ):
            errors.append(
                "WARNING/CRITICAL braking active with zero deceleration"
            )

        if emergency:
            if not active:
                errors.append(
                    "emergency braking is not active"
                )

            if control != "CRITICAL":
                errors.append(
                    "emergency braking without CRITICAL control hazard"
                )

        if state.startswith("STOPPED"):
            if (
                speed is not None
                and speed > 0.05
            ):
                errors.append(
                    "STOPPED state while ego speed is non-zero"
                )

            if (
                target is not None
                and target > 0.05
            ):
                errors.append(
                    "STOPPED state while target speed is non-zero"
                )

        latency = snapshot.get("latency_ms") or {}

        for key, value in latency.items():
            numeric = _number(value)

            if numeric is not None and numeric < 0.0:
                errors.append(
                    f"negative latency: {key}"
                )

        if self.cooperative_expected is True:
            v2i = _number(
                latency.get("v2i_latency_ms")
            )

            if (
                primary
                and any(
                    str(source).upper().startswith("RSU")
                    for source in (
                        primary.get("sources") or []
                    )
                )
                and v2i is None
            ):
                errors.append(
                    "cooperative snapshot missing live V2I latency"
                )

        return {
            "errors": errors,
            "warnings": warnings,
        }
