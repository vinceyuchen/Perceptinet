from __future__ import annotations

from typing import Any, Dict


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def derive_presentation_state(
    snapshot: Dict[str, Any],
    *,
    stale: bool = False,
) -> Dict[str, Any]:
    """
    Derive the small set of user-facing dashboard states from one snapshot.

    Driver Mode and Pro Mode should present the same semantic truth even when
    they use different layouts. In particular, stale transport data must not
    be presented as LIVE/HEALTHY merely because the last valid snapshot was
    healthy.
    """

    hazard = snapshot.get("hazard") or {}
    braking = snapshot.get("braking") or {}
    ego = snapshot.get("ego") or {}
    health = snapshot.get("system_health") or {}

    level = str(
        hazard.get("control_level", "NORMAL")
    ).upper()

    primary = hazard.get("primary") or None

    sources = (
        list(primary.get("sources") or [])
        if primary
        else []
    )

    cooperative = any(
        str(source).upper().startswith("RSU")
        for source in sources
    )

    speed_mps = _number(
        ego.get("speed_mps"),
        0.0,
    )

    target_raw = braking.get("target_speed_mps")
    target_mps = (
        None
        if target_raw is None
        else _number(target_raw, 0.0)
    )

    stopped = (
        speed_mps <= 0.05
        and (
            target_mps is None
            or target_mps <= 0.05
        )
    )

    if stopped:
        vehicle_action = "STOPPED"
    elif bool(braking.get("emergency")):
        vehicle_action = "EMERGENCY BRAKING"
    elif bool(braking.get("active")):
        vehicle_action = "SLOWING"
    else:
        vehicle_action = "DRIVING"

    raw_health = str(
        health.get("status", "UNKNOWN")
    ).upper()

    if stale:
        data_status = "DATA STALE"
        display_health = "DATA LIMITED"
        perception_text = "COOPERATIVE LIMITED"
    else:
        data_status = "LIVE"
        display_health = raw_health
        perception_text = (
            "COOPERATIVE ACTIVE"
            if cooperative
            else "ONBOARD"
        )

    # Fault injection status is separate from the freshness of the local
    # JSON feed. A fresh dashboard can still have degraded V2I communication.
    experiment = snapshot.get("experiment") or {}
    if experiment.get("scenario") == "rain_v2i" and not stale:
        data_status = "V2I DEGRADED"
        display_health = "DATA LIMITED"
        perception_text = "COOPERATIVE LIMITED"

    # Pedestrian contact is an observed outcome in EITHER test, rather
    # than a predetermined label assigned only to the rainy scene.
    if not stale and experiment.get("scenario") in ("clear", "rain_v2i"):
        if any(
            str(event.get("other_type", "")).startswith("walker.")
            for event in experiment.get("carla_collision_events", [])
        ):
            vehicle_action = "CARLA CONTACT"
        elif experiment.get("kinematic_contact_observed"):
            # Geometric contact is not a full physics-simulated crash.
            vehicle_action = "CONTACT (SIM)"

    if stale:
        footer_message = "COOPERATIVE DATA STALE"
    elif experiment.get("scenario") == "rain_v2i":
        footer_message = "HEAVY RAIN - SIMULATED V2I DEGRADATION"
    elif stopped and primary:
        class_name = str(
            primary.get("class_name") or "hazard"
        ).upper()
        footer_message = (
            f"VEHICLE STOPPED - {class_name} AHEAD"
        )
    elif stopped:
        footer_message = "VEHICLE STOPPED"
    elif level == "CRITICAL":
        footer_message = (
            "BRAKE NOW - IMMINENT COLLISION RISK"
        )
    elif level == "WARNING":
        footer_message = (
            "HAZARD AHEAD - VEHICLE SLOWING"
        )
    elif level == "CAUTION":
        footer_message = (
            "COOPERATIVE HAZARD DETECTED AHEAD"
        )
    else:
        footer_message = (
            "COOPERATIVE PERCEPTION ACTIVE"
            if cooperative
            else "ONBOARD PERCEPTION ACTIVE"
        )

    return {
        "hazard_level": level,
        "primary": primary,
        "sources": sources,
        "cooperative_active": cooperative,
        "vehicle_action": vehicle_action,
        "stopped": stopped,
        "data_status": data_status,
        "raw_health": raw_health,
        "display_health": display_health,
        "perception_text": perception_text,
        "footer_message": footer_message,
    }
