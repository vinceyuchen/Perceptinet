from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BrakingConfig:
    """
    Perceptinet deterministic braking controller v1.

    Prototype/demo parameters only.
    They are not validated production vehicle braking limits.
    """

    # CAUTION is advisory only in the cooperative
    # braking demo. Automatic deceleration begins at WARNING.
    caution_decel_mps2: float = 0.0
    warning_decel_mps2: float = 1.5
    critical_decel_mps2: float = 6.0

    minimum_speed_mps: float = 0.0


class BrakingController:
    """
    Converts Risk Engine hazard levels into deterministic
    kinematic braking commands.

    NORMAL:
        no braking

    CAUTION:
        advisory only, no automatic braking

    WARNING:
        controlled progressive braking strong enough to
        exploit early cooperative warning without allowing
        prolonged low-deceleration creep

    CRITICAL:
        emergency braking
    """

    def __init__(
        self,
        config: BrakingConfig | None = None,
    ):
        self.config = config or BrakingConfig()

    def deceleration_for_hazard(
        self,
        hazard_level: str,
    ) -> float:

        if hazard_level == "CAUTION":
            return self.config.caution_decel_mps2

        if hazard_level == "WARNING":
            return self.config.warning_decel_mps2

        if hazard_level == "CRITICAL":
            return self.config.critical_decel_mps2

        return 0.0

    def stopping_distance(
        self,
        speed_mps: float,
        deceleration_mps2: float,
    ) -> float | None:
        """
        Ideal kinematic braking distance:

            d = v^2 / (2a)

        Does not include driver/system reaction delay.
        """

        if deceleration_mps2 <= 0.0:
            return None

        return (
            speed_mps ** 2
            / (2.0 * deceleration_mps2)
        )

    def update(
        self,
        current_speed_mps: float,
        hazard_level: str,
        dt_s: float,
    ) -> Dict[str, Any]:

        current_speed_mps = max(
            self.config.minimum_speed_mps,
            float(current_speed_mps),
        )

        dt_s = max(0.0, float(dt_s))

        deceleration = self.deceleration_for_hazard(
            hazard_level
        )

        target_speed = max(
            self.config.minimum_speed_mps,
            current_speed_mps
            - deceleration * dt_s,
        )

        braking_active = deceleration > 0.0

        return {
            "hazard_level": hazard_level,

            "speed_before_mps": current_speed_mps,
            "target_speed_mps": target_speed,

            "deceleration_mps2": deceleration,

            "braking_active": braking_active,
            "emergency_braking": (
                hazard_level == "CRITICAL"
                and braking_active
            ),

            "estimated_stopping_distance_m":
                self.stopping_distance(
                    current_speed_mps,
                    deceleration,
                ),
        }
