from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RiskConfig:
    """
    Perceptinet Risk Assessment v1.

    Thresholds are prototype/demo parameters only.
    They are not validated automotive safety limits.
    """

    lane_half_width_m: float = 2.5

    caution_ttc_s: float = 5.0
    warning_ttc_s: float = 3.0
    critical_ttc_s: float = 1.5

    minimum_closing_speed_mps: float = 0.1


class RiskEngine:

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    @staticmethod
    def _distance_2d(
        a: Dict[str, float],
        b: Dict[str, float],
    ) -> float:
        return math.hypot(
            b["x"] - a["x"],
            b["y"] - a["y"],
        )

    @staticmethod
    def _relative_coordinates(
        ego_position: Dict[str, float],
        object_position: Dict[str, float],
        ego_yaw_deg: float,
    ):
        dx = object_position["x"] - ego_position["x"]
        dy = object_position["y"] - ego_position["y"]

        yaw = math.radians(ego_yaw_deg)

        longitudinal = (
            dx * math.cos(yaw)
            + dy * math.sin(yaw)
        )

        lateral = (
            -dx * math.sin(yaw)
            + dy * math.cos(yaw)
        )

        return longitudinal, lateral

    def assess_object(
        self,
        ego_state: Dict[str, Any],
        obj: Dict[str, Any],
    ) -> Dict[str, Any]:

        ego_position = ego_state["world_position"]
        object_position = obj["world_position"]

        ego_speed = float(
            ego_state.get("speed_mps", 0.0)
        )

        ego_yaw = float(
            ego_state.get("yaw_deg", 0.0)
        )

        object_velocity = obj.get(
            "world_velocity",
            {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            },
        )

        longitudinal, lateral = (
            self._relative_coordinates(
                ego_position,
                object_position,
                ego_yaw,
            )
        )

        distance = self._distance_2d(
            ego_position,
            object_position,
        )

        yaw = math.radians(ego_yaw)

        object_forward_speed = (
            object_velocity.get("x", 0.0)
            * math.cos(yaw)
            +
            object_velocity.get("y", 0.0)
            * math.sin(yaw)
        )

        closing_speed = (
            ego_speed - object_forward_speed
        )

        in_path = (
            longitudinal > 0.0
            and abs(lateral)
            <= self.config.lane_half_width_m
        )

        ttc = None

        if (
            in_path
            and closing_speed
            > self.config.minimum_closing_speed_mps
        ):
            ttc = longitudinal / closing_speed

        hazard = "NORMAL"

        if ttc is not None:
            if ttc <= self.config.critical_ttc_s:
                hazard = "CRITICAL"

            elif ttc <= self.config.warning_ttc_s:
                hazard = "WARNING"

            elif ttc <= self.config.caution_ttc_s:
                hazard = "CAUTION"

        return {
            "class_name": obj["class_name"],
            "actor_id": obj.get("actor_id"),
            "sources": obj.get("sources", []),

            "distance_m": distance,
            "longitudinal_m": longitudinal,
            "lateral_m": lateral,

            "closing_speed_mps": closing_speed,
            "ttc_s": ttc,

            "in_path": in_path,
            "hazard_level": hazard,

            "world_position": object_position,
        }

    def assess_world(
        self,
        ego_state: Dict[str, Any],
        fused_world: Dict[str, Any],
    ) -> Dict[str, Any]:

        risks: List[Dict[str, Any]] = []

        for obj in fused_world.get("objects", []):
            risks.append(
                self.assess_object(
                    ego_state,
                    obj,
                )
            )

        priority = {
            "NORMAL": 0,
            "CAUTION": 1,
            "WARNING": 2,
            "CRITICAL": 3,
        }

        highest = "NORMAL"

        for risk in risks:
            if (
                priority[risk["hazard_level"]]
                > priority[highest]
            ):
                highest = risk["hazard_level"]

        return {
            "highest_hazard": highest,
            "risks": risks,
        }
