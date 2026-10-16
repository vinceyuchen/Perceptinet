"""Deterministic crossing *trigger* for a CARLA WalkerControl pedestrian.

Only Scenario 2 uses this controller. The walker itself moves via CARLA
physics/animation, not by teleporting transforms. A geometric footprint
overlap is diagnostic and is distinct from a CARLA collision-sensor event.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


def relative_position(
    ego_xy: Tuple[float, float],
    pedestrian_xy: Tuple[float, float],
    forward_xy: Tuple[float, float],
) -> Tuple[float, float]:
    """Longitudinal and left-handed lateral offsets in Ego path coordinates."""
    fx, fy = forward_xy
    norm = math.hypot(fx, fy)
    if norm < 1e-6:
        raise ValueError("Ego direction is zero")
    fx, fy = fx / norm, fy / norm
    dx = pedestrian_xy[0] - ego_xy[0]
    dy = pedestrian_xy[1] - ego_xy[1]
    return dx * fx + dy * fy, -dx * fy + dy * fx


@dataclass
class CrossingDecision:
    walking: bool
    direction_xy: Tuple[float, float]
    speed_mps: float
    longitudinal_m: float
    lateral_m: float


class PedestrianCrossing:
    """Start walking into Ego's lane when it is approximately 8 m away."""

    def __init__(
        self, *, trigger_distance_m: float = 8.5,
        walking_speed_mps: float = 1.05,
        finish_opposite_offset_m: float = 2.4,
    ):
        if min(trigger_distance_m, walking_speed_mps,
               finish_opposite_offset_m) <= 0:
            raise ValueError("Crossing parameters must be positive")
        self.trigger_distance_m = trigger_distance_m
        self.walking_speed_mps = walking_speed_mps
        self.finish_opposite_offset_m = finish_opposite_offset_m
        self.triggered = False
        self.finished = False
        self.original_side = 0

    def step(self, ego_xy, pedestrian_xy, forward_xy) -> CrossingDecision:
        longitudinal, lateral = relative_position(
            ego_xy, pedestrian_xy, forward_xy
        )
        if (not self.triggered and not self.finished
                and 0 < longitudinal <= self.trigger_distance_m):
            self.triggered = True
            self.original_side = 1 if lateral >= 0 else -1

        if (self.triggered
                and lateral * self.original_side <= -self.finish_opposite_offset_m):
            self.triggered = False
            self.finished = True

        fx, fy = forward_xy
        norm = math.hypot(fx, fy)
        fx, fy = fx / norm, fy / norm
        sign = -self.original_side if self.original_side else 0
        direction = (-fy * sign, fx * sign) if self.triggered else (0.0, 0.0)
        return CrossingDecision(
            self.triggered, direction,
            self.walking_speed_mps if self.triggered else 0.0,
            longitudinal, lateral,
        )


def kinematic_footprint_overlap(
    ego_xy, pedestrian_xy, ego_heading_xy,
    *, ego_half_length_m: float, ego_half_width_m: float,
    pedestrian_radius_m: float,
) -> bool:
    """2D envelope contact; NOT a measured physical CARLA collision."""
    if min(ego_half_length_m, ego_half_width_m, pedestrian_radius_m) <= 0:
        raise ValueError("Invalid bounding-box dimensions")
    longitudinal, lateral = relative_position(
        ego_xy, pedestrian_xy, ego_heading_xy
    )
    return (
        abs(longitudinal) <= ego_half_length_m + pedestrian_radius_m
        and abs(lateral) <= ego_half_width_m + pedestrian_radius_m
    )
