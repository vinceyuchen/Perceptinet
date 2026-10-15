from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


HAZARD_COLORS = {
    "NORMAL": (210, 210, 210),
    "CAUTION": (0, 190, 255),
    "WARNING": (0, 150, 255),
    "CRITICAL": (40, 40, 235),
}

BLUE = (235, 170, 60)
GREEN = (90, 210, 120)
WHITE = (240, 240, 240)
MUTED = (155, 155, 155)
PANEL = (24, 24, 24)
PANEL_LIGHT = (34, 34, 34)
BACKGROUND = (10, 10, 10)


class DriverModeRenderer:
    """
    Perceptinet Driver Mode v1.

    This renderer intentionally shows only driver-relevant information.
    Engineering metrics remain in Pro Mode.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
    ):
        self.width = int(width)
        self.height = int(height)

    @staticmethod
    def _number(
        value: Any,
        default: Optional[float] = None,
    ) -> Optional[float]:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _is_stopped(
        self,
        snapshot: Dict[str, Any],
    ) -> bool:
        ego = snapshot.get("ego") or {}
        braking = snapshot.get("braking") or {}

        speed_mps = self._number(
            ego.get("speed_mps"),
            0.0,
        ) or 0.0

        target_speed_mps = self._number(
            braking.get("target_speed_mps")
        )

        if target_speed_mps is None:
            return speed_mps <= 0.05

        return (
            speed_mps <= 0.05
            and target_speed_mps <= 0.05
        )

    def _vehicle_action(
        self,
        snapshot: Dict[str, Any],
    ) -> str:
        braking = snapshot.get("braking") or {}
        braking_state = str(
            braking.get("state", "DRIVING")
        )

        if self._is_stopped(snapshot):
            return "STOPPED"

        if braking.get("emergency"):
            return "EMERGENCY BRAKING"

        if braking.get("active"):
            return "SLOWING"

        if braking_state == "STOPPED_HOLD":
            return "STOPPED"

        return "DRIVING"

    @staticmethod
    def _hazard_color(level: str) -> Tuple[int, int, int]:
        return HAZARD_COLORS.get(
            str(level).upper(),
            HAZARD_COLORS["NORMAL"],
        )

    @staticmethod
    def _put_text(
        canvas: np.ndarray,
        text: str,
        xy: Tuple[int, int],
        *,
        scale: float = 0.7,
        color: Tuple[int, int, int] = WHITE,
        thickness: int = 2,
    ) -> None:
        cv2.putText(
            canvas,
            str(text),
            xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _panel(
        canvas: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
    ) -> None:
        cv2.rectangle(canvas, p1, p2, PANEL, -1)
        cv2.rectangle(canvas, p1, p2, (55, 55, 55), 1)

    @staticmethod
    def read_snapshot(
        path: Any,
        *,
        retries: int = 4,
        retry_delay_s: float = 0.01,
    ) -> Dict[str, Any]:
        snapshot_path = Path(path)

        last_error: Optional[Exception] = None

        for _ in range(max(1, retries)):
            try:
                return json.loads(
                    snapshot_path.read_text(
                        encoding="utf-8"
                    )
                )
            except (
                FileNotFoundError,
                json.JSONDecodeError,
                PermissionError,
            ) as exc:
                last_error = exc
                time.sleep(retry_delay_s)

        if last_error is not None:
            raise last_error

        raise RuntimeError("Unable to read dashboard snapshot")

    def _draw_header(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
        *,
        stale: bool,
    ) -> None:
        hazard = snapshot.get("hazard") or {}
        level = str(
            hazard.get("control_level", "NORMAL")
        ).upper()
        color = self._hazard_color(level)

        cv2.rectangle(
            canvas,
            (0, 0),
            (self.width, 86),
            PANEL,
            -1,
        )

        self._put_text(
            canvas,
            "PERCEPTINET",
            (30, 38),
            scale=0.82,
            color=WHITE,
            thickness=2,
        )

        self._put_text(
            canvas,
            "COOPERATIVE DRIVER ASSIST",
            (30, 66),
            scale=0.48,
            color=MUTED,
            thickness=1,
        )

        status_text = (
            "LIVE"
            if not stale
            else "DATA STALE"
        )

        status_color = GREEN if not stale else color

        self._put_text(
            canvas,
            status_text,
            (1035, 34),
            scale=0.62,
            color=status_color,
            thickness=2,
        )

        cv2.circle(
            canvas,
            (1015, 28),
            6,
            status_color,
            -1,
        )

        cv2.rectangle(
            canvas,
            (875, 48),
            (1245, 78),
            color,
            -1,
        )

        self._put_text(
            canvas,
            level,
            (895, 71),
            scale=0.65,
            color=(10, 10, 10),
            thickness=2,
        )

    def _draw_speed(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
    ) -> None:
        ego = snapshot.get("ego") or {}
        speed_mps = self._number(
            ego.get("speed_mps"),
            0.0,
        ) or 0.0

        speed_kmh = speed_mps * 3.6

        self._panel(
            canvas,
            (30, 115),
            (310, 300),
        )

        self._put_text(
            canvas,
            "SPEED",
            (55, 150),
            scale=0.55,
            color=MUTED,
            thickness=1,
        )

        self._put_text(
            canvas,
            f"{speed_kmh:4.0f}",
            (55, 235),
            scale=2.25,
            color=WHITE,
            thickness=4,
        )

        self._put_text(
            canvas,
            "km/h",
            (205, 235),
            scale=0.58,
            color=MUTED,
            thickness=1,
        )

        self._put_text(
            canvas,
            f"{speed_mps:.2f} m/s",
            (57, 275),
            scale=0.5,
            color=MUTED,
            thickness=1,
        )

    def _draw_hazard_card(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
    ) -> None:
        hazard = snapshot.get("hazard") or {}
        primary = hazard.get("primary")
        level = str(
            hazard.get("control_level", "NORMAL")
        ).upper()
        color = self._hazard_color(level)

        self._panel(
            canvas,
            (30, 325),
            (310, 625),
        )

        self._put_text(
            canvas,
            "CURRENT HAZARD",
            (55, 365),
            scale=0.55,
            color=MUTED,
            thickness=1,
        )

        if not primary:
            self._put_text(
                canvas,
                "ROAD CLEAR",
                (55, 425),
                scale=0.9,
                color=GREEN,
                thickness=2,
            )
            return

        class_name = str(
            primary.get("class_name") or "object"
        ).upper()

        distance_m = self._number(
            primary.get("distance_m")
        )

        ttc_s = self._number(
            primary.get("ttc_s")
        )

        sources = list(primary.get("sources") or [])
        cooperative = any(
            str(source).upper().startswith("RSU")
            for source in sources
        )

        self._put_text(
            canvas,
            class_name,
            (55, 420),
            scale=0.9,
            color=color,
            thickness=2,
        )

        distance_text = (
            "N/A"
            if distance_m is None
            else f"{distance_m:.1f} m"
        )

        ttc_text = (
            "N/A"
            if ttc_s is None
            else f"{ttc_s:.1f} s"
        )

        self._put_text(
            canvas,
            "DISTANCE",
            (55, 465),
            scale=0.45,
            color=MUTED,
            thickness=1,
        )
        self._put_text(
            canvas,
            distance_text,
            (55, 500),
            scale=0.78,
            color=WHITE,
            thickness=2,
        )

        self._put_text(
            canvas,
            "TTC",
            (55, 545),
            scale=0.45,
            color=MUTED,
            thickness=1,
        )
        self._put_text(
            canvas,
            ttc_text,
            (55, 580),
            scale=0.78,
            color=color,
            thickness=2,
        )

        if cooperative:
            cv2.rectangle(
                canvas,
                (55, 594),
                (280, 616),
                BLUE,
                -1,
            )
            self._put_text(
                canvas,
                "COOPERATIVE DETECTION",
                (65, 611),
                scale=0.38,
                color=(10, 10, 10),
                thickness=1,
            )

    def _draw_spatial_view(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
    ) -> None:
        x1, y1 = 340, 115
        x2, y2 = 930, 625

        self._panel(
            canvas,
            (x1, y1),
            (x2, y2),
        )

        self._put_text(
            canvas,
            "COOPERATIVE SPATIAL VIEW",
            (365, 150),
            scale=0.55,
            color=MUTED,
            thickness=1,
        )

        # Road corridor
        road_top_y = 190
        road_bottom_y = 600
        center_x = (x1 + x2) // 2

        cv2.line(
            canvas,
            (center_x - 145, road_bottom_y),
            (center_x - 55, road_top_y),
            (80, 80, 80),
            3,
        )
        cv2.line(
            canvas,
            (center_x + 145, road_bottom_y),
            (center_x + 55, road_top_y),
            (80, 80, 80),
            3,
        )
        cv2.line(
            canvas,
            (center_x, road_bottom_y),
            (center_x, road_top_y),
            (55, 55, 55),
            2,
        )

        for i in range(5):
            y = road_bottom_y - 65 - i * 72
            width = 26 - i * 3
            cv2.line(
                canvas,
                (center_x - width, y),
                (center_x + width, y),
                (85, 85, 85),
                2,
            )

        # Ego vehicle
        ego_y = road_bottom_y - 42
        cv2.rectangle(
            canvas,
            (center_x - 28, ego_y - 24),
            (center_x + 28, ego_y + 24),
            WHITE,
            2,
        )
        self._put_text(
            canvas,
            "EGO",
            (center_x - 19, ego_y + 6),
            scale=0.42,
            color=WHITE,
            thickness=1,
        )

        hazard = snapshot.get("hazard") or {}
        primary = hazard.get("primary")

        if not primary:
            self._put_text(
                canvas,
                "No immediate hazard",
                (center_x - 105, 220),
                scale=0.58,
                color=GREEN,
                thickness=2,
            )
            return

        level = str(
            hazard.get("control_level", "NORMAL")
        ).upper()
        color = self._hazard_color(level)

        longitudinal_m = self._number(
            primary.get("longitudinal_m")
        )
        lateral_m = self._number(
            primary.get("lateral_m")
        )

        if longitudinal_m is None:
            longitudinal_m = self._number(
                primary.get("distance_m"),
                0.0,
            ) or 0.0

        if lateral_m is None:
            lateral_m = 0.0

        long_clamped = max(
            0.0,
            min(30.0, longitudinal_m),
        )
        lat_clamped = max(
            -4.0,
            min(4.0, lateral_m),
        )

        hazard_y = int(
            road_bottom_y
            - 70
            - (long_clamped / 30.0) * 330
        )
        hazard_x = int(
            center_x
            + (lat_clamped / 4.0) * 120
        )

        radius = 22 if level != "CRITICAL" else 28

        cv2.circle(
            canvas,
            (hazard_x, hazard_y),
            radius,
            color,
            3,
        )
        cv2.circle(
            canvas,
            (hazard_x, hazard_y),
            6,
            color,
            -1,
        )

        self._put_text(
            canvas,
            str(
                primary.get("class_name")
                or "object"
            ).upper(),
            (hazard_x + 32, hazard_y + 6),
            scale=0.5,
            color=color,
            thickness=2,
        )

        cv2.line(
            canvas,
            (center_x, ego_y - 25),
            (hazard_x, hazard_y + radius),
            color,
            1,
        )

    def _draw_action_panel(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
    ) -> None:
        self._panel(
            canvas,
            (960, 115),
            (1250, 625),
        )

        hazard = snapshot.get("hazard") or {}
        braking = snapshot.get("braking") or {}
        health = snapshot.get("system_health") or {}

        level = str(
            hazard.get("control_level", "NORMAL")
        ).upper()
        color = self._hazard_color(level)

        self._put_text(
            canvas,
            "VEHICLE RESPONSE",
            (985, 150),
            scale=0.52,
            color=MUTED,
            thickness=1,
        )

        action = self._vehicle_action(snapshot)

        action_color = (
            GREEN
            if action in ("DRIVING", "STOPPED")
            else color
        )

        self._put_text(
            canvas,
            action,
            (985, 205),
            scale=0.64,
            color=action_color,
            thickness=2,
        )

        target_speed = self._number(
            braking.get("target_speed_mps")
        )

        if target_speed is not None:
            self._put_text(
                canvas,
                "TARGET SPEED",
                (985, 255),
                scale=0.42,
                color=MUTED,
                thickness=1,
            )
            self._put_text(
                canvas,
                f"{target_speed * 3.6:.0f} km/h",
                (985, 290),
                scale=0.72,
                color=WHITE,
                thickness=2,
            )

        primary = hazard.get("primary")
        sources = (
            list(primary.get("sources") or [])
            if primary
            else []
        )

        self._put_text(
            canvas,
            "PERCEPTION",
            (985, 350),
            scale=0.42,
            color=MUTED,
            thickness=1,
        )

        cooperative = any(
            str(source).upper().startswith("RSU")
            for source in sources
        )

        perception_text = (
            "RSU + ONBOARD"
            if cooperative
            else "ONBOARD"
        )

        if cooperative:
            perception_text = "COOPERATIVE ACTIVE"

        self._put_text(
            canvas,
            perception_text,
            (985, 385),
            scale=0.5,
            color=BLUE if cooperative else WHITE,
            thickness=2,
        )

        health_status = str(
            health.get("status", "UNKNOWN")
        )

        self._put_text(
            canvas,
            "SYSTEM",
            (985, 455),
            scale=0.42,
            color=MUTED,
            thickness=1,
        )

        self._put_text(
            canvas,
            health_status,
            (985, 490),
            scale=0.58,
            color=(
                GREEN
                if health_status == "HEALTHY"
                else color
            ),
            thickness=2,
        )

        if health_status != "HEALTHY":
            self._put_text(
                canvas,
                "Cooperative perception",
                (985, 535),
                scale=0.4,
                color=MUTED,
                thickness=1,
            )
            self._put_text(
                canvas,
                "limited",
                (985, 560),
                scale=0.4,
                color=MUTED,
                thickness=1,
            )

    def _draw_footer(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
    ) -> None:
        cv2.rectangle(
            canvas,
            (0, 655),
            (self.width, self.height),
            PANEL_LIGHT,
            -1,
        )

        hazard = snapshot.get("hazard") or {}
        level = str(
            hazard.get("control_level", "NORMAL")
        ).upper()

        primary = hazard.get("primary")
        stopped = self._is_stopped(snapshot)

        if stopped and primary:
            class_name = str(
                primary.get("class_name") or "hazard"
            ).upper()
            message = f"VEHICLE STOPPED - {class_name} AHEAD"
        elif stopped:
            message = "VEHICLE STOPPED"
        elif level == "CRITICAL":
            message = "BRAKE NOW - IMMINENT COLLISION RISK"
        elif level == "WARNING":
            message = "HAZARD AHEAD - VEHICLE SLOWING"
        elif level == "CAUTION":
            message = "COOPERATIVE HAZARD DETECTED AHEAD"
        else:
            message = "COOPERATIVE PERCEPTION ACTIVE"

        self._put_text(
            canvas,
            message,
            (35, 697),
            scale=0.72,
            color=self._hazard_color(level),
            thickness=2,
        )

    def render(
        self,
        snapshot: Dict[str, Any],
        *,
        stale: bool = False,
    ) -> np.ndarray:
        canvas = np.full(
            (self.height, self.width, 3),
            BACKGROUND,
            dtype=np.uint8,
        )

        self._draw_header(
            canvas,
            snapshot,
            stale=stale,
        )
        self._draw_speed(canvas, snapshot)
        self._draw_hazard_card(canvas, snapshot)
        self._draw_spatial_view(canvas, snapshot)
        self._draw_action_panel(canvas, snapshot)
        self._draw_footer(canvas, snapshot)

        return canvas
