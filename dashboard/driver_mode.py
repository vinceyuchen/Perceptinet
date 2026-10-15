from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .presentation_state import derive_presentation_state
from .ui_theme import (
    BACKGROUND,
    BORDER,
    BORDER_SOFT,
    CAUTION,
    COOPERATIVE,
    CRITICAL,
    HEALTHY,
    SURFACE_1,
    SURFACE_2,
    SURFACE_3,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
    draw_navigation_marker,
    draw_object_marker,
    draw_text,
    rounded_rect,
    status_pill,
)


HAZARD_COLORS = {
    "NORMAL": TEXT_PRIMARY,
    "CAUTION": CAUTION,
    "WARNING": WARNING,
    "CRITICAL": CRITICAL,
}

BLUE = COOPERATIVE
GREEN = HEALTHY
WHITE = TEXT_PRIMARY
MUTED = TEXT_SECONDARY
PANEL = SURFACE_1
PANEL_LIGHT = SURFACE_2


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
        return str(
            derive_presentation_state(
                snapshot,
                stale=False,
            )["vehicle_action"]
        )

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
        draw_text(
            canvas,
            text,
            xy,
            scale=scale,
            color=color,
            thickness=thickness,
        )

    @staticmethod
    def _panel(
        canvas: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
    ) -> None:
        rounded_rect(
            canvas,
            p1,
            p2,
            fill=PANEL,
            border=BORDER_SOFT,
            border_thickness=1,
            radius=10,
        )

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
            "DRIVER MODE",
            (30, 55),
            scale=0.78,
            color=WHITE,
            thickness=2,
        )

        presentation = derive_presentation_state(
            snapshot,
            stale=stale,
        )

        status_text = str(
            presentation["data_status"]
        )

        status_color = GREEN if not stale else CAUTION

        status_pill(
            canvas,
            status_text,
            (785, 56),
            foreground=status_color,
            fill=SURFACE_2,
            border=BORDER_SOFT,
            scale=0.40,
            height=26,
            dot=True,
        )

        rounded_rect(
            canvas,
            (950, 23),
            (1245, 72),
            fill=SURFACE_3,
            border=MUTED if stale else color,
            border_thickness=1,
            radius=8,
        )
        cv2.rectangle(
            canvas,
            (950, 32),
            (955, 64),
            MUTED if stale else color,
            -1,
        )

        self._put_text(
            canvas,
            f"{level} / LAST KNOWN" if stale else level,
            (975, 55),
            scale=0.54,
            color=MUTED if stale else color,
            thickness=2,
        )

    def _draw_speed(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
        *,
        stale: bool = False,
    ) -> None:
        ego = snapshot.get("ego") or {}
        speed_mps = self._number(
            ego.get("speed_mps"),
            0.0,
        ) or 0.0

        speed_kmh = speed_mps * 3.6
        speed_text = f"{speed_kmh:.0f}"
        speed_scale = 2.12 if len(speed_text) <= 2 else 1.72
        (speed_width, _), _ = cv2.getTextSize(
            speed_text, cv2.FONT_HERSHEY_DUPLEX, speed_scale, 4
        )
        speed_x = max(55, 207 - speed_width)

        self._panel(
            canvas,
            (30, 115),
            (310, 300),
        )

        self._put_text(
            canvas,
            "LAST KNOWN SPEED" if stale else "CURRENT SPEED",
            (55, 150),
            scale=0.55,
            color=MUTED,
            thickness=1,
        )

        self._put_text(
            canvas,
            speed_text,
            (speed_x, 242),
            scale=speed_scale,
            color=MUTED if stale else WHITE,
            thickness=4,
        )

        self._put_text(
            canvas,
            "km/h",
            (219, 242),
            scale=0.58,
            color=MUTED,
            thickness=1,
        )

    def _draw_hazard_card(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
        *,
        stale: bool = False,
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
            "LAST KNOWN HAZARD" if stale else "CURRENT HAZARD",
            (55, 365),
            scale=0.55,
            color=MUTED,
            thickness=1,
        )

        if not primary:
            self._put_text(
                canvas,
                "NO RECENT DATA" if stale else "ROAD CLEAR",
                (55, 425),
                scale=0.9,
                color=MUTED if stale else GREEN,
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
            scale=0.86,
            color=MUTED if stale else color,
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
            (55, 503),
            scale=0.80,
            color=MUTED if stale else WHITE,
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
            (55, 577),
            scale=0.80,
            color=MUTED if stale else color,
            thickness=2,
        )

        if cooperative:
            # Attribution label, not an interactive control.
            attribution_color = MUTED if stale else BLUE
            cv2.circle(
                canvas, (60, 604), 4,
                attribution_color, -1, cv2.LINE_AA,
            )
            self._put_text(
                canvas,
                "RSU IN LAST SNAPSHOT" if stale else "RSU COOPERATIVE DETECTION",
                (73, 609),
                scale=0.36,
                color=attribution_color,
                thickness=1,
            )

    def _draw_spatial_view(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
        *,
        stale: bool = False,
    ) -> None:
        x1, y1, x2, y2 = 340, 115, 930, 625
        self._panel(canvas, (x1, y1), (x2, y2))

        self._put_text(
            canvas,
            "LAST KNOWN ROAD VIEW" if stale else "ROAD AHEAD",
            (365, 150),
            scale=0.59,
            color=MUTED if stale else WHITE,
            thickness=2,
        )
        self._put_text(
            canvas,
            "TOP-DOWN  /  EGO-RELATIVE SCHEMATIC",
            (365, 177),
            scale=0.34,
            color=MUTED,
            thickness=1,
        )

        # This is an ego-relative schematic, not a geocoded map or
        # a claim of calibrated lane geometry. Distance bands and
        # detected positions share the same 0-30 m plotting scale.
        center_x = (x1 + x2) // 2
        road_left, road_right = center_x - 125, center_x + 125
        road_top, road_bottom = 204, 604
        ego_y, forward_span = 568, 340

        rounded_rect(
            canvas,
            (road_left - 18, road_top),
            (road_right + 18, road_bottom),
            fill=(21, 21, 24),
            radius=15,
        )
        rounded_rect(
            canvas,
            (road_left, road_top),
            (road_right, road_bottom),
            fill=(34, 34, 38),
            border=(53, 53, 60),
            radius=12,
        )

        # Boundaries are illustrative. The center dashes indicate
        # the ego path, not a divider between separate lanes.
        for edge_x in (road_left + 12, road_right - 12):
            cv2.line(
                canvas,
                (edge_x, road_top + 18),
                (edge_x, road_bottom - 18),
                (90, 90, 98),
                2,
                cv2.LINE_AA,
            )

        for distance_m in (10, 20, 30):
            tick_y = int(ego_y - distance_m / 30.0 * forward_span)
            cv2.line(
                canvas,
                (road_left + 18, tick_y),
                (road_right - 18, tick_y),
                (48, 48, 53),
                1,
                cv2.LINE_AA,
            )
            self._put_text(
                canvas,
                f"{distance_m} m",
                (374, tick_y + 5),
                scale=0.36,
                color=MUTED,
                thickness=1,
            )

        for dash_y in range(road_top + 31, ego_y - 25, 48):
            cv2.line(
                canvas,
                (center_x, dash_y),
                (center_x, dash_y + 23),
                BLUE if not stale else (73, 73, 78),
                2,
                cv2.LINE_AA,
            )

        draw_navigation_marker(
            canvas,
            (center_x, ego_y),
            size=19,
            color=MUTED if stale else WHITE,
            heading_deg=0.0,
            halo=True,
        )
        self._put_text(
            canvas,
            "EGO",
            (center_x - 18, ego_y + 38),
            scale=0.34,
            color=MUTED,
            thickness=1,
        )

        primary = (snapshot.get("hazard") or {}).get("primary")
        if not primary:
            self._put_text(
                canvas,
                "NO RECENT TARGET DATA" if stale else "NO IMMEDIATE HAZARD",
                (center_x - 105, 279),
                scale=0.43,
                color=MUTED if stale else GREEN,
                thickness=1,
            )
            return

        forward = self._number(primary.get("longitudinal_m"))
        lateral = self._number(primary.get("lateral_m"))
        if forward is None or lateral is None or not all(
            math.isfinite(value) for value in (forward, lateral)
        ):
            self._put_text(
                canvas,
                "TARGET POSITION UNAVAILABLE",
                (center_x - 132, 279),
                scale=0.43,
                color=MUTED,
                thickness=1,
            )
            return

        # Do not place off-scale objects at misleading map-edge coordinates.
        if forward < 0.0 or forward > 30.0 or abs(lateral) > 4.0:
            self._put_text(
                canvas,
                "TARGET OUTSIDE CURRENT VIEW",
                (center_x - 140, 279),
                scale=0.43,
                color=MUTED,
                thickness=1,
            )
            return

        hazard_y = int(ego_y - forward / 30.0 * forward_span)
        hazard_x = int(center_x + lateral / 4.0 * 110)
        hazard_level = str(
            (snapshot.get("hazard") or {}).get("control_level", "NORMAL")
        ).upper()
        color = MUTED if stale else self._hazard_color(hazard_level)
        class_name = str(primary.get("class_name") or "object").upper()

        draw_object_marker(
            canvas,
            (hazard_x, hazard_y),
            class_name,
            color=color,
            size=22 if hazard_level == "CRITICAL" else 19,
            primary=True,
        )
        (text_width, _), _ = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_DUPLEX, 0.49, 2
        )
        label_x = hazard_x + 30
        if label_x + text_width > x2 - 18:
            label_x = hazard_x - text_width - 30
        label_x = max(x1 + 16, label_x)
        label_y = max(road_top + 36, min(road_bottom - 40, hazard_y - 4))
        self._put_text(
            canvas,
            class_name,
            (label_x, label_y),
            scale=0.49,
            color=color,
            thickness=2,
        )

        sources = primary.get("sources") or []
        source_name = str(sources[0]).upper() if sources else ""
        if source_name:
            self._put_text(
                canvas,
                source_name[:18],
                (label_x, label_y + 20),
                scale=0.35,
                color=MUTED if stale else BLUE,
                thickness=1,
            )

    def _draw_action_panel(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
        *,
        stale: bool = False,
    ) -> None:
        self._panel(
            canvas,
            (960, 115),
            (1250, 625),
        )

        hazard = snapshot.get("hazard") or {}
        braking = snapshot.get("braking") or {}
        presentation = derive_presentation_state(
            snapshot,
            stale=stale,
        )

        level = str(
            hazard.get("control_level", "NORMAL")
        ).upper()
        color = self._hazard_color(level)

        self._put_text(
            canvas,
            "LAST KNOWN RESPONSE" if stale else "VEHICLE RESPONSE",
            (985, 150),
            scale=0.52,
            color=MUTED,
            thickness=1,
        )

        action = str(
            presentation["vehicle_action"]
        )

        action_color = (
            GREEN
            if action in ("DRIVING", "STOPPED")
            else color
        )

        # Fit long states such as EMERGENCY BRAKING within the panel.
        action_scale = 0.64
        (action_width, _), _ = cv2.getTextSize(
            action, cv2.FONT_HERSHEY_DUPLEX, action_scale, 2
        )
        if action_width > 238:
            action_scale *= 238.0 / action_width

        self._put_text(
            canvas,
            action,
            (985, 211),
            scale=action_scale,
            color=MUTED if stale else action_color,
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
                f"{target_speed * 3.6:.1f} km/h",
                (985, 302),
                scale=0.76,
                color=MUTED if stale else WHITE,
                thickness=2,
            )

        cv2.line(
            canvas, (985, 336), (1225, 336),
            BORDER_SOFT, 1, cv2.LINE_AA,
        )
        self._put_text(
            canvas,
            "PERCEPTION",
            (985, 367),
            scale=0.42,
            color=MUTED,
            thickness=1,
        )

        cooperative = bool(
            presentation["cooperative_active"]
        )

        perception_text = str(
            presentation["perception_text"]
        )

        self._put_text(
            canvas,
            perception_text,
            (985, 402),
            scale=0.5,
            color=MUTED if stale else BLUE if cooperative else WHITE,
            thickness=2,
        )

        # Attribution uses the primary hazard's existing source metadata.
        # No inference, transport or fusion values are changed here.
        primary_sources = [
            str(source).upper()
            for source in (presentation["sources"] or [])
            if source
        ]
        source_text = (
            " + ".join(primary_sources)
            if primary_sources else "NO PRIMARY TARGET"
        )
        if len(source_text) > 25:
            source_text = source_text[:22] + "..."

        self._put_text(
            canvas,
            "TARGET SOURCES",
            (985, 442),
            scale=0.42,
            color=MUTED,
            thickness=1,
        )
        self._put_text(
            canvas,
            source_text,
            (985, 475),
            scale=0.45,
            color=MUTED if stale else BLUE if cooperative else WHITE,
            thickness=1,
        )

        health_status = str(
            presentation["display_health"]
        )

        cv2.line(
            canvas, (985, 503), (1225, 503),
            BORDER_SOFT, 1, cv2.LINE_AA,
        )
        self._put_text(
            canvas,
            "SYSTEM STATUS",
            (985, 543),
            scale=0.42,
            color=MUTED,
            thickness=1,
        )

        self._put_text(
            canvas,
            health_status,
            (985, 581),
            scale=0.58,
            color=(
                GREEN
                if health_status == "HEALTHY"
                else CAUTION
                if stale
                else color
            ),
            thickness=2,
        )

    def _draw_footer(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
        *,
        stale: bool = False,
    ) -> None:
        cv2.rectangle(
            canvas,
            (0, 663),
            (self.width, self.height),
            PANEL_LIGHT,
            -1,
        )

        cv2.line(
            canvas, (0, 663), (self.width, 663),
            BORDER_SOFT, 1, cv2.LINE_AA,
        )

        presentation = derive_presentation_state(
            snapshot,
            stale=stale,
        )

        level = str(
            presentation["hazard_level"]
        )
        message = str(
            presentation["footer_message"]
        )

        footer_color = (
            CAUTION if stale else self._hazard_color(level)
        )
        cv2.rectangle(
            canvas,
            (30, 675),
            (34, 705),
            footer_color,
            -1,
        )
        self._put_text(
            canvas,
            message,
            (50, 701),
            scale=0.61,
            color=footer_color,
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
        self._draw_speed(canvas, snapshot, stale=stale)
        self._draw_hazard_card(canvas, snapshot, stale=stale)
        self._draw_spatial_view(canvas, snapshot, stale=stale)
        self._draw_action_panel(
            canvas,
            snapshot,
            stale=stale,
        )
        self._draw_footer(
            canvas,
            snapshot,
            stale=stale,
        )

        return canvas
