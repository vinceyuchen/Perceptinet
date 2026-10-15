from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .presentation_state import derive_presentation_state
from .ui_theme import (
    BACKGROUND,
    BORDER_SOFT,
    CAUTION,
    COOPERATIVE,
    CRITICAL,
    GRID,
    HEALTHY,
    SURFACE_1,
    SURFACE_2,
    SURFACE_3,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
    badge,
    draw_navigation_marker,
    draw_object_marker,
    draw_text,
    rounded_rect,
    status_pill,
)


WHITE = TEXT_PRIMARY
MUTED = TEXT_SECONDARY
PANEL = SURFACE_1
PANEL_LIGHT = SURFACE_2
GREEN = HEALTHY
BLUE = COOPERATIVE
AMBER = CAUTION
RED = CRITICAL

HAZARD_COLORS = {
    "NORMAL": WHITE,
    "CAUTION": CAUTION,
    "WARNING": WARNING,
    "CRITICAL": CRITICAL,
}


class ProModeRenderer:
    """
    Perceptinet Pro Mode v1.

    Engineering/research view built from the same dashboard-v1 snapshot
    consumed by Driver Mode. The fused world model remains the primary view.
    """

    def __init__(
        self,
        width: int = 1600,
        height: int = 900,
    ):
        self.width = int(width)
        self.height = int(height)
        self.selected_source = "ALL"

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

    @staticmethod
    def _put_text(
        canvas: np.ndarray,
        text: str,
        xy: Tuple[int, int],
        *,
        scale: float = 0.55,
        color: Tuple[int, int, int] = WHITE,
        thickness: int = 1,
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
        retries: int = 6,
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
                PermissionError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                time.sleep(retry_delay_s)

        if last_error is not None:
            raise last_error

        raise RuntimeError("Unable to read dashboard snapshot")

    @staticmethod
    def available_sources(
        snapshot: Dict[str, Any],
    ) -> List[str]:
        sources = {"ALL"}

        for obj in (
            snapshot.get("fusion", {}).get("objects", [])
            or []
        ):
            for source in obj.get("sources", []) or []:
                if source:
                    sources.add(str(source))

        ordered = ["ALL"]

        if "onboard" in sources:
            ordered.append("onboard")

        ordered.extend(
            sorted(
                source
                for source in sources
                if source not in {"ALL", "onboard"}
            )
        )

        return ordered

    def cycle_source(
        self,
        snapshot: Dict[str, Any],
        step: int = 1,
    ) -> str:
        sources = self.available_sources(snapshot)

        if self.selected_source not in sources:
            self.selected_source = "ALL"

        index = sources.index(self.selected_source)
        self.selected_source = sources[
            (index + step) % len(sources)
        ]
        return self.selected_source

    @staticmethod
    def _hazard_color(level: str) -> Tuple[int, int, int]:
        return HAZARD_COLORS.get(
            str(level).upper(),
            WHITE,
        )

    @staticmethod
    def _object_sources(obj: Dict[str, Any]) -> List[str]:
        sources = (
            obj.get("display_sources")
            or obj.get("sources")
            or []
        )
        return [str(x) for x in sources]

    def _object_visible_for_source(
        self,
        obj: Dict[str, Any],
    ) -> bool:
        if self.selected_source == "ALL":
            return True
        return self.selected_source in self._object_sources(obj)

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
            (self.width, 82),
            PANEL_LIGHT,
            -1,
        )

        self._put_text(
            canvas,
            "ENGINEERING VIEW",
            (24, 36),
            scale=0.70,
            color=WHITE,
            thickness=2,
        )
        self._put_text(
            canvas,
            "FUSED WORLD / PIPELINE DIAGNOSTICS",
            (24, 62),
            scale=0.40,
            color=MUTED,
            thickness=1,
        )

        frame_id = snapshot.get("frame_id")
        sim_time = self._number(snapshot.get("sim_time_s"))

        self._put_text(
            canvas,
            f"FRAME {frame_id}",
            (340, 34),
            scale=0.52,
            color=MUTED,
        )
        self._put_text(
            canvas,
            (
                "SIM N/A"
                if sim_time is None
                else f"SIM {sim_time:.2f}s"
            ),
            (340, 60),
            scale=0.48,
            color=MUTED,
        )

        presentation = derive_presentation_state(
            snapshot,
            stale=stale,
        )

        health = str(
            presentation["display_health"]
        )

        cv2.circle(
            canvas,
            (610, 28),
            5,
            GREEN if health == "HEALTHY" else AMBER,
            -1,
            cv2.LINE_AA,
        )
        self._put_text(
            canvas,
            f"SYSTEM {health}",
            (625, 34),
            scale=0.48,
            color=MUTED if health == "HEALTHY" else AMBER,
            thickness=1 if health == "HEALTHY" else 2,
        )

        live_text = str(
            presentation["data_status"]
        )

        status_pill(
            canvas,
            live_text,
            (1100, 35),
            foreground=GREEN if not stale else AMBER,
            fill=SURFACE_2,
            border=BORDER_SOFT,
            scale=0.38,
            height=25,
            dot=True,
        )

        rounded_rect(
            canvas,
            (1325, 14),
            (1574, 64),
            fill=SURFACE_3,
            border=color,
            border_thickness=1,
            radius=9,
        )
        cv2.rectangle(
            canvas,
            (1325, 24),
            (1331, 54),
            color,
            -1,
        )
        self._put_text(
            canvas,
            level,
            (1350, 48),
            scale=0.70,
            color=color,
            thickness=2,
        )

    def _relative_xy(
        self,
        snapshot: Dict[str, Any],
        obj: Dict[str, Any],
    ) -> Optional[Tuple[float, float]]:
        ego = snapshot.get("ego") or {}
        ego_pos = ego.get("world_position") or {}
        obj_pos = (
            obj.get("display_world_position")
            or obj.get("world_position")
            or {}
        )

        ex = self._number(ego_pos.get("x"))
        ey = self._number(ego_pos.get("y"))
        ox = self._number(obj_pos.get("x"))
        oy = self._number(obj_pos.get("y"))
        yaw_deg = self._number(ego.get("yaw_deg"), 0.0) or 0.0

        if None in (ex, ey, ox, oy):
            return None

        dx = ox - ex
        dy = oy - ey
        yaw = math.radians(yaw_deg)

        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        right_x = -math.sin(yaw)
        right_y = math.cos(yaw)

        longitudinal = dx * forward_x + dy * forward_y
        lateral = dx * right_x + dy * right_y

        return longitudinal, lateral

    def _fused_view_limits(
        self,
        snapshot: Dict[str, Any],
    ) -> Tuple[float, float]:
        """
        Choose a useful engineering scale from the currently visible fused
        objects instead of always forcing a fixed 40 m / +/-10 m view.
        """
        longitudinal_values: List[float] = []
        lateral_values: List[float] = []

        for obj in (
            (snapshot.get("fusion") or {}).get("objects", [])
            or []
        ):
            if not self._object_visible_for_source(obj):
                continue

            rel = self._relative_xy(snapshot, obj)
            if rel is None:
                continue

            longitudinal, lateral = rel

            if longitudinal < -2.0:
                continue

            longitudinal_values.append(max(0.0, longitudinal))
            lateral_values.append(abs(lateral))

        farthest = max(longitudinal_values, default=10.0)
        widest = max(lateral_values, default=2.0)

        if farthest <= 10.0:
            forward_max = 15.0
        elif farthest <= 15.0:
            forward_max = 20.0
        elif farthest <= 25.0:
            forward_max = 30.0
        else:
            forward_max = 40.0

        lateral_half = max(
            4.0,
            min(
                10.0,
                math.ceil((widest + 1.0) / 2.0) * 2.0,
            ),
        )

        return forward_max, lateral_half

    @staticmethod
    def _first_detection_source(
        snapshot: Dict[str, Any],
    ) -> Optional[str]:
        detection = (
            (snapshot.get("metrics") or {}).get("detection")
            or {}
        )
        sources = detection.get("sources") or {}

        candidates: List[Tuple[float, str]] = []

        for source, details in sources.items():
            if not isinstance(details, dict):
                continue

            when = ProModeRenderer._number(
                details.get("first_stable_time_s")
            )

            if when is not None:
                candidates.append((when, str(source)))

        if candidates:
            return min(candidates, key=lambda item: item[0])[1]

        # Metrics summaries may be absent in lightweight/test snapshots while
        # the significant timeline still contains stable-detection events.
        # Use the earliest stable-detection event as the authoritative fallback.
        timeline_candidates: List[Tuple[float, int, str]] = []

        for event in snapshot.get("timeline") or []:
            if str(event.get("event_type")) != "stable_detection":
                continue

            source = event.get("source_id") or event.get("source")
            if not source:
                continue

            when = ProModeRenderer._number(
                event.get("sim_time_s"),
                float("inf"),
            )
            frame_id = event.get("frame_id")

            try:
                frame_order = int(frame_id)
            except (TypeError, ValueError):
                frame_order = 2**63 - 1

            timeline_candidates.append(
                (
                    float("inf") if when is None else when,
                    frame_order,
                    str(source),
                )
            )

        if timeline_candidates:
            return min(
                timeline_candidates,
                key=lambda item: (item[0], item[1]),
            )[2]

        # Backward-compatible fallback for Metrics v1.
        source = detection.get("earliest_rsu_source")
        return None if source is None else str(source)

    @staticmethod
    def _object_color(
        obj: Dict[str, Any],
        *,
        is_primary: bool,
        hazard_level: str,
    ) -> Tuple[int, int, int]:
        level = str(hazard_level).upper()

        if is_primary and level != "NORMAL":
            return HAZARD_COLORS.get(level, WHITE)

        sources = {
            str(source)
            for source in (obj.get("sources") or [])
        }

        if any(source.startswith("RSU") for source in sources):
            return BLUE

        return WHITE

    def _label_origin(
        self,
        *,
        px: int,
        py: int,
        label: str,
        sources: str,
        plot_left: int,
        plot_top: int,
        plot_right: int,
        plot_bottom: int,
        occupied: List[Tuple[int, int, int, int]],
    ) -> Tuple[int, int]:
        label_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            1,
        )
        source_size, _ = cv2.getTextSize(
            sources,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            1,
        )

        width = max(label_size[0], source_size[0]) + 10
        height = 32

        candidates = [
            (px + 15, py - 8),
            (px + 15, py + 30),
            (px - width - 15, py - 8),
            (px - width - 15, py + 30),
            (px + 15, py - 34),
            (px - width - 15, py - 34),
        ]

        for tx, ty in candidates:
            tx = max(plot_left + 4, min(tx, plot_right - width - 4))
            ty = max(plot_top + height, min(ty, plot_bottom - 4))

            rect = (
                tx - 4,
                ty - 20,
                tx + width,
                ty + 12,
            )

            overlaps = any(
                not (
                    rect[2] < other[0]
                    or rect[0] > other[2]
                    or rect[3] < other[1]
                    or rect[1] > other[3]
                )
                for other in occupied
            )

            if not overlaps:
                occupied.append(rect)
                return tx, ty

        tx = max(plot_left + 4, min(px + 15, plot_right - width - 4))
        ty = max(plot_top + height, min(py - 8, plot_bottom - 4))
        occupied.append((tx - 4, ty - 20, tx + width, ty + 12))
        return tx, ty

    def _draw_fused_world(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
    ) -> None:
        x1, y1, x2, y2 = 20, 102, 1030, 690
        self._panel(canvas, (x1, y1), (x2, y2))

        fused = snapshot.get("fusion") or {}
        objects = fused.get("objects", []) or []
        hazard = snapshot.get("hazard") or {}
        hazard_level = str(
            hazard.get("control_level", "NORMAL")
        ).upper()

        self._put_text(
            canvas,
            "FUSED WORLD MODEL",
            (45, 137),
            scale=0.62,
            thickness=2,
        )

        forward_max, lateral_half = self._fused_view_limits(
            snapshot
        )

        self._put_text(
            canvas,
            (
                f"{len(objects)} objects | "
                f"filter: {self.selected_source} | "
                f"view: 0-{forward_max:.0f} m / "
                f"+/-{lateral_half:.0f} m"
            ),
            (45, 165),
            scale=0.42,
            color=MUTED,
        )

        plot_left, plot_top = 60, 195
        plot_right, plot_bottom = 995, 655
        center_x = (plot_left + plot_right) // 2

        # Dynamic engineering grid.
        lateral_ticks = [
            -lateral_half,
            -lateral_half / 2.0,
            0.0,
            lateral_half / 2.0,
            lateral_half,
        ]

        for lateral in lateral_ticks:
            px = int(
                center_x
                + (lateral / lateral_half)
                * ((plot_right - plot_left) / 2.0)
            )
            cv2.line(
                canvas,
                (px, plot_top),
                (px, plot_bottom),
                GRID,
                1,
            )
            self._put_text(
                canvas,
                f"{lateral:+.0f}m",
                (px - 18, plot_bottom + 20),
                scale=0.34,
                color=MUTED,
            )

        for fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
            longitudinal = forward_max * fraction
            py = int(
                plot_bottom
                - fraction * (plot_bottom - plot_top)
            )
            cv2.line(
                canvas,
                (plot_left, py),
                (plot_right, py),
                GRID,
                1,
            )
            self._put_text(
                canvas,
                f"{longitudinal:.0f}m",
                (plot_left + 5, py - 5),
                scale=0.34,
                color=MUTED,
            )

        # Ego. Navigation-style marker keeps the engineering view directional
        # while removing the placeholder/debug-box appearance.
        ego_marker_y = plot_bottom - 9
        draw_navigation_marker(
            canvas,
            (center_x, ego_marker_y),
            size=13,
            color=WHITE,
            heading_deg=0.0,
            halo=True,
        )
        self._put_text(
            canvas,
            "EGO",
            (center_x + 22, ego_marker_y + 5),
            scale=0.32,
            color=MUTED,
            thickness=1,
        )

        primary = hazard.get("primary")
        primary_actor = (
            primary.get("actor_id")
            if primary
            else None
        )

        points: List[
            Tuple[
                float,
                float,
                int,
                int,
                Dict[str, Any],
                bool,
            ]
        ] = []

        for obj in objects:
            if not self._object_visible_for_source(obj):
                continue

            rel = self._relative_xy(snapshot, obj)
            if rel is None:
                continue

            longitudinal, lateral = rel

            if (
                longitudinal < 0.0
                or longitudinal > forward_max
                or abs(lateral) > lateral_half
            ):
                continue

            px = int(
                center_x
                + (lateral / lateral_half)
                * ((plot_right - plot_left) / 2.0)
            )
            py = int(
                plot_bottom
                - (longitudinal / forward_max)
                * (plot_bottom - plot_top)
            )

            is_primary = (
                primary_actor is not None
                and obj.get("actor_id") == primary_actor
            )

            points.append(
                (
                    longitudinal,
                    lateral,
                    px,
                    py,
                    obj,
                    is_primary,
                )
            )

        # Draw nearest objects last so labels remain readable.
        points.sort(key=lambda item: item[0], reverse=True)

        occupied: List[Tuple[int, int, int, int]] = []

        for (
            longitudinal,
            lateral,
            px,
            py,
            obj,
            is_primary,
        ) in points:
            color = self._object_color(
                obj,
                is_primary=is_primary,
                hazard_level=hazard_level,
            )
            class_name = str(
                obj.get("class_name") or "object"
            )

            draw_object_marker(
                canvas,
                (px, py),
                class_name,
                color=color,
                size=12 if is_primary else 10,
                primary=is_primary,
            )
            confidence = self._number(obj.get("confidence"))

            label = class_name
            if confidence is not None:
                label += f" {confidence:.2f}"

            sources = ",".join(
                self._object_sources(obj)
            )

            tx, ty = self._label_origin(
                px=px,
                py=py,
                label=label,
                sources=sources,
                plot_left=plot_left,
                plot_top=plot_top,
                plot_right=plot_right,
                plot_bottom=plot_bottom,
                occupied=occupied,
            )

            cv2.line(
                canvas,
                (px, py),
                (tx - 3, ty - 5),
                color,
                1,
            )

            self._put_text(
                canvas,
                label,
                (tx, ty),
                scale=0.38,
                color=color,
            )
            self._put_text(
                canvas,
                sources,
                (tx, ty + 15),
                scale=0.30,
                color=MUTED,
            )

        if not points:
            self._put_text(
                canvas,
                "No fused objects in current view/filter",
                (330, 430),
                scale=0.62,
                color=MUTED,
            )

    def _draw_source_inspector(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
    ) -> None:
        x1, y1, x2, y2 = 1050, 102, 1580, 315
        self._panel(canvas, (x1, y1), (x2, y2))

        self._put_text(
            canvas,
            "SOURCE INSPECTOR",
            (1075, 137),
            scale=0.6,
            thickness=2,
        )

        sources = self.available_sources(snapshot)
        primary = (
            (snapshot.get("hazard") or {}).get("primary")
        )
        primary_sources = (
            set(primary.get("sources") or [])
            if primary
            else set()
        )
        first_source = self._first_detection_source(snapshot)

        y = 175

        for source in sources:
            if source == "ALL":
                count = int(
                    (snapshot.get("fusion") or {}).get(
                        "object_count",
                        0,
                    )
                )
            else:
                count = sum(
                    source in self._object_sources(obj)
                    for obj in (
                        (snapshot.get("fusion") or {}).get(
                            "objects",
                            [],
                        )
                        or []
                    )
                )

            selected = source == self.selected_source
            contributor = source in primary_sources

            if selected:
                rounded_rect(
                    canvas,
                    (1070, y - 20),
                    (1560, y + 8),
                    fill=SURFACE_3,
                    radius=7,
                )

            prefix = ">" if selected else " "
            text = f"{prefix} {source:<10} {count} object{'s' if count != 1 else ''}"

            self._put_text(
                canvas,
                text,
                (1080, y),
                scale=0.46,
                color=BLUE if selected else WHITE,
                thickness=2 if selected else 1,
            )

            badge_x = 1315

            if source == first_source:
                rect = badge(
                    canvas,
                    "FIRST DETECTION",
                    (badge_x, y),
                    fill=AMBER,
                    foreground=(24, 24, 26),
                    scale=0.29,
                )
                badge_x = rect[2] + 7

            if contributor:
                badge(
                    canvas,
                    "CONTRIBUTOR",
                    (badge_x, y),
                    fill=BLUE,
                    foreground=(24, 24, 26),
                    scale=0.29,
                )

            y += 32

        self._put_text(
            canvas,
            "TAB: cycle source filter",
            (1080, 292),
            scale=0.38,
            color=MUTED,
        )

    def _draw_latency(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
    ) -> None:
        x1, y1, x2, y2 = 1050, 332, 1580, 550
        self._panel(canvas, (x1, y1), (x2, y2))

        self._put_text(
            canvas,
            "LIVE PIPELINE LATENCY",
            (1075, 366),
            scale=0.58,
            thickness=2,
        )

        latency = snapshot.get("latency_ms") or {}

        fields = [
            ("Onboard inference", "onboard_inference_ms"),
            ("RSU inference", "rsu_inference_ms"),
            ("V2I", "v2i_latency_ms"),
            ("Fusion", "fusion_latency_ms"),
            ("Risk", "risk_latency_ms"),
            ("Control", "control_latency_ms"),
            ("End-to-end", "end_to_end_latency_ms"),
        ]

        y = 397

        for label, key in fields:
            value = self._number(latency.get(key))
            value_text = "N/A" if value is None else f"{value:.2f} ms"

            if key == "end_to_end_latency_ms":
                cv2.line(
                    canvas,
                    (1080, y - 12),
                    (1550, y - 12),
                    GRID,
                    1,
                )

            self._put_text(
                canvas,
                label,
                (1080, y),
                scale=0.42,
                color=WHITE if key == "end_to_end_latency_ms" else MUTED,
                thickness=2 if key == "end_to_end_latency_ms" else 1,
            )
            self._put_text(
                canvas,
                value_text,
                (1415, y),
                scale=0.45,
                color=(
                    BLUE
                    if key == "end_to_end_latency_ms"
                    else AMBER
                    if value is not None
                    and key == "v2i_latency_ms"
                    and value > 50.0
                    else WHITE
                ),
                thickness=2 if key == "end_to_end_latency_ms" else 1,
            )
            y += 22

    def _draw_state(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
    ) -> None:
        x1, y1, x2, y2 = 1050, 568, 1580, 690
        self._panel(canvas, (x1, y1), (x2, y2))

        ego = snapshot.get("ego") or {}
        hazard = snapshot.get("hazard") or {}
        braking = snapshot.get("braking") or {}
        primary = hazard.get("primary") or {}

        speed = self._number(ego.get("speed_mps"), 0.0) or 0.0
        target = self._number(braking.get("target_speed_mps"))
        decel = self._number(braking.get("deceleration_mps2"), 0.0) or 0.0
        ttc = self._number(primary.get("ttc_s"))

        target_text = "N/A" if target is None else f"{target:.2f}"
        ttc_text = "N/A" if ttc is None else f"{ttc:.2f}s"
        brake_state = str(braking.get("state", "N/A"))

        self._put_text(
            canvas,
            "CONTROL / SYSTEM STATE",
            (1075, 600),
            scale=0.52,
            thickness=2,
        )

        state_color = (
            AMBER
            if bool(braking.get("active"))
            else GREEN
        )

        self._put_text(
            canvas,
            brake_state,
            (1080, 630),
            scale=0.48,
            color=state_color,
            thickness=2,
        )
        self._put_text(
            canvas,
            f"{speed:.2f} m/s  ->  {target_text} m/s",
            (1280, 630),
            scale=0.42,
            color=WHITE,
            thickness=1,
        )

        self._put_text(
            canvas,
            f"TTC {ttc_text}   |   decel {decel:.2f} m/s^2",
            (1080, 654),
            scale=0.38,
            color=MUTED,
        )

        observed = str(hazard.get("observed_level", "NORMAL"))
        control = str(hazard.get("control_level", "NORMAL"))

        self._put_text(
            canvas,
            f"Observed {observed}   |   Control {control}",
            (1080, 678),
            scale=0.38,
            color=self._hazard_color(control),
            thickness=1,
        )

    @staticmethod
    def _timeline_detail(event: Dict[str, Any]) -> str:
        event_type = str(event.get("event_type", ""))

        if event_type == "hazard_transition":
            old_level = (
                event.get("old_level")
                or event.get("previous_level")
                or "?"
            )
            new_level = event.get("new_level") or "?"
            return f"{old_level} -> {new_level}"

        if event_type == "stable_detection":
            source = event.get("source_id") or event.get("source")
            return f"source={source}"

        if event_type == "brake_trigger":
            distance = ProModeRenderer._number(
                event.get("distance_m")
            )
            ttc = ProModeRenderer._number(
                event.get("ttc_s")
            )
            distance_text = (
                "N/A" if distance is None else f"{distance:.2f}m"
            )
            ttc_text = (
                "N/A" if ttc is None else f"{ttc:.2f}s"
            )
            return f"distance={distance_text} TTC={ttc_text}"

        if event_type == "vehicle_stopped":
            clearance = ProModeRenderer._number(
                event.get("final_clearance_m")
            )
            return (
                "clearance=N/A"
                if clearance is None
                else f"clearance={clearance:.2f}m"
            )

        return ""

    def _draw_timeline(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
    ) -> None:
        x1, y1, x2, y2 = 20, 710, 1580, 882
        self._panel(canvas, (x1, y1), (x2, y2))

        self._put_text(
            canvas,
            "COOPERATIVE EVENT TIMELINE",
            (45, 742),
            scale=0.58,
            thickness=2,
        )

        events = list(snapshot.get("timeline") or [])
        latency = snapshot.get("latency_ms") or {}
        fused = snapshot.get("fusion") or {}
        braking = snapshot.get("braking") or {}

        has_rsu_detection = any(
            str(event.get("event_type")) == "stable_detection"
            and str(
                event.get("source_id")
                or event.get("source")
                or ""
            ).upper().startswith("RSU")
            for event in events
        )

        has_v2i = (
            has_rsu_detection
            and self._number(latency.get("v2i_latency_ms")) is not None
        )

        has_fusion = (
            has_rsu_detection
            and int(fused.get("object_count", 0) or 0) > 0
        )

        has_warning = any(
            str(event.get("event_type")) == "hazard_transition"
            and str(event.get("new_level", "")).upper()
            in {"WARNING", "CRITICAL"}
            for event in events
        )

        has_brake = any(
            str(event.get("event_type")) == "brake_trigger"
            for event in events
        ) or bool(braking.get("active"))

        has_safe_stop = any(
            str(event.get("event_type")) == "vehicle_stopped"
            for event in events
        ) or str(braking.get("state", "")).upper().startswith("STOPPED")

        steps = [
            ("RSU DETECT", has_rsu_detection, BLUE),
            ("V2I", has_v2i, BLUE),
            ("FUSION", has_fusion, BLUE),
            ("WARNING", has_warning, AMBER),
            ("BRAKE", has_brake, AMBER),
            ("SAFE STOP", has_safe_stop, GREEN),
        ]

        start_x = 115
        end_x = 1485
        step_y = 798
        gap = (end_x - start_x) / (len(steps) - 1)

        cv2.line(
            canvas,
            (start_x, step_y),
            (end_x, step_y),
            GRID,
            3,
            cv2.LINE_AA,
        )

        for index, (label, complete, color) in enumerate(steps):
            x = int(round(start_x + index * gap))

            if index > 0 and complete:
                previous_x = int(
                    round(start_x + (index - 1) * gap)
                )
                cv2.line(
                    canvas,
                    (previous_x, step_y),
                    (x, step_y),
                    color,
                    3,
                    cv2.LINE_AA,
                )

            node_color = color if complete else (85, 85, 85)

            cv2.circle(
                canvas,
                (x, step_y),
                11,
                node_color,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                (x, step_y),
                15,
                node_color,
                2,
                cv2.LINE_AA,
            )

            self._put_text(
                canvas,
                label,
                (x - 48, 775),
                scale=0.34,
                color=node_color,
                thickness=2 if complete else 1,
            )

        latest = events[-1] if events else None

        if latest is not None:
            event_type = str(latest.get("event_type", "event"))
            event_label = {
                "stable_detection": "RSU DETECTION",
                "hazard_transition": "HAZARD UPDATE",
                "brake_trigger": "BRAKE TRIGGER",
                "vehicle_stopped": "SAFE STOP",
            }.get(event_type, event_type.replace("_", " ").upper())

            detail = self._timeline_detail(latest)
            sim_time = self._number(latest.get("sim_time_s"))

            latest_text = f"Latest event  {event_label}"

            if detail:
                latest_text += f"   ·   {detail.replace('=', ' ')}"

            if sim_time is not None:
                latest_text += f"   ·   sim {sim_time:.2f} s"

            self._put_text(
                canvas,
                latest_text,
                (45, 856),
                scale=0.39,
                color=MUTED,
            )
        else:
            self._put_text(
                canvas,
                "Waiting for significant cooperative perception events",
                (45, 856),
                scale=0.39,
                color=MUTED,
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

        self._draw_header(canvas, snapshot, stale=stale)
        self._draw_fused_world(canvas, snapshot)
        self._draw_source_inspector(canvas, snapshot)
        self._draw_latency(canvas, snapshot)
        self._draw_state(canvas, snapshot)
        self._draw_timeline(canvas, snapshot)

        return canvas
