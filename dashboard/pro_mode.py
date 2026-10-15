from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


WHITE = (240, 240, 240)
MUTED = (155, 155, 155)
GRID = (65, 65, 65)
PANEL = (24, 24, 24)
PANEL_LIGHT = (34, 34, 34)
BACKGROUND = (10, 10, 10)
GREEN = (90, 210, 120)
BLUE = (235, 170, 60)
AMBER = (0, 170, 255)
RED = (50, 50, 235)

HAZARD_COLORS = {
    "NORMAL": WHITE,
    "CAUTION": AMBER,
    "WARNING": (0, 140, 255),
    "CRITICAL": RED,
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
        return [str(x) for x in (obj.get("sources") or [])]

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
            color=BLUE,
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

        health = str(
            (snapshot.get("system_health") or {}).get(
                "status",
                "UNKNOWN",
            )
        )

        self._put_text(
            canvas,
            f"SYSTEM {health}",
            (610, 34),
            scale=0.52,
            color=GREEN if health == "HEALTHY" else AMBER,
            thickness=2,
        )

        live_text = "LIVE" if not stale else "DATA STALE"
        self._put_text(
            canvas,
            live_text,
            (1120, 34),
            scale=0.58,
            color=GREEN if not stale else AMBER,
            thickness=2,
        )

        cv2.rectangle(
            canvas,
            (1325, 14),
            (1574, 64),
            color,
            -1,
        )
        self._put_text(
            canvas,
            level,
            (1350, 48),
            scale=0.72,
            color=(10, 10, 10),
            thickness=2,
        )

    def _relative_xy(
        self,
        snapshot: Dict[str, Any],
        obj: Dict[str, Any],
    ) -> Optional[Tuple[float, float]]:
        ego = snapshot.get("ego") or {}
        ego_pos = ego.get("world_position") or {}
        obj_pos = obj.get("world_position") or {}

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

        if not candidates:
            # Backward-compatible fallback for the current Metrics v1
            # summary, which separately exposes the earliest RSU.
            source = detection.get("earliest_rsu_source")
            return None if source is None else str(source)

        return min(candidates, key=lambda item: item[0])[1]

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

        # Ego.
        cv2.rectangle(
            canvas,
            (center_x - 24, plot_bottom - 22),
            (center_x + 24, plot_bottom + 3),
            WHITE,
            2,
        )
        self._put_text(
            canvas,
            "EGO",
            (center_x - 17, plot_bottom - 6),
            scale=0.38,
            color=WHITE,
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
            radius = 13 if is_primary else 9

            cv2.circle(
                canvas,
                (px, py),
                radius,
                color,
                2,
            )
            cv2.circle(
                canvas,
                (px, py),
                3,
                color,
                -1,
            )

            class_name = str(
                obj.get("class_name") or "object"
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

            prefix = ">" if selected else " "
            text = f"{prefix} {source:<10} objects={count}"

            self._put_text(
                canvas,
                text,
                (1080, y),
                scale=0.48,
                color=BLUE if selected else WHITE,
                thickness=2 if selected else 1,
            )

            badges: List[str] = []

            if source == first_source:
                badges.append("FIRST DETECTION")

            if contributor:
                badges.append("CONTRIBUTOR")

            if badges:
                self._put_text(
                    canvas,
                    " | ".join(badges),
                    (1325, y),
                    scale=0.30,
                    color=AMBER if source == first_source else BLUE,
                    thickness=2,
                )

            y += 30

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
            value_text = "N/A" if value is None else f"{value:7.2f} ms"

            self._put_text(
                canvas,
                label,
                (1080, y),
                scale=0.41,
                color=MUTED,
            )
            self._put_text(
                canvas,
                value_text,
                (1390, y),
                scale=0.43,
                color=(
                    AMBER
                    if value is not None
                    and key == "v2i_latency_ms"
                    and value > 50.0
                    else WHITE
                ),
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

        target_text = (
            "N/A"
            if target is None
            else f"{target:.2f}"
        )
        ttc_text = (
            "N/A"
            if ttc is None
            else f"{ttc:.2f}s"
        )

        self._put_text(
            canvas,
            "CONTROL / SYSTEM STATE",
            (1075, 600),
            scale=0.52,
            thickness=2,
        )

        self._put_text(
            canvas,
            (
                f"Speed {speed:.2f} m/s | "
                f"target {target_text} m/s | TTC {ttc_text}"
            ),
            (1080, 628),
            scale=0.38,
        )
        self._put_text(
            canvas,
            (
                f"Brake {braking.get('state', 'N/A')} | "
                f"active={bool(braking.get('active'))} | "
                f"decel {decel:.2f} m/s^2"
            ),
            (1080, 652),
            scale=0.36,
        )
        self._put_text(
            canvas,
            (
                "Hazard observed "
                f"{hazard.get('observed_level', 'NORMAL')} | "
                "control "
                f"{hazard.get('control_level', 'NORMAL')}"
            ),
            (1080, 676),
            scale=0.38,
            color=MUTED,
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
            "EVENT TIMELINE",
            (45, 742),
            scale=0.58,
            thickness=2,
        )
        self._put_text(
            canvas,
            "event",
            (45, 770),
            scale=0.38,
            color=MUTED,
        )
        self._put_text(
            canvas,
            "frame",
            (360, 770),
            scale=0.38,
            color=MUTED,
        )
        self._put_text(
            canvas,
            "sim time",
            (500, 770),
            scale=0.38,
            color=MUTED,
        )
        self._put_text(
            canvas,
            "detail",
            (690, 770),
            scale=0.38,
            color=MUTED,
        )

        events = list(snapshot.get("timeline") or [])[-5:]
        y = 798

        if not events:
            self._put_text(
                canvas,
                "No significant events yet",
                (45, 820),
                scale=0.48,
                color=MUTED,
            )
            return

        for event in events:
            event_type = str(
                event.get("event_type", "event")
            )
            frame_id = event.get("frame_id", "")
            sim_time = self._number(event.get("sim_time_s"))
            detail = self._timeline_detail(event)

            event_color = (
                GREEN
                if event_type == "vehicle_stopped"
                else AMBER
                if event_type in {
                    "hazard_transition",
                    "brake_trigger",
                }
                else BLUE
            )

            self._put_text(
                canvas,
                event_type,
                (45, y),
                scale=0.40,
                color=event_color,
            )
            self._put_text(
                canvas,
                str(frame_id),
                (360, y),
                scale=0.40,
            )
            self._put_text(
                canvas,
                (
                    "N/A"
                    if sim_time is None
                    else f"{sim_time:.2f}s"
                ),
                (500, y),
                scale=0.40,
            )
            self._put_text(
                canvas,
                detail,
                (690, y),
                scale=0.40,
                color=MUTED,
            )
            y += 22

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
