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
            "PERCEPTINET",
            (24, 35),
            scale=0.82,
            thickness=2,
        )
        self._put_text(
            canvas,
            "PRO MODE",
            (24, 63),
            scale=0.5,
            color=BLUE,
            thickness=2,
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

    def _draw_fused_world(
        self,
        canvas: np.ndarray,
        snapshot: Dict[str, Any],
    ) -> None:
        x1, y1, x2, y2 = 20, 102, 1030, 690
        self._panel(canvas, (x1, y1), (x2, y2))

        fused = snapshot.get("fusion") or {}
        objects = fused.get("objects", []) or []

        self._put_text(
            canvas,
            "FUSED WORLD MODEL",
            (45, 137),
            scale=0.62,
            thickness=2,
        )
        self._put_text(
            canvas,
            f"{len(objects)} objects | source filter: {self.selected_source}",
            (45, 165),
            scale=0.45,
            color=MUTED,
        )

        plot_left, plot_top = 60, 195
        plot_right, plot_bottom = 995, 655
        center_x = (plot_left + plot_right) // 2

        # Engineering grid: +/- 10 m lateral, 0..40 m longitudinal.
        for lateral in range(-10, 11, 5):
            px = int(
                center_x
                + (lateral / 10.0)
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
                f"{lateral:+d}m",
                (px - 18, plot_bottom + 20),
                scale=0.34,
                color=MUTED,
            )

        for longitudinal in range(0, 41, 10):
            py = int(
                plot_bottom
                - (longitudinal / 40.0)
                * (plot_bottom - plot_top)
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
                f"{longitudinal}m",
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

        primary = (
            (snapshot.get("hazard") or {}).get("primary")
        )
        primary_actor = (
            primary.get("actor_id")
            if primary
            else None
        )

        plotted = 0

        for obj in objects:
            if not self._object_visible_for_source(obj):
                continue

            rel = self._relative_xy(snapshot, obj)
            if rel is None:
                continue

            longitudinal, lateral = rel

            if (
                longitudinal < 0.0
                or longitudinal > 40.0
                or abs(lateral) > 10.0
            ):
                continue

            px = int(
                center_x
                + (lateral / 10.0)
                * ((plot_right - plot_left) / 2.0)
            )
            py = int(
                plot_bottom
                - (longitudinal / 40.0)
                * (plot_bottom - plot_top)
            )

            is_primary = (
                primary_actor is not None
                and obj.get("actor_id") == primary_actor
            )

            color = RED if is_primary else BLUE
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

            self._put_text(
                canvas,
                label,
                (px + 15, py - 2),
                scale=0.38,
                color=color,
            )

            sources = ",".join(
                self._object_sources(obj)
            )
            self._put_text(
                canvas,
                sources,
                (px + 15, py + 15),
                scale=0.30,
                color=MUTED,
            )

            plotted += 1

        if plotted == 0:
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

            if contributor:
                self._put_text(
                    canvas,
                    "ACTIVE CONTRIBUTOR",
                    (1360, y),
                    scale=0.34,
                    color=AMBER,
                    thickness=2,
                )

            y += 30

        self._put_text(
            canvas,
            "TAB: cycle source",
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

        speed = self._number(ego.get("speed_mps"), 0.0) or 0.0
        decel = self._number(braking.get("deceleration_mps2"), 0.0) or 0.0

        self._put_text(
            canvas,
            "CONTROL / SYSTEM STATE",
            (1075, 600),
            scale=0.52,
            thickness=2,
        )

        self._put_text(
            canvas,
            f"Speed {speed:.2f} m/s",
            (1080, 630),
            scale=0.42,
        )
        self._put_text(
            canvas,
            f"Brake {braking.get('state', 'N/A')} | decel {decel:.2f} m/s^2",
            (1080, 653),
            scale=0.40,
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
            scale=0.39,
            color=MUTED,
        )

    @staticmethod
    def _timeline_detail(event: Dict[str, Any]) -> str:
        event_type = str(event.get("event_type", ""))

        if event_type == "hazard_transition":
            return (
                f"{event.get('previous_level', '?')}"
                f" -> {event.get('new_level', '?')}"
            )

        if event_type == "stable_detection":
            source = event.get("source_id") or event.get("source")
            return f"source={source}"

        if event_type == "brake_trigger":
            distance = event.get("distance_m")
            ttc = event.get("ttc_s")
            return f"distance={distance}m TTC={ttc}s"

        if event_type == "vehicle_stopped":
            clearance = event.get("final_clearance_m")
            return f"clearance={clearance}m"

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
                RED
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
