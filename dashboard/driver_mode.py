from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .road_corridors import draw_road_corridors
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
        self._base_canvas = self._make_base()

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


    @staticmethod
    def _glass_mask(w: int, h: int, radius: int = 26) -> np.ndarray:
        mask = np.zeros((h, w), np.uint8)
        r = min(radius, w // 2, h // 2)
        cv2.rectangle(mask, (r, 0), (w-r-1, h-1), 255, -1)
        cv2.rectangle(mask, (0, r), (w-1, h-r-1), 255, -1)
        for cx, cy in ((r, r), (w-r-1, r), (r, h-r-1), (w-r-1, h-r-1)):
            cv2.circle(mask, (cx, cy), r, 255, -1)
        return mask

    def _glass_card(self, canvas: np.ndarray, box: Tuple[int, int, int, int],
                    road: bool = False) -> None:
        """Cached translucent gradient over a blurred backdrop."""
        x1, y1, x2, y2 = box
        w, h = x2-x1, y2-y1
        roi = canvas[y1:y2, x1:x2]
        blur = cv2.GaussianBlur(roi, (0, 0), 11)
        top = np.array((48, 39, 33) if road else (62, 45, 34), np.float32)
        bot = np.array((30, 27, 26) if road else (43, 34, 29), np.float32)
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        tint = top[None, None, :]*(1-t)+bot[None, None, :]*t
        fill = np.clip(blur.astype(np.float32)*0.31+tint*0.69, 0, 255).astype(np.uint8)
        mask = self._glass_mask(w, h)
        roi[mask > 0] = fill[mask > 0]
        inner = cv2.erode(mask, np.ones((3, 3), np.uint8))
        rim = cv2.subtract(mask, inner)
        roi[rim > 0] = (83, 73, 65)
        cv2.line(canvas, (x1+30, y1+3), (x2-30, y1+3),
                 (101, 89, 75), 1, cv2.LINE_AA)

    def _make_base(self) -> np.ndarray:
        """Four frosted-glass regions rendered once per instance."""
        yy, xx = np.mgrid[:720, :1280].astype(np.float32)
        glow = np.exp(-((xx-210)/650)**2 - ((yy-75)/500)**2)
        glow2 = np.exp(-((xx-1120)/600)**2 - ((yy-800)/500)**2)
        frame = np.empty((720, 1280, 3), np.uint8)
        frame[:, :, 0] = np.clip(18+16*glow+13*glow2, 0, 255)
        frame[:, :, 1] = np.clip(16+11*glow+6*glow2, 0, 255)
        frame[:, :, 2] = np.clip(15+6*glow+5*glow2, 0, 255)
        self._glass_card(frame, (20, 20, 824, 700), road=True)
        for bounds in ((844, 20, 1260, 239),
                       (844, 254, 1260, 476),
                       (844, 491, 1260, 700)):
            self._glass_card(frame, bounds)
        # Roads are rendered per snapshot from CARLA waypoints in
        # _draw_road; no fixed geometry is baked into the glass background.

        return frame

    @staticmethod
    def _finite(value: Any) -> Optional[float]:
        number = DriverModeRenderer._number(value)
        return number if number is not None and math.isfinite(number) else None

    def _fit_text(self, canvas: np.ndarray, text: str, xy: Tuple[int, int],
                  max_width: int, scale: float = 0.65,
                  color: Tuple[int, int, int] = WHITE,
                  thickness: int = 2) -> None:
        text = str(text)
        (tw, _), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
        if tw > max_width:
            scale = max(0.30, scale*max_width/tw)
        self._put_text(canvas, text, xy, scale=scale,
                       color=color, thickness=thickness)

    def _road_projection(self, lateral: float, forward: float) -> Tuple[int, int]:
        """Ego-relative [lateral, forward] metres -> Road View pixel."""
        return (int(round(430 + lateral * 10.0)),
                int(round(590 - forward * 10.2)))

    def _draw_fallback_road(self, canvas: np.ndarray) -> None:
        """Minimal parallel street boundaries when no CARLA map is available."""
        for x in (385, 475):
            cv2.line(
                canvas, (x, 180), (x, 644),
                (149, 156, 164), 2, cv2.LINE_AA,
            )

    def _draw_context_roads(
        self, canvas: np.ndarray, road_context: Any
    ) -> bool:
        """Only street boundary pairs; never an outlined union of lanes."""
        if not isinstance(road_context, dict):
            return False
        if (road_context.get("coordinate_frame")
                != "ego_relative_lateral_forward_m"):
            return False
        return draw_road_corridors(
            canvas,
            road_context.get("segments"),
            self._road_projection,
        )

    def _draw_detected_objects(
        self, canvas: np.ndarray, snapshot: Dict[str, Any],
        *, stale: bool, active_primary: Optional[Dict[str, Any]]
    ) -> int:
        """Show roadside/tracked road users, excluding ego-overlap detections.

        Use display coordinates only for presentation. Do not alter
        fused tracks, hazard assessment, or CARLA actors.
        """
        ego = snapshot.get("ego") or {}
        position = ego.get("world_position") or {}
        ex, ey = self._finite(position.get("x")), self._finite(position.get("y"))
        yaw = self._finite(ego.get("yaw_deg"))
        if ex is None or ey is None or yaw is None:
            return 0
        radians = math.radians(yaw)
        co, si = math.cos(radians), math.sin(radians)
        primary_id = active_primary.get("actor_id") if active_primary else None
        objects = (snapshot.get("fusion") or {}).get("objects") or []
        candidates = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            loc = obj.get("display_world_position") or obj.get("world_position") or {}
            if not isinstance(loc, dict):
                continue
            wx, wy = self._finite(loc.get("x")), self._finite(loc.get("y"))
            if wx is None or wy is None:
                continue
            dx, dy = wx-ex, wy-ey
            if math.hypot(dx, dy) < 1.2:
                # An apparent car at the exact ego pose may be self-detection.
                # Exclude it from the *map drawing only*, not from raw data.
                continue
            forward, lateral = dx*co + dy*si, -dx*si + dy*co
            if not (-6 <= forward <= 40 and abs(lateral) <= 18):
                continue
            class_name = str(obj.get("class_name") or "OBJECT").upper()
            if class_name not in {"PEDESTRIAN", "CAR", "TRUCK", "BUS",
                                  "MOTORCYCLE", "BICYCLE"}:
                continue
            is_primary = bool(active_primary and obj.get("actor_id") == primary_id)
            candidates.append((is_primary, forward, lateral, class_name, obj))
        candidates.sort(key=lambda item: (item[0], -item[1]))
        count = 0
        for is_primary, forward, lateral, klass, obj in candidates[:10]:
            px, py = self._road_projection(lateral, forward)
            if not (70 <= px <= 782 and 172 <= py <= 659):
                continue
            level = str((snapshot.get("hazard") or {}).get(
                "control_level", "NORMAL")).upper()
            color = (MUTED if stale else
                     self._hazard_color(level) if is_primary
                     else BLUE if klass == "PEDESTRIAN" else (147, 153, 164))
            draw_object_marker(canvas, (px, py), klass,
                               color=color, size=18 if is_primary else 13,
                               primary=is_primary)
            (tw, _), _ = cv2.getTextSize(
                klass, cv2.FONT_HERSHEY_DUPLEX, 0.43, 1)
            tx = px+23 if px+23+tw < 778 else px-tw-25
            ty = min(635, max(192, py-7))
            self._fit_text(canvas, klass, (tx, ty), 145,
                           0.43, color, 1)
            if is_primary:
                sources = obj.get("display_sources") or obj.get("sources") or []
                if sources:
                    self._fit_text(canvas, str(sources[0]).upper(),
                                   (tx, ty+18), 145, 0.34,
                                   MUTED if stale else BLUE, 1)
            count += 1
        return count

    def _active_primary(
        self, snapshot: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Ignore a NORMAL-only or ego-overlap primary for Driver warnings.

        Control state remains authoritative. This only decides whether an
        individual object can be presented as its current risk target.
        """
        hazard = snapshot.get("hazard") or {}
        if str(hazard.get("control_level", "NORMAL")).upper() not in {
                "CAUTION", "WARNING", "CRITICAL"}:
            return None
        primary = hazard.get("primary")
        if not isinstance(primary, dict):
            return None
        if str(primary.get("hazard_level", "NORMAL")).upper() not in {
                "CAUTION", "WARNING", "CRITICAL"}:
            return None
        forward = self._finite(primary.get("longitudinal_m"))
        if forward is None or forward <= 0.5:
            return None
        return primary

    def _draw_road(self, canvas: np.ndarray, snapshot: Dict[str, Any],
                   stale: bool) -> None:
        road_context = snapshot.get("road_context")
        map_available = self._draw_context_roads(canvas, road_context)
        if not map_available:
            self._draw_fallback_road(canvas)
        self._put_text(canvas, "ROAD AHEAD", (54, 73),
                       scale=0.87, color=WHITE, thickness=2)
        self._put_text(
            canvas,
            "CARLA ROAD CENTERLINES / SCHEMATIC" if map_available
            else "EGO-RELATIVE ROAD SCHEMATIC",
            (55, 102), scale=0.37, color=MUTED, thickness=1,
        )
        # Recorded layout previews must never claim they are live data.
        pill_label = (
            "DATA STALE" if stale
            else "PREVIEW" if snapshot.get("_preview_recorded")
            else "LIVE"
        )
        status_pill(canvas, pill_label,
                    (652, 76), foreground=CAUTION if stale or snapshot.get("_preview_recorded") else GREEN,
                    fill=(47, 45, 46), border=(85, 73, 67),
                    scale=0.40, height=29)
        cx, ego_y = self._road_projection(0.0, 0.0)
        draw_navigation_marker(canvas, (cx, ego_y), size=24,
                               color=MUTED if stale else WHITE,
                               heading_deg=0.0, halo=True)
        self._put_text(canvas, "EGO", (413, ego_y+45),
                       scale=0.39, color=MUTED, thickness=1)

        hazard = snapshot.get("hazard") or {}
        level = str(hazard.get("control_level", "NORMAL")).upper()
        active_primary = self._active_primary(snapshot)
        drawn_count = self._draw_detected_objects(
            canvas, snapshot, stale=stale, active_primary=active_primary
        )
        if drawn_count:
            return

        # Older snapshots may omit fused objects. Draw only a genuine
        # non-NORMAL primary risk using its existing ego-relative coordinates.
        if active_primary:
            forward = self._finite(active_primary.get("longitudinal_m"))
            lateral = self._finite(active_primary.get("lateral_m"))
            if (forward is not None and lateral is not None
                    and -6 <= forward <= 40 and abs(lateral) <= 18):
                px, py = self._road_projection(lateral, forward)
                color = MUTED if stale else self._hazard_color(level)
                klass = str(active_primary.get("class_name") or "OBJECT").upper()
                draw_object_marker(canvas, (px, py), klass,
                                   color=color, size=19, primary=True)
                self._fit_text(canvas, klass, (px+28, py-7), 155,
                               0.45, color, 1)
                return
        self._put_text(canvas, "NO RECENT TRACKS" if stale else "NO TRACKED OBJECTS",
                       (583, 651), scale=0.36, color=MUTED, thickness=1)

    def _draw_hazard(self, canvas: np.ndarray, snapshot: Dict[str, Any],
                     stale: bool) -> None:
        hazard = snapshot.get("hazard") or {}
        level = str(hazard.get("control_level", "NORMAL")).upper()
        # Avoid showing the first NORMAL risk as an active danger:
        # the data layer may pick an ego-overlap car in a NORMAL-only run.
        primary = self._active_primary(snapshot)
        color = MUTED if stale else self._hazard_color(level)
        self._put_text(canvas, "HAZARD", (875, 58),
                       scale=0.52, color=MUTED, thickness=1)
        self._fit_text(canvas, "LAST KNOWN" if stale else level,
                       (1098, 58), 134, 0.43, color, 2)
        klass = (str(primary.get("class_name") or "OBJECT").upper()
                 if primary else ("TARGET UNCONFIRMED" if level != "NORMAL"
                                  else "NO ACTIVE ALERT"))
        self._fit_text(canvas, klass, (875, 114), 347, 0.91,
                       color if primary or level != "NORMAL"
                       else (MUTED if stale else GREEN), 2)
        self._put_text(canvas, "DISTANCE", (875, 157),
                       scale=0.42, color=MUTED, thickness=1)
        self._put_text(canvas, "TIME TO COLLISION", (1054, 157),
                       scale=0.38, color=MUTED, thickness=1)
        distance = self._finite(primary.get("distance_m")) if primary else None
        ttc = self._finite(primary.get("ttc_s")) if primary else None
        self._put_text(canvas, "--" if distance is None else f"{distance:.1f} m",
                       (875, 209), scale=0.83,
                       color=MUTED if stale else WHITE, thickness=2)
        self._put_text(canvas, "--" if ttc is None else f"{ttc:.1f} s",
                       (1054, 209), scale=0.83, color=color, thickness=2)

    def _draw_vehicle(self, canvas: np.ndarray, snapshot: Dict[str, Any],
                      stale: bool, presentation: Dict[str, Any]) -> None:
        self._put_text(canvas, "VEHICLE RESPONSE", (875, 290),
                       scale=0.52, color=MUTED, thickness=1)
        action = str(presentation["vehicle_action"])
        color = (MUTED if stale else GREEN
                 if action in ("DRIVING", "STOPPED")
                 else self._hazard_color(str(presentation["hazard_level"])))
        self._fit_text(canvas, action, (875, 351), 350, 0.87, color, 2)
        cv2.line(canvas, (875, 373), (1224, 373),
                 (67, 64, 68), 1, cv2.LINE_AA)
        self._put_text(canvas, "CURRENT SPEED", (875, 403),
                       scale=0.40, color=MUTED, thickness=1)
        self._put_text(canvas, "TARGET SPEED", (1063, 403),
                       scale=0.40, color=MUTED, thickness=1)
        current = self._finite((snapshot.get("ego") or {}).get("speed_mps"))
        target = self._finite((snapshot.get("braking") or {}).get("target_speed_mps"))
        current_txt = "--" if current is None else f"{current*3.6:.1f}"
        target_txt = "--" if target is None else f"{target*3.6:.1f}"
        for x, value, maxw in ((875, current_txt, 167),
                               (1063, target_txt, 165)):
            self._fit_text(canvas, f"{value} km/h", (x, 446), maxw,
                           0.72, MUTED if stale else WHITE, 2)

    def _draw_system(self, canvas: np.ndarray, snapshot: Dict[str, Any],
                     stale: bool, presentation: Dict[str, Any]) -> None:
        self._put_text(canvas, "COOPERATIVE SYSTEM", (875, 529),
                       scale=0.53, color=MUTED, thickness=1)
        cooperative = bool(presentation["cooperative_active"]) and not stale
        perception = str(presentation["perception_text"])
        status_color = CAUTION if stale else BLUE if cooperative else WHITE
        self._fit_text(canvas, perception, (875, 574), 345,
                       0.65, status_color, 2)
        cv2.line(canvas, (875, 590), (1224, 590),
                 (67, 64, 68), 1, cv2.LINE_AA)
        self._put_text(canvas, "DATA", (875, 616),
                       scale=0.36, color=MUTED, thickness=1)
        self._put_text(canvas, "SYSTEM", (1062, 616),
                       scale=0.36, color=MUTED, thickness=1)
        preview = bool(snapshot.get("_preview_recorded"))
        self._fit_text(canvas, str(presentation["data_status"]),
                       (875, 647), 159, 0.51,
                       CAUTION if stale or preview else GREEN, 2)
        health = str(presentation["display_health"])
        self._fit_text(canvas, health, (1062, 647), 166, 0.51,
                       CAUTION if stale else MUTED if preview
                       else GREEN if health == "HEALTHY" else MUTED, 2)
        sources = presentation["sources"] or []
        text_sources = " + ".join(str(s).upper() for s in sources)
        if not text_sources:
            text_sources = "NO LIVE SOURCES" if stale else "ONBOARD ONLY"
        self._fit_text(canvas, f"SOURCE  {text_sources}",
                       (875, 679), 345, 0.35, MUTED, 1)

    def render(self, snapshot: Dict[str, Any],
               *, stale: bool = False) -> np.ndarray:
        """Render four connected cards from the unchanged snapshot schema."""
        presentation = derive_presentation_state(snapshot, stale=stale)
        if snapshot.get("_preview_recorded") and not stale:
            # Normal-color layout check of historical data, not a live feed.
            presentation = {
                **presentation,
                "data_status": "PREVIEW",
                "display_health": "RECORDED",
                "perception_text": "RECORDED SOURCES",
            }
        canvas = self._base_canvas.copy()
        self._draw_road(canvas, snapshot, stale)
        self._draw_hazard(canvas, snapshot, stale)
        self._draw_vehicle(canvas, snapshot, stale, presentation)
        self._draw_system(canvas, snapshot, stale, presentation)
        if self.width != 1280 or self.height != 720:
            canvas = cv2.resize(
                canvas, (self.width, self.height),
                interpolation=cv2.INTER_AREA)
        return canvas
