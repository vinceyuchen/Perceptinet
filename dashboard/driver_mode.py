from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .road_corridors import draw_road_corridors
from .object_icon_assets import draw_icon
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


# Driver-only dark glass, with semantic color reserved for critical status.
# OpenCV color tuples are BGR. Pro Mode palette and controls are unchanged.
WHITE = (246, 246, 246)
MUTED = (178, 178, 178)
BLUE = (209, 209, 209)
GREEN = (202, 202, 202)
CAUTION = (211, 211, 211)
WARNING = (242, 242, 242)
CRITICAL = (255, 255, 255)
PANEL = (34, 34, 34)
PANEL_LIGHT = (43, 43, 43)

# Calm -> caution -> danger. Stale/recorded readings stay neutral grey.
SAFE_GREEN = (135, 204, 121)
ALERT_YELLOW = (58, 203, 248)
DANGER_RED = (84, 83, 244)
SEVERE_RED = (64, 60, 255)
HAZARD_COLORS = {
    "NORMAL": SAFE_GREEN,
    "CAUTION": ALERT_YELLOW,
    "WARNING": DANGER_RED,
    "CRITICAL": SEVERE_RED,
}


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
        """Dark graphite translucent glass with monochrome highlights."""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        roi = canvas[y1:y2, x1:x2]
        blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=11)
        top_value, bottom_value = ((49, 29) if road else (55, 33))
        t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        shade = top_value * (1.0 - t) + bottom_value * t
        tinted = np.broadcast_to(shade, (h, w, 3))
        blended = np.clip(
            blur.astype(np.float32) * 0.26 + tinted * 0.74,
            0, 255,
        ).astype(np.uint8)
        mask = self._glass_mask(w, h)
        roi[mask > 0] = blended[mask > 0]
        inner = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
        edge = cv2.subtract(mask, inner)
        roi[edge > 0] = (80, 80, 80)
        cv2.line(canvas, (x1 + 30, y1 + 3), (x2 - 30, y1 + 3),
                 (111, 111, 111), 1, cv2.LINE_AA)

    def _make_base(self) -> np.ndarray:
        """Four dark frosted-glass cards on a graphite background."""
        yy, xx = np.mgrid[:720, :1280].astype(np.float32)
        upper_glow = np.exp(-((xx - 260) / 710) ** 2
                            - ((yy - 60) / 460) ** 2)
        lower_glow = np.exp(-((xx - 1090) / 680) ** 2
                            - ((yy - 750) / 530) ** 2)
        tone = np.clip(13.0 + 13.0 * upper_glow + 7.0 * lower_glow,
                       0, 255).astype(np.uint8)
        frame = np.repeat(tone[:, :, None], 3, axis=2)
        self._glass_card(frame, (20, 20, 824, 700), road=True)
        for bounds in ((844, 20, 1260, 239),
                       (844, 254, 1260, 476),
                       (844, 491, 1260, 700)):
            self._glass_card(frame, bounds)
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

    @staticmethod
    def _vehicle_state_color(action: str, *, stale: bool) -> Tuple[int, int, int]:
        """Use urgency, not class, to color the dominant vehicle state."""
        if stale:
            return MUTED
        action = str(action).upper()
        if action == "EMERGENCY BRAKING":
            return SEVERE_RED
        if action == "SLOWING":
            return ALERT_YELLOW
        return SAFE_GREEN  # DRIVING / STOPPED are non-escalating states.

    def _unified_status_color(self, snapshot: Dict[str, Any],
                              *, stale: bool) -> Tuple[int, int, int]:
        """Single authoritative accent for both Driver status cards.

        Hazard urgency takes priority over a vehicle's current motion:
        DRIVING does not mean SAFE when a CAUTION pedestrian is detected.
        Stale measurements cannot claim a current level of safety.
        """
        if stale:
            return MUTED
        level = str((snapshot.get("hazard") or {}).get(
            "control_level", "NORMAL"
        )).upper()
        braking = snapshot.get("braking") or {}
        action = self._vehicle_action(snapshot).upper()
        if level == "CRITICAL" or braking.get("emergency") or action == "EMERGENCY BRAKING":
            return SEVERE_RED
        if level == "WARNING":
            return DANGER_RED
        if level == "CAUTION" or action in {"SLOWING", "BRAKING"}:
            return ALERT_YELLOW
        if (snapshot.get("experiment") or {}).get("scenario") == "rain_v2i":
            return ALERT_YELLOW  # reduced visibility/communications != clear
        return SAFE_GREEN

    def _target_heading(self, canvas: np.ndarray, label: str,
                        *, accent: Tuple[int, int, int], stale: bool) -> None:
        """Center a large SVG icon + target name as one visual unit."""
        label = str(label).upper()
        font_scale, thickness = 1.24, 2
        icon_side, gap = 46, 13
        available = 355 - icon_side - gap
        (text_width, _), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
        )
        if text_width > available:
            font_scale *= available / max(text_width, 1)
            (text_width, _), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
            )
        left = int(1052 - (icon_side + gap + text_width) / 2)
        icon_center = (left + icon_side // 2, 112)
        if not draw_icon(
            canvas, icon_center, label, size=icon_side // 2,
            stale=stale, color=accent,
        ):
            # Unknown object class: do not substitute another class's icon.
            cv2.circle(canvas, icon_center, 17, accent, 2, cv2.LINE_AA)
            self._put_text(canvas, "?", (icon_center[0]-7, 120),
                           scale=0.64, color=accent, thickness=2)
        self._put_text(
            canvas, label, (left + icon_side + gap, 125),
            scale=font_scale, color=accent, thickness=thickness
        )

    def _centered_text(self, canvas: np.ndarray, text: str,
                       center_x: int, baseline_y: int,
                       *, max_width: int, scale: float,
                       color: Tuple[int, int, int],
                       thickness: int = 2) -> None:
        """Center a bold state label, scaling only if text is unusually long."""
        text = str(text)
        width, _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness
        )
        if width[0] > max_width:
            scale = max(0.47, scale * max_width / width[0])
            width, _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness
            )
        self._put_text(
            canvas, text, (int(center_x - width[0] / 2), baseline_y),
            scale=scale, color=color, thickness=thickness,
        )

    def _road_projection(self, lateral: float, forward: float) -> Tuple[int, int]:
        """Fixed v6.6 schematic projection. No dynamic or risk-based zoom.

        Show a wider side/forward context while giving nearby road users
        more screen space. The longitudinal compression is illustrative,
        not a metric-accurate distance ruler.
        """
        screen_x = 430.0 + lateral * 10.5
        if forward < 0.0:
            screen_y = 616.0 - forward * 9.0
        elif forward <= 50.0:
            screen_y = 616.0 - 16.0 * forward + 0.13 * forward * forward
        else:
            # Continue monotonically beyond the top of the visible window.
            screen_y = 141.0 - 3.0 * (forward - 50.0)
        return int(round(screen_x)), int(round(screen_y))

    def _draw_fallback_road(self, canvas: np.ndarray) -> None:
        """Minimal parallel street boundaries when no CARLA map is available."""
        # Extend the fallback corridor to the Road View edges too.
        # Keep all drawing clipped below the title and inside the card.
        left, top, right, bottom = 54, 142, 800, 682
        panel = canvas[top:bottom, left:right]
        for lateral in (-5.4, 5.4):
            start = self._road_projection(lateral, -12.0)
            end = self._road_projection(lateral, 62.0)
            cv2.line(
                panel,
                (start[0] - left, start[1] - top),
                (end[0] - left, end[1] - top),
                (177, 177, 177), 2, cv2.LINE_AA,
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

    @staticmethod
    def _ego_marker(canvas: np.ndarray, center: Tuple[int, int],
                    *, stale: bool) -> None:
        """Draw the uploaded EGO SVG emblem at the established map anchor."""
        draw_icon(canvas, center, "EGO", stale=stale)

    @staticmethod
    def _road_object(canvas: np.ndarray, center: Tuple[int, int],
                     klass: str, color: Tuple[int, int, int],
                     *, primary: bool, size: int,
                     stale: bool = False) -> None:
        """Render SVG-derived object artwork; preserve unknown fallback."""
        if not draw_icon(canvas, center, klass, size=size,
                         primary=primary, stale=stale):
            cv2.circle(canvas, center, size, color,
                       2 if primary else 1, cv2.LINE_AA)

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
            if not (-8 <= forward <= 52 and abs(lateral) <= 24):
                continue
            class_name = str(obj.get("class_name") or "OBJECT").upper()
            # YOLO's raw 'person' and legacy 'motorbike' names can appear
            # in snapshots. Normalize *display* labels without modifying
            # detection, fusion or risk-engine object records.
            class_name = {
                "PERSON": "PEDESTRIAN",
                "MOTORBIKE": "MOTORCYCLE",
            }.get(class_name, class_name)
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
                     else BLUE if klass == "PEDESTRIAN" else (187, 187, 187))
            self._road_object(canvas, (px, py), klass, color,
                              size=18 if is_primary else 13,
                              primary=is_primary, stale=stale)
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
        self._put_text(canvas, "ROAD MAP", (54, 73),
                       scale=0.87, color=WHITE, thickness=2)
        # Recorded layout previews must never claim they are live data.
        pill_label = (
            "DATA STALE" if stale
            else "PREVIEW" if snapshot.get("_preview_recorded")
            else "LIVE"
        )
        status_pill(canvas, pill_label,
                    (652, 76),
                    foreground=CAUTION if stale or snapshot.get("_preview_recorded") else GREEN,
                    fill=(48, 48, 48), border=(99, 99, 99),
                    scale=0.40, height=29)
        cx, ego_y = self._road_projection(0.0, 0.0)
        self._ego_marker(canvas, (cx, ego_y), stale=stale)

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
                self._road_object(canvas, (px, py), klass, color,
                                  size=19, primary=True, stale=stale)
                return

    def _draw_hazard(self, canvas: np.ndarray, snapshot: Dict[str, Any],
                     stale: bool) -> None:
        """Dominant icon + name (SAFE/UNCONFIRMED/class), distance, TTC."""
        level = str((snapshot.get("hazard") or {}).get(
            "control_level", "NORMAL"
        )).upper()
        primary = self._active_primary(snapshot)
        accent = self._unified_status_color(snapshot, stale=stale)

        if level == "NORMAL" and accent == SAFE_GREEN:
            label = "SAFE"
        elif primary is None:
            # No validated target while braking is active or risk is
            # elevated: avoid a misleading green-style SAFE claim.
            label = "UNCONFIRMED"
        else:
            label = str(primary.get("class_name") or "UNCONFIRMED").upper()
            label = {
                "PERSON": "PEDESTRIAN", "MOTORBIKE": "MOTORCYCLE"
            }.get(label, label)
        self._target_heading(canvas, label, accent=accent, stale=stale)

        # Distance and TTC stay legible and aligned for every heading.
        self._put_text(canvas, "DISTANCE", (875, 177),
                       scale=0.38, color=MUTED, thickness=1)
        self._put_text(canvas, "TTC", (1070, 177),
                       scale=0.38, color=MUTED, thickness=1)
        distance = self._finite(primary.get("distance_m")) if primary else None
        ttc = self._finite(primary.get("ttc_s")) if primary else None
        self._fit_text(
            canvas, "--" if distance is None else f"{distance:.1f} m",
            (875, 215), 164, 0.83,
            MUTED if stale else WHITE, 2
        )
        self._fit_text(
            canvas, "--" if ttc is None else f"{ttc:.1f} s",
            (1070, 215), 164, 0.83,
            MUTED if stale else accent, 2
        )

    def _draw_vehicle(self, canvas: np.ndarray, snapshot: Dict[str, Any],
                      stale: bool, presentation: Dict[str, Any]) -> None:
        """Large, centered action first. Speeds remain secondary evidence."""
        action = str(presentation["vehicle_action"]).upper()
        # Share precisely the same risk-colored accent as the target card.
        color = self._unified_status_color(snapshot, stale=stale)
        self._centered_text(
            canvas, action, 1052, 352, max_width=363,
            scale=1.22 if action != "EMERGENCY BRAKING" else 0.95,
            color=color, thickness=3
        )
        cv2.line(canvas, (875, 377), (1224, 377),
                 (84, 84, 84), 1, cv2.LINE_AA)
        self._put_text(canvas, "CURRENT SPEED", (875, 406),
                       scale=0.40, color=MUTED, thickness=1)
        self._put_text(canvas, "TARGET SPEED", (1063, 406),
                       scale=0.40, color=MUTED, thickness=1)
        current = self._finite((snapshot.get("ego") or {}).get("speed_mps"))
        target = self._finite((snapshot.get("braking") or {}).get("target_speed_mps"))
        current_txt = "--" if current is None else f"{current*3.6:.1f}"
        target_txt = "--" if target is None else f"{target*3.6:.1f}"
        for x, value, maxw in ((875, current_txt, 167),
                               (1063, target_txt, 165)):
            self._fit_text(canvas, f"{value} km/h", (x, 449), maxw,
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
                 (84, 84, 84), 1, cv2.LINE_AA)
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
