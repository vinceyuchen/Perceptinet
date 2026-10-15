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
        # Basemap-style accents are deliberately abstract, not map tiles.
        for x, y, w, h in ((69, 197, 160, 123),
                           (87, 374, 140, 173),
                           (642, 202, 134, 172),
                           (662, 433, 112, 152)):
            rounded_rect(frame, (x, y), (x+w, y+h),
                         fill=(33, 41, 40), radius=19)
        rounded_rect(frame, (254, 146), (605, 658),
                     fill=(49, 49, 51), radius=20)
        rounded_rect(frame, (267, 149), (592, 655),
                     fill=(45, 44, 46), radius=17)
        for x in (280, 579):
            cv2.line(frame, (x, 168), (x, 633),
                     (125, 115, 105), 2, cv2.LINE_AA)
        for y in range(205, 565, 66):
            cv2.line(frame, (430, y), (430, y+31),
                     (130, 115, 93), 3, cv2.LINE_AA)
        cv2.line(frame, (430, 565), (430, 600), BLUE, 3, cv2.LINE_AA)
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

    def _draw_road(self, canvas: np.ndarray, snapshot: Dict[str, Any],
                   stale: bool) -> None:
        self._put_text(canvas, "ROAD AHEAD", (54, 73),
                       scale=0.87, color=WHITE, thickness=2)
        self._put_text(canvas, "TOP-DOWN / EGO-RELATIVE SCHEMATIC",
                       (55, 102), scale=0.37, color=MUTED, thickness=1)
        status_pill(canvas, "DATA STALE" if stale else "LIVE",
                    (652, 76), foreground=CAUTION if stale else GREEN,
                    fill=(47, 45, 46), border=(85, 73, 67),
                    scale=0.40, height=29)
        cx, ego_y, span = 430, 591, 400
        for meters in (10, 20, 30):
            y = int(ego_y - meters/30.0*span)
            cv2.line(canvas, (283, y), (577, y),
                     (65, 64, 67), 1, cv2.LINE_AA)
            self._put_text(canvas, f"{meters} m", (171, y+6),
                           scale=0.40, color=MUTED, thickness=1)
        draw_navigation_marker(canvas, (cx, ego_y), size=24,
                               color=MUTED if stale else WHITE,
                               heading_deg=0.0, halo=True)
        self._put_text(canvas, "EGO", (412, 641),
                       scale=0.39, color=MUTED, thickness=1)

        hazard = snapshot.get("hazard") or {}
        primary = hazard.get("primary") or None
        if not primary:
            self._put_text(canvas, "LAST KNOWN VIEW" if stale else "ROAD CLEAR",
                           (634, 631), scale=0.46,
                           color=MUTED if stale else GREEN, thickness=1)
            return
        forward = self._finite(primary.get("longitudinal_m"))
        lateral = self._finite(primary.get("lateral_m"))
        if forward is None or lateral is None:
            self._put_text(canvas, "POSITION UNAVAILABLE", (553, 630),
                           scale=0.43, color=MUTED, thickness=1)
            return
        if not (0 <= forward <= 30 and -4 <= lateral <= 4):
            self._put_text(canvas, "TARGET OUTSIDE VIEW", (552, 630),
                           scale=0.43, color=MUTED, thickness=1)
            return
        px = int(cx+lateral/4.0*138)
        py = int(ego_y-forward/30.0*span)
        level = str(hazard.get("control_level", "NORMAL")).upper()
        color = MUTED if stale else self._hazard_color(level)
        klass = str(primary.get("class_name") or "OBJECT").upper()
        # Glow is local to the target to avoid frame-sized allocations.
        glow = canvas[max(0, py-37):py+38, max(0, px-37):px+38].copy()
        if glow.size:
            gx, gy = min(37, px), min(37, py)
            tinted = glow.copy()
            cv2.circle(tinted, (gx, gy), 31, color, -1, cv2.LINE_AA)
            cv2.addWeighted(tinted, 0.13, glow, 0.87, 0, dst=glow)
            canvas[max(0, py-37):py+38, max(0, px-37):px+38] = glow
        draw_object_marker(canvas, (px, py), klass,
                           color=color, size=20, primary=True)
        (tw, _), _ = cv2.getTextSize(
            klass, cv2.FONT_HERSHEY_DUPLEX, 0.52, 2)
        tx = px+37 if px+37+tw <= 794 else px-tw-37
        tx = max(56, tx)
        ty = max(192, min(606, py+5))
        self._fit_text(canvas, klass, (tx, ty), 178, 0.52, color, 2)
        sources = primary.get("sources") or []
        if sources:
            self._fit_text(canvas, str(sources[0]).upper(),
                           (tx, ty+22), 170, 0.38,
                           MUTED if stale else BLUE, 1)

    def _draw_hazard(self, canvas: np.ndarray, snapshot: Dict[str, Any],
                     stale: bool) -> None:
        hazard = snapshot.get("hazard") or {}
        primary = hazard.get("primary") or None
        level = str(hazard.get("control_level", "NORMAL")).upper()
        color = MUTED if stale else self._hazard_color(level)
        self._put_text(canvas, "HAZARD", (875, 58),
                       scale=0.52, color=MUTED, thickness=1)
        self._fit_text(canvas, "LAST KNOWN" if stale else level,
                       (1098, 58), 134, 0.43, color, 2)
        klass = (str(primary.get("class_name") or "OBJECT").upper()
                 if primary else "ROAD CLEAR")
        self._fit_text(canvas, klass, (875, 114), 347, 0.91,
                       color if primary else (MUTED if stale else GREEN), 2)
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
        self._fit_text(canvas, str(presentation["data_status"]),
                       (875, 647), 159, 0.51,
                       CAUTION if stale else GREEN, 2)
        health = str(presentation["display_health"])
        self._fit_text(canvas, health, (1062, 647), 166, 0.51,
                       CAUTION if stale else GREEN if health == "HEALTHY" else MUTED, 2)
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
