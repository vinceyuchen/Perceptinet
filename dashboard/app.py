from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .driver_mode import DriverModeRenderer
from .pro_mode import ProModeRenderer
from .ui_theme import draw_text, rounded_rect


BACKGROUND = (10, 10, 10)
NAV = (22, 22, 22)
NAV_BORDER = (55, 55, 55)
WHITE = (240, 240, 240)
MUTED = (155, 155, 155)
BLUE = (235, 170, 60)


class PerceptinetDashboardApp:
    """
    Unified Perceptinet dashboard shell.

    Driver Mode and Pro Mode consume the same live dashboard snapshot and can
    be switched instantly with on-screen buttons. The underlying CARLA /
    perception / fusion / risk / braking pipeline continues running unchanged.
    """

    DRIVER_MODE = "DRIVER"
    PRO_MODE = "PRO"

    def __init__(
        self,
        width: int = 1600,
        height: int = 900,
        *,
        default_mode: str = DRIVER_MODE,
    ):
        self.width = int(width)
        self.height = int(height)
        self.nav_height = 52

        self.driver_renderer = DriverModeRenderer(
            width=1280,
            height=720,
        )
        self.pro_renderer = ProModeRenderer(
            width=1600,
            height=900,
        )

        self.mode = (
            self.PRO_MODE
            if str(default_mode).upper().startswith("PRO")
            else self.DRIVER_MODE
        )

        self.driver_button = (
            self.width - 420,
            7,
            self.width - 225,
            45,
        )
        self.pro_button = (
            self.width - 215,
            7,
            self.width - 20,
            45,
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
    def _contains(
        rect: Tuple[int, int, int, int],
        x: int,
        y: int,
    ) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def set_mode(self, mode: str) -> str:
        mode_upper = str(mode).upper()

        if mode_upper.startswith("PRO"):
            self.mode = self.PRO_MODE
        else:
            self.mode = self.DRIVER_MODE

        return self.mode

    def toggle_mode(self) -> str:
        if self.mode == self.DRIVER_MODE:
            self.mode = self.PRO_MODE
        else:
            self.mode = self.DRIVER_MODE
        return self.mode

    def handle_click(
        self,
        x: int,
        y: int,
    ) -> Optional[str]:
        if self._contains(self.driver_button, x, y):
            return self.set_mode(self.DRIVER_MODE)

        if self._contains(self.pro_button, x, y):
            return self.set_mode(self.PRO_MODE)

        return None

    def _draw_button(
        self,
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        label: str,
        *,
        active: bool,
    ) -> None:
        x1, y1, x2, y2 = rect

        fill = BLUE if active else (38, 38, 38)
        text_color = (10, 10, 10) if active else WHITE

        rounded_rect(
            canvas,
            (x1, y1),
            (x2, y2),
            fill=fill,
            border=NAV_BORDER,
            border_thickness=1,
            radius=9,
        )

        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2,
        )

        tx = x1 + max(
            8,
            ((x2 - x1) - text_size[0]) // 2,
        )
        ty = y1 + (
            (y2 - y1 + text_size[1]) // 2
        )

        self._put_text(
            canvas,
            label,
            (tx, ty),
            scale=0.5,
            color=text_color,
            thickness=2,
        )

    @staticmethod
    def _fit_frame(
        frame: np.ndarray,
        target_width: int,
        target_height: int,
    ) -> np.ndarray:
        src_h, src_w = frame.shape[:2]

        scale = min(
            target_width / src_w,
            target_height / src_h,
        )

        out_w = max(1, int(src_w * scale))
        out_h = max(1, int(src_h * scale))

        interpolation = (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        )

        return cv2.resize(
            frame,
            (out_w, out_h),
            interpolation=interpolation,
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

        if self.mode == self.PRO_MODE:
            content = self.pro_renderer.render(
                snapshot,
                stale=stale,
            )
        else:
            content = self.driver_renderer.render(
                snapshot,
                stale=stale,
            )

        available_h = self.height - self.nav_height
        fitted = self._fit_frame(
            content,
            self.width,
            available_h,
        )

        fh, fw = fitted.shape[:2]

        x0 = (self.width - fw) // 2
        y0 = self.nav_height + (
            (available_h - fh) // 2
        )

        canvas[
            y0 : y0 + fh,
            x0 : x0 + fw,
        ] = fitted

        cv2.rectangle(
            canvas,
            (0, 0),
            (self.width, self.nav_height),
            NAV,
            -1,
        )
        cv2.line(
            canvas,
            (0, self.nav_height - 1),
            (self.width, self.nav_height - 1),
            NAV_BORDER,
            1,
        )

        self._put_text(
            canvas,
            "PERCEPTINET",
            (22, 33),
            scale=0.76,
            color=WHITE,
            thickness=2,
        )
        self._put_text(
            canvas,
            "COOPERATIVE PERCEPTION",
            (190, 32),
            scale=0.38,
            color=MUTED,
            thickness=1,
        )

        self._draw_button(
            canvas,
            self.driver_button,
            "DRIVER MODE",
            active=self.mode == self.DRIVER_MODE,
        )
        self._draw_button(
            canvas,
            self.pro_button,
            "PRO MODE",
            active=self.mode == self.PRO_MODE,
        )

        return canvas
