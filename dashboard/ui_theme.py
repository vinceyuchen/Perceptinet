from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


Color = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Perceptinet "Graphite" palette
#
# A restrained automotive dark theme inspired by premium EV interfaces:
# neutral graphite surfaces carry the hierarchy, while colour is reserved
# for meaning (cooperative perception, health, caution and critical hazards).
# OpenCV uses BGR tuples.
# ---------------------------------------------------------------------------

BACKGROUND = (15, 15, 16)
NAV = (20, 20, 22)

SURFACE_1 = (24, 24, 26)
SURFACE_2 = (30, 30, 33)
SURFACE_3 = (38, 38, 42)

BORDER = (54, 54, 59)
BORDER_SOFT = (43, 43, 47)
GRID = (57, 57, 62)

TEXT_PRIMARY = (241, 241, 243)
TEXT_SECONDARY = (177, 177, 183)
TEXT_TERTIARY = (112, 112, 120)

# Desaturated steel-blue. Reserved for cooperative perception / selected data.
COOPERATIVE = (166, 145, 112)

# Muted semantic colours. Saturated colour is deliberately avoided.
HEALTHY = (132, 166, 126)
CAUTION = (82, 166, 210)
WARNING = (82, 132, 218)
CRITICAL = (92, 92, 216)

# Neutral selected control, Tesla-like rather than bright-blue.
SELECTED = (218, 218, 222)
SELECTED_TEXT = (24, 24, 26)


def draw_text(
    canvas: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    *,
    scale: float = 0.55,
    color: Color = (240, 240, 240),
    thickness: int = 1,
    font_face: int = cv2.FONT_HERSHEY_DUPLEX,
    supersample: int = 3,
) -> None:
    """
    Draw sharper OpenCV text by rasterising the glyph mask at higher
    resolution and downsampling it before alpha-compositing onto the canvas.

    This avoids the soft/blurry look produced by drawing small Hershey glyphs
    directly on a dashboard frame, while keeping the project dependency-free.
    """
    value = str(text)

    if not value:
        return

    ss = max(2, int(supersample))
    high_scale = float(scale) * ss
    high_thickness = max(1, int(round(thickness * ss)))

    (text_w, text_h), baseline = cv2.getTextSize(
        value,
        font_face,
        high_scale,
        high_thickness,
    )

    pad = 4 * ss

    mask = np.zeros(
        (
            text_h + baseline + pad * 2,
            text_w + pad * 2,
        ),
        dtype=np.uint8,
    )

    cv2.putText(
        mask,
        value,
        (pad, pad + text_h),
        font_face,
        high_scale,
        255,
        high_thickness,
        cv2.LINE_AA,
    )

    final_w = max(1, int(round(mask.shape[1] / ss)))
    final_h = max(1, int(round(mask.shape[0] / ss)))

    mask = cv2.resize(
        mask,
        (final_w, final_h),
        interpolation=cv2.INTER_AREA,
    )

    x, baseline_y = int(xy[0]), int(xy[1])

    baseline_offset = int(round((pad + text_h) / ss))
    left = int(round(x - pad / ss))
    top = baseline_y - baseline_offset

    canvas_h, canvas_w = canvas.shape[:2]

    x1 = max(0, left)
    y1 = max(0, top)
    x2 = min(canvas_w, left + final_w)
    y2 = min(canvas_h, top + final_h)

    if x1 >= x2 or y1 >= y2:
        return

    mx1 = x1 - left
    my1 = y1 - top
    mx2 = mx1 + (x2 - x1)
    my2 = my1 + (y2 - y1)

    alpha = (
        mask[my1:my2, mx1:mx2]
        .astype(np.float32)
        / 255.0
    )[..., None]

    region = canvas[y1:y2, x1:x2].astype(np.float32)
    foreground = np.array(color, dtype=np.float32).reshape(1, 1, 3)

    blended = (
        region * (1.0 - alpha)
        + foreground * alpha
    )

    canvas[y1:y2, x1:x2] = np.clip(
        blended,
        0,
        255,
    ).astype(np.uint8)


def _filled_rounded_rect(
    canvas: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Color,
    radius: int,
) -> None:
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)

    if x2 <= x1 or y2 <= y1:
        return

    r = max(
        0,
        min(
            int(radius),
            (x2 - x1) // 2,
            (y2 - y1) // 2,
        ),
    )

    if r <= 0:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        return

    cv2.rectangle(
        canvas,
        (x1 + r, y1),
        (x2 - r, y2),
        color,
        -1,
    )
    cv2.rectangle(
        canvas,
        (x1, y1 + r),
        (x2, y2 - r),
        color,
        -1,
    )

    for cx, cy in (
        (x1 + r, y1 + r),
        (x2 - r, y1 + r),
        (x1 + r, y2 - r),
        (x2 - r, y2 - r),
    ):
        cv2.circle(
            canvas,
            (cx, cy),
            r,
            color,
            -1,
            cv2.LINE_AA,
        )


def rounded_rect(
    canvas: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    *,
    fill: Color,
    border: Optional[Color] = None,
    border_thickness: int = 1,
    radius: int = 10,
) -> None:
    if border is None or border_thickness <= 0:
        _filled_rounded_rect(canvas, p1, p2, fill, radius)
        return

    _filled_rounded_rect(
        canvas,
        p1,
        p2,
        border,
        radius,
    )

    inset = max(1, int(border_thickness))

    inner_p1 = (p1[0] + inset, p1[1] + inset)
    inner_p2 = (p2[0] - inset, p2[1] - inset)

    _filled_rounded_rect(
        canvas,
        inner_p1,
        inner_p2,
        fill,
        max(0, radius - inset),
    )


def badge(
    canvas: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    *,
    fill: Color,
    foreground: Color = (12, 12, 12),
    scale: float = 0.31,
    height: int = 22,
    padding_x: int = 9,
) -> Tuple[int, int, int, int]:
    (text_w, _), _ = cv2.getTextSize(
        str(text),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        1,
    )

    x, y = int(xy[0]), int(xy[1])
    width = text_w + padding_x * 2

    rect = (
        x,
        y - height + 5,
        x + width,
        y + 5,
    )

    rounded_rect(
        canvas,
        (rect[0], rect[1]),
        (rect[2], rect[3]),
        fill=fill,
        radius=height // 2,
    )

    draw_text(
        canvas,
        text,
        (x + padding_x, y - 2),
        scale=scale,
        color=foreground,
        thickness=1,
    )

    return rect
