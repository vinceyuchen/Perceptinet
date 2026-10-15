"""Monochrome road-user icons precompiled from user-supplied SVG artwork.

CAR, TRUCK, MOTORCYCLE, PEDESTRIAN and EGO were rasterized into
48x48 one-bit transparent masks at development time. No SVG renderer,
file-system asset paths or new Python dependency is needed on the VM.
The original uploads included front-view car/truck illustrations; this
module deliberately preserves their selected artwork. Presentation only.
"""

from __future__ import annotations

import base64
import zlib
from functools import lru_cache
from typing import Tuple

import cv2
import numpy as np

# Compressed row-major 48x48 monochrome 1-bit masks, Base85-encoded.
# Artwork: uploaded ego.svg, car-svgrepo-com.svg, truck.svg,
# motorcycle-svgrepo-com.svg, pedestr.svg.
_MASKS = {
    "EGO": "c-lqKu?++<2n4`8IdJwa!KKjBcfghAQe1*V2Z%lGBi;lPq1D25#gswoq-`DhK8hXHkHFs>c+vulv;s42z)Cx?Qvobhz*B(HKX3Ykg^sY&88*7YPIoA<4im1KGQ`<lU0J}D^;}rqK0N0ee|Z5RbchE",
    "CAR": "c-oE7y$u5)3<cnaQcyC2H#rtV2cu;aMxb<o@cD&1DNUrf_zM5YGO$C`NoE7cW)rAp3os9NBt!5HX~7%PV+Udrs+F#9GoA7VUK(lNChi<Ub(`tD^_0$&mz;$jC7SiePi9vu?_8Br|Ji`>TR!3Tpk@E%0z|5nXa",
    "TRUCK": "c-pPdu?@f=3<SVOE*OJR7|lpoj8s_zDN9fwoP0<DMY_1S!tz^|fv4;)IGTj!#RB*&8IT_gv?}ehk+!dt=M5$Bo*HRW+Fwtzqw8+At&aXQIkIDq?|>GedDb-`rtz2ycjb!)",
    "MOTORCYCLE": "c-nQ)u?+(u3<Xf5D<~O(P2$BO!Dx)a2$U`mKmNy^PKpu>y~4&eLS#qkR42u9!c;m^6-AG*f)+7P*O{3Tx>qGwuI$1yq0j($J_0^E1Lj3|lN4|PJb2}{9*^ypBhh`ly&Ws|;^yJbY*bDE8fcZb+KcZr`7&<@js5x-`2ZpIf$I",
    "PEDESTRIAN": "c-no=F%H5o3<Xd}9Wwd`_$K693`owFkt=oM8mW4Luz84LY!$UdLj2(kUrua*Eg`$uT$01o|9PH_6@KLka0g(a0ic#|S=D+{wJ#>s*&gtK-EGUMxv5?tG*!&et@s-JXtWG2lGdJ{o!!B!khwH0W{qW}7BY5{X(quQYiiMNsKL#jORTE<Ue6nmsBF^",
}

ICON_ALIASES = {
    "BUS": "TRUCK",
    "BICYCLE": "MOTORCYCLE",
    "PERSON": "PEDESTRIAN",
    "MOTORBIKE": "MOTORCYCLE",
}


def resolve_icon(kind: str) -> str:
    return ICON_ALIASES.get(str(kind).upper(), str(kind).upper())


@lru_cache(maxsize=128)
def _alpha_mask(kind: str, side: int) -> np.ndarray | None:
    source = _MASKS.get(resolve_icon(kind))
    if source is None:
        return None
    packed = zlib.decompress(base64.b85decode(source))
    raw = np.unpackbits(np.frombuffer(packed, np.uint8)).reshape(48, 48)
    raw = cv2.GaussianBlur((raw * 255).astype(np.uint8), (3, 3), 0.45)
    return cv2.resize(raw, (side, side),
                      interpolation=cv2.INTER_AREA if side < 48
                      else cv2.INTER_CUBIC)


def draw_icon(canvas: np.ndarray, center: Tuple[int, int], kind: str,
              *, size: int = 13, primary: bool = False,
              stale: bool = False) -> bool:
    """Alpha-composite a monochrome SVG-derived marker into a BGR dashboard.

    Stale tones are dimmed; a primary target gets a visible surrounding ring.
    The mapping never affects fusion, object classification or braking.
    """
    kind = resolve_icon(kind)
    if kind not in _MASKS:
        return False
    cx, cy = center
    if kind == "EGO":
        # Driver Map v7.2: compact self marker to avoid dominating targets.
        cv2.circle(canvas, center, 24, (43, 43, 43), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 24, (130, 130, 130), 1, cv2.LINE_AA)
        side = 31
    elif kind == "PEDESTRIAN":
        if primary:
            cv2.circle(canvas, center, size + 8,
                       (221, 221, 221), 2, cv2.LINE_AA)
        side = max(20, 2 * size)
    else:
        if primary:
            cv2.circle(canvas, center, size + 7,
                       (224, 224, 224), 2, cv2.LINE_AA)
        side = max(22, 2 * size)
    alpha = _alpha_mask(kind, side)
    if alpha is None:
        return False
    x0, y0 = cx - side // 2, cy - side // 2
    x1, y1 = x0 + side, y0 + side
    left, top = max(0, x0), max(0, y0)
    right, bottom = min(canvas.shape[1], x1), min(canvas.shape[0], y1)
    if right <= left or bottom <= top:
        return True
    a = alpha[top - y0:bottom - y0,
              left - x0:right - x0].astype(np.float32) / 255.0
    if stale:
        a *= 0.55
    ink = 202 if stale else 247 if kind == "EGO" or primary else 218
    region = canvas[top:bottom, left:right]
    region[:] = np.rint(region.astype(np.float32) * (1-a[..., None])
                        + ink * a[..., None]).astype(np.uint8)
    return True
