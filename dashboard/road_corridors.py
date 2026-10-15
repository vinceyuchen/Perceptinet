"""Simplified street corridor outlines for the Driver Mode.

CARLA gives sampled lane centerlines, NOT finished street boundaries.
This module estimates dominant local street corridors and draws only two
outer edge lines per corridor. It never draws a filled road surface.

This presentation intentionally omits turn connector lanes and minor roads;
it is a schematic of dominant nearby corridors, not an HD map.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np


def draw_road_corridors(
    canvas: np.ndarray,
    segments: List[Dict[str, Any]],
    project: Callable[[float, float], Tuple[int, int]],
    *,
    clip_rect: Tuple[int, int, int, int] = (54, 142, 800, 682),
) -> bool:
    """Draw the dominant longitudinal and cross streets as unfilled lines.

    Coordinates in input segments: [lateral_m, forward_m] relative to Ego.
    Intermediate width estimates are derived from lane center positions plus
    half lane width. Turning connectors and isolated stubs are suppressed.
    """
    if not isinstance(segments, list):
        return False
    x0, y0, x1, y1 = clip_rect
    layer = canvas[y0:y1, x0:x1]
    if layer.size == 0:
        return False

    groups: Dict[str, Dict[int, List[Tuple[float, float]]]] = {
        "forward": defaultdict(list),
        "cross": defaultdict(list),
    }
    for segment in segments[:400]:
        if not isinstance(segment, dict):
            continue
        points = segment.get("points")
        if not isinstance(points, list) or len(points) != 2:
            continue
        try:
            (lat0, fwd0), (lat1, fwd1) = points
            lat0, fwd0, lat1, fwd1 = (
                float(lat0), float(fwd0), float(lat1), float(fwd1)
            )
            width = float(segment.get("width_m", 3.5))
        except (ValueError, TypeError, IndexError):
            continue
        if (not all(math.isfinite(v) for v in
                    (lat0, fwd0, lat1, fwd1, width))
                or not 1.0 <= width <= 10.0):
            continue
        dlat, dfwd = lat1 - lat0, fwd1 - fwd0
        if math.hypot(dlat, dfwd) < 0.25:
            continue
        # Ignore turning connectors in the junction interior. Do not
        # convert them into extra street ends or diagonal stubs.
        direction = (
            "forward" if abs(dfwd) >= 1.35 * abs(dlat)
            else "cross" if abs(dlat) >= 1.35 * abs(dfwd)
            else None
        )
        if direction is None:
            continue
        along0, along1 = (
            (fwd0, fwd1) if direction == "forward"
            else (lat0, lat1)
        )
        samples = max(2, int(math.ceil(abs(along1 - along0) / 0.9)) + 1)
        for index in range(samples):
            t = index / (samples - 1)
            lat = lat0 + t * dlat
            fwd = fwd0 + t * dfwd
            along, perpendicular = (
                (fwd, lat) if direction == "forward"
                else (lat, fwd)
            )
            groups[direction][int(round(along / 2.0))].append(
                (perpendicular, width)
            )

    profiles: Dict[str, np.ndarray] = {}
    for direction, bins in groups.items():
        keys = sorted(bins)
        runs: List[List[int]] = []
        current: List[int] = []
        for key in keys:
            if current and key - current[-1] > 2:
                runs.append(current)
                current = []
            current.append(key)
        if current:
            runs.append(current)
        runs = [run for run in runs if (run[-1] - run[0]) * 2 >= 8]
        if not runs:
            continue
        # Dominant corridor only. Minor alleys and disconnected lane
        # stubs should not dominate this driver-facing local road view.
        chosen = max(
            runs,
            key=lambda run: (run[-1] - run[0]) * 2 + min(len(run), 25),
        )
        rows = []
        for key in chosen:
            items = bins[key]
            mid = float(np.median([value for value, _ in items]))
            # Suppress isolated connector samples far from the street
            # centre without erasing normal adjacent driving lanes.
            items = [
                (value, width)
                for value, width in items if abs(value - mid) <= 8.5
            ]
            if not items:
                continue
            values = [value for value, _ in items]
            half = float(np.median([width for _, width in items])) * 0.5
            rows.append((
                key * 2.0,
                float(np.quantile(values, 0.10)) - half,
                float(np.quantile(values, 0.90)) + half,
            ))
        if len(rows) < 4:
            continue
        raw = np.asarray(rows, dtype=float)
        smooth = raw.copy()
        for col in (1, 2):
            for i in range(len(raw)):
                smooth[i, col] = float(np.median(
                    raw[max(0, i - 2):min(len(raw), i + 3), col]
                ))
        profiles[direction] = smooth

    if not profiles:
        return False

    def nearest(direction: str, axis: float):
        profile = profiles.get(direction)
        if profile is None:
            return None
        if axis < profile[0, 0] - 3 or axis > profile[-1, 0] + 3:
            return None
        return (
            float(np.interp(axis, profile[:, 0], profile[:, 1])),
            float(np.interp(axis, profile[:, 0], profile[:, 2])),
        )

    painted = 0

    def stroke(points: List[Tuple[float, float]]) -> None:
        nonlocal painted
        if len(points) < 2:
            return
        coords = [
            (int(project(lat, fwd)[0] - x0),
             int(project(lat, fwd)[1] - y0))
            for lat, fwd in points
        ]
        for a, b in zip(coords[:-1], coords[1:]):
            cv2.line(layer, a, b, (78, 96, 112), 3, cv2.LINE_AA)
            cv2.line(layer, a, b, (164, 175, 186), 1, cv2.LINE_AA)
            painted += 1

    for direction, profile in profiles.items():
        other = "cross" if direction == "forward" else "forward"
        for boundary in (1, 2):
            path: List[Tuple[float, float]] = []
            for along, lo, hi in profile:
                edge = lo if boundary == 1 else hi
                crossing = nearest(other, float(edge))
                internal = (
                    crossing is not None
                    and crossing[0] - 0.7 <= along <= crossing[1] + 0.7
                )
                if internal:
                    stroke(path)
                    path = []
                else:
                    path.append(
                        (float(edge), float(along))
                        if direction == "forward"
                        else (float(along), float(edge))
                    )
            stroke(path)

    return painted > 0
