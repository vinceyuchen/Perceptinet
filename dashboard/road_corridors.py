"""Stylized road centerline schematic for Perceptinet Driver Mode.

CARLA provides sampled lane centerlines, not finished street boundaries.
Infer a dominant forward street and crossing street, smooth their centerlines,
then render constant-width paired edges. No road surface or buildings.
The resulting map is illustrative, not survey-accurate.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


# Fixed schematic width, independent of sampled CARLA lane widths.
# Slightly wider *visual* road tube at the v6.6 camera framing.
# This remains independent of the CARLA lane width in metres.
HALF_STREET_WIDTH_M = 5.3
SAMPLE_BIN_M = 2.0
MIN_ROAD_SPAN_M = 12.0


def _dominant_centerline(
    samples: List[Tuple[float, float]]
) -> Optional[np.ndarray]:
    """Infer smooth [along, offset] road axis from noisy lane centre points."""
    if len(samples) < 8:
        return None
    values = np.asarray(samples, dtype=np.float64)
    global_center = float(np.median(values[:, 1]))
    values = values[np.abs(values[:, 1] - global_center) <= 8.0]
    if len(values) < 8:
        return None
    bins: Dict[int, List[float]] = defaultdict(list)
    for along, offset in values:
        bins[int(round(along / SAMPLE_BIN_M))].append(float(offset))
    rows = np.asarray(
        [(key * SAMPLE_BIN_M, float(np.median(offsets)))
         for key, offsets in sorted(bins.items())], dtype=np.float64,
    )
    if len(rows) < 5:
        return None

    runs: List[np.ndarray] = []
    begin = 0
    for i in range(1, len(rows)):
        if rows[i, 0] - rows[i - 1, 0] > 4.1:
            runs.append(rows[begin:i])
            begin = i
    runs.append(rows[begin:])
    rows = max(runs, key=lambda run: (run[-1, 0] - run[0, 0], len(run)))
    extent = float(rows[-1, 0] - rows[0, 0])
    if extent < MIN_ROAD_SPAN_M or len(rows) < 5:
        return None

    u, v = rows[:, 0], rows[:, 1]
    origin = float(np.mean(u))
    scale = max(1.0, extent * 0.5)
    u = (u - origin) / scale
    keep = np.ones(len(u), dtype=bool)
    coefficients = np.polynomial.polynomial.polyfit(u, v, deg=2)
    for _ in range(3):
        coefficients = np.polynomial.polynomial.polyfit(
            u[keep], v[keep], deg=min(2, np.count_nonzero(keep) - 1)
        )
        errors = v - np.polynomial.polynomial.polyval(u, coefficients)
        median_error = float(np.median(errors[keep]))
        median_dev = float(np.median(np.abs(errors[keep] - median_error)))
        cutoff = max(2.2, 3.0 * 1.4826 * median_dev)
        next_keep = np.abs(errors - median_error) <= cutoff
        if np.count_nonzero(next_keep) < 5 or np.array_equal(next_keep, keep):
            break
        keep = next_keep

    alongs = np.arange(float(rows[0, 0]), float(rows[-1, 0]) + 0.01, 0.5)
    centers = np.polynomial.polynomial.polyval(
        (alongs - origin) / scale, coefficients
    )
    return np.column_stack((alongs, centers))



def _extend_profile(
    profile: np.ndarray,
    target_min: float,
    target_max: float,
    *,
    step: float = 0.5,
) -> np.ndarray:
    """Visually continue a fitted street to the Road View's clipping edges.

    Use local endpoint tangents only beyond the last observed samples. Cap
    extrapolated slope to avoid inventing sharply turning roads. This
    extends *presentation only*, not CARLA road/topology observations.
    """
    if profile is None or len(profile) < 2:
        return profile

    def endpoint_slope(points: np.ndarray) -> float:
        axis_span = float(points[-1, 0] - points[0, 0])
        if abs(axis_span) < 1e-6:
            return 0.0
        slope = float((points[-1, 1] - points[0, 1]) / axis_span)
        return float(np.clip(slope, -0.35, 0.35))

    head = profile[:min(7, len(profile))]
    tail = profile[-min(7, len(profile)):]
    first, last = profile[0], profile[-1]
    head_slope = endpoint_slope(head)
    tail_slope = endpoint_slope(tail)

    before = np.arange(
        float(first[0]) - step, target_min - step * 0.5, -step,
        dtype=np.float64,
    ) if target_min < first[0] else np.empty(0, dtype=np.float64)
    after = np.arange(
        float(last[0]) + step, target_max + step * 0.5, step,
        dtype=np.float64,
    ) if target_max > last[0] else np.empty(0, dtype=np.float64)

    parts = []
    if len(before):
        before = before[::-1]
        parts.append(np.column_stack((
            before, float(first[1]) + (before - first[0]) * head_slope,
        )))
    parts.append(profile)
    if len(after):
        parts.append(np.column_stack((
            after, float(last[1]) + (after - last[0]) * tail_slope,
        )))
    return np.vstack(parts)


def _sample_roads(segments: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    groups: Dict[str, List[Tuple[float, float]]] = {
        "forward": [], "cross": [],
    }
    for segment in segments[:500]:
        if not isinstance(segment, dict):
            continue
        points = segment.get("points")
        if not isinstance(points, (tuple, list)) or len(points) != 2:
            continue
        try:
            (lat0, fwd0), (lat1, fwd1) = points
            lat0, fwd0, lat1, fwd1 = (
                float(lat0), float(fwd0), float(lat1), float(fwd1)
            )
        except (ValueError, TypeError, IndexError):
            continue
        if not all(math.isfinite(x) for x in (lat0, fwd0, lat1, fwd1)):
            continue
        dlat, dfwd = lat1 - lat0, fwd1 - fwd0
        length = math.hypot(dlat, dfwd)
        if not 0.4 <= length <= 15.0:
            continue
        # Turning connectors must not create street stubs.
        if abs(dfwd) >= 1.6 * abs(dlat):
            direction = "forward"
        elif abs(dlat) >= 1.6 * abs(dfwd):
            direction = "cross"
        else:
            continue
        count = max(2, int(length / 1.0) + 1)
        for i in range(count):
            t = i / (count - 1)
            lat = lat0 + dlat * t
            fwd = fwd0 + dfwd * t
            groups[direction].append(
                (fwd, lat) if direction == "forward" else (lat, fwd)
            )
    return {
        direction: axis
        for direction, samples in groups.items()
        if (axis := _dominant_centerline(samples)) is not None
    }


def draw_road_corridors(
    canvas: np.ndarray,
    segments: List[Dict[str, Any]],
    project: Callable[[float, float], Tuple[int, int]],
    *,
    clip_rect: Tuple[int, int, int, int] = (54, 142, 800, 682),
) -> bool:
    """Render fixed-width paired street edges without road/terrain fill."""
    if not isinstance(segments, list) or not segments:
        return False
    profiles = _sample_roads(segments)
    if not profiles:
        return False
    # Extend only fitted schematic centerlines, never raw CARLA segments.
    # Projected coordinates are clipped to the Road View panel below.
    if "forward" in profiles:
        profiles["forward"] = _extend_profile(
            profiles["forward"], -12.0, 62.0
        )
    if "cross" in profiles:
        profiles["cross"] = _extend_profile(
            profiles["cross"], -45.0, 45.0
        )
    x0, y0, x1, y1 = clip_rect
    layer = canvas[y0:y1, x0:x1]
    if layer.size == 0:
        return False

    def street_center(direction: str, along: float) -> Optional[float]:
        profile = profiles.get(direction)
        if profile is None or along < profile[0, 0] or along > profile[-1, 0]:
            return None
        return float(np.interp(along, profile[:, 0], profile[:, 1]))

    count = 0
    # Keep only the ends beside the *junction gap*, not screen-border ends.
    # Each end stores its adjacent sample for a reliable local tangent.
    junction_ends: List[Tuple[str, Tuple[float, float], Tuple[float, float]]] = []

    def stroke(points: List[Tuple[float, float]]) -> None:
        nonlocal count
        if len(points) < 2:
            return
        coords = np.asarray(
            [(project(lat, fwd)[0] - x0, project(lat, fwd)[1] - y0)
             for lat, fwd in points], dtype=np.int32
        )
        cv2.polylines(layer, [coords], False, (138, 138, 138), 4, cv2.LINE_AA)
        cv2.polylines(layer, [coords], False, (70, 70, 70), 1, cv2.LINE_AA)
        count += len(points) - 1

    for direction, profile in profiles.items():
        for side in (-1.0, 1.0):
            path: List[Tuple[float, float]] = []
            after_gap = False
            pending_start = False
            for i, (along, center) in enumerate(profile):
                prev = profile[max(0, i - 1)]
                nxt = profile[min(len(profile) - 1, i + 1)]
                delta = float(nxt[0] - prev[0])
                slope = float(nxt[1] - prev[1]) / delta if delta else 0.0
                norm = math.hypot(1.0, slope)
                # Normal offset gives consistent visual width on curves.
                perpendicular = float(
                    center + side * HALF_STREET_WIDTH_M / norm
                )
                shifted_along = float(
                    along - side * HALF_STREET_WIDTH_M * slope / norm
                )
                if direction == "forward":
                    lateral, forward = perpendicular, shifted_along
                    cross = street_center("cross", lateral)
                    within_junction = (
                        cross is not None
                        and abs(forward - cross) < HALF_STREET_WIDTH_M + 0.4
                    )
                else:
                    lateral, forward = shifted_along, perpendicular
                    main = street_center("forward", forward)
                    within_junction = (
                        main is not None
                        and abs(lateral - main) < HALF_STREET_WIDTH_M + 0.4
                    )
                if within_junction:
                    if len(path) >= 2:
                        junction_ends.append((direction, path[-1], path[-2]))
                    stroke(path)
                    path = []
                    after_gap = True
                    pending_start = False
                else:
                    if not path and after_gap:
                        pending_start = True
                        after_gap = False
                    path.append((lateral, forward))
                    if pending_start and len(path) >= 2:
                        junction_ends.append((direction, path[0], path[1]))
                        pending_start = False
            stroke(path)

    # Smoothly connect the four street *outer corners* rather than leaving
    # disconnected perpendicular line ends. Matching is done by proximity
    # and road orientation so single-road and T-junction cases are safe.
    candidates = []
    for i, (di, pi, ai) in enumerate(junction_ends):
        for j in range(i + 1, len(junction_ends)):
            dj, pj, aj = junction_ends[j]
            if di == dj:
                continue
            distance = math.hypot(pi[0] - pj[0], pi[1] - pj[1])
            if distance <= 6.5:
                candidates.append((distance, i, j))
    used = set()
    for _, i, j in sorted(candidates):
        if i in used or j in used:
            continue
        _, start, adjacent_start = junction_ends[i]
        _, end, adjacent_end = junction_ends[j]

        def projected(p: Tuple[float, float]) -> np.ndarray:
            px, py = project(*p)
            return np.asarray((float(px - x0), float(py - y0)))

        p0, p2 = projected(start), projected(end)
        t0 = p0 - projected(adjacent_start)
        t2 = p2 - projected(adjacent_end)
        determinant = float(np.linalg.det(np.column_stack((t0, -t2))))
        control = (p0 + p2) / 2.0
        if abs(determinant) > 1e-4:
            coeff = np.linalg.solve(
                np.column_stack((t0, -t2)), p2 - p0
            )
            candidate = p0 + coeff[0] * t0
            if (np.isfinite(candidate).all()
                    and np.linalg.norm(candidate - p0) < 95
                    and np.linalg.norm(candidate - p2) < 95):
                control = candidate
        # Quadratic Bézier matches both approaching road tangents.
        arc = np.asarray([
            (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t ** 2 * p2
            for t in np.linspace(0.0, 1.0, 17)
        ], dtype=np.int32)
        cv2.polylines(layer, [arc], False, (138, 138, 138), 4, cv2.LINE_AA)
        cv2.polylines(layer, [arc], False, (70, 70, 70), 1, cv2.LINE_AA)
        used.update((i, j))
    return count > 0
