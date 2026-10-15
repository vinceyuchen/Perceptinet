"""CARLA-free tests for v6.5 smoothed centerline road schematic."""

import math
import sys

import cv2
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.road_corridors import _extend_profile, draw_road_corridors


def project(lat, fwd):
    return (int(round(430 + lat * 10.0)),
            int(round(590 - fwd * 10.2)))


def make_junction():
    segments = []
    # Three centerline tracks share a single forward road corridor.
    for lateral in (-3.5, 0.0, 3.5):
        for forward in np.arange(-8.0, 40.0, 3.0):
            segments.append({
                "points": [[float(lateral), float(forward)],
                           [float(lateral), float(forward + 3)]],
                "width_m": 3.5,
                "junction": True,
            })
    for forward in (7.0, 10.5, 14.0):
        for lateral in np.arange(-18.0, 18.0, 3.0):
            segments.append({
                "points": [[float(lateral), float(forward)],
                           [float(lateral + 3), float(forward)]],
                "width_m": 3.5,
                "junction": True,
            })
    # Short curved turning links should not create unwanted street spurs.
    for angle in np.linspace(0, math.pi / 2, 12):
        nxt = angle + math.pi / 24
        segments.append({
            "points": [
                [7 + 6 * math.cos(angle), 18 + 6 * math.sin(angle)],
                [7 + 6 * math.cos(nxt), 18 + 6 * math.sin(nxt)],
            ],
            "width_m": 3.5,
            "junction": True,
        })
    return segments


def main():
    segments = make_junction()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert draw_road_corridors(frame, segments, project)

    # Fitted street endpoints must continue to all four view boundaries,
    # without changing the original samples or filling the street.
    assert np.any(frame[142:160, 300:560])  # Forward road at top
    assert np.any(frame[664:682, 300:560])  # Forward road at bottom
    assert np.any(frame[240:570, 54:75])    # Cross street at left
    assert np.any(frame[240:570, 778:800])  # Cross street at right

    # Each corner joins the longitudinal and transverse street edge.
    # Four joined corner branches, not eight disconnected line stubs.
    bw = (frame[:, :, 0] > 35).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    long_lines = sum(
        int(area) >= 35 for area in stats[1:, cv2.CC_STAT_AREA]
    )
    assert 1 <= long_lines <= 5, f"Disconnected junction edges: {long_lines}"

    # The main road uses fixed +/-4.5m edges, independent of the number
    # and actual widths of sampled CARLA driving lanes.
    # The v6.6 tube is visually wider, while remaining independent
    # of the sampled CARLA lane widths.
    assert np.any(frame[326:349, 367:387])
    assert np.any(frame[326:349, 473:493])
    assert np.any(frame[416:449, 293:323])
    assert np.any(frame[526:546, 564:588])
    # No road interior, terrain or junction line crossing.
    assert not np.any(frame[488, 430])
    assert not np.any(frame[488, 385])
    assert not np.any(frame[488, 475])
    assert not np.any(frame[340, 430])
    assert np.count_nonzero(frame[:, :, 0]) < 20000

    profile = np.asarray([[0.0, 1.0], [1.0, 1.1],
                          [2.0, 1.2], [3.0, 1.3]])
    original = profile.copy()
    extended = _extend_profile(profile, -4.0, 8.0)
    assert np.array_equal(profile, original)
    assert extended[0, 0] <= -4.0
    assert extended[-1, 0] >= 8.0
    assert np.all(np.diff(extended[:, 0]) > 0)
    assert np.any(np.all(extended == profile[0], axis=1))
    assert np.any(np.all(extended == profile[-1], axis=1))

    wider = [dict(s, width_m=7.0) for s in segments]
    wider_frame = np.zeros_like(frame)
    assert draw_road_corridors(wider_frame, wider, project)
    assert np.array_equal(frame, wider_frame)

    # A curved road can be smoothed without inventing a perpendicular street.
    curved = []
    for forward in np.arange(-7.0, 39.0, 3.0):
        lat = 0.0085 * (forward - 10) ** 2
        next_lat = 0.0085 * (forward - 7) ** 2
        curved.append({
            "points": [[float(lat), float(forward)],
                       [float(next_lat), float(forward + 3)]],
            "width_m": 3.5,
        })
    curve_frame = np.zeros_like(frame)
    assert draw_road_corridors(curve_frame, curved, project)
    assert not np.any(curve_frame[590, 430])

    empty = np.zeros_like(frame)
    assert not draw_road_corridors(empty, [], project)
    assert not draw_road_corridors(
        empty, [{"points": [[float("nan"), 0], [0, 1]]}], project
    )
    print("V6.8 SMOOTH JUNCTION CORRIDOR TEST PASSED")


if __name__ == "__main__":
    main()
