"""CARLA-free regression for v6.4 line-only road corridor display."""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.road_corridors import draw_road_corridors


def main():
    segments = []

    # Two multi-lane roads crossing in Ego's local reference frame.
    for lateral in (-3.5, 0.0, 3.5):
        for forward in np.arange(-8.0, 40.0, 3.0):
            segments.append({
                "points": [[float(lateral), float(forward)],
                           [float(lateral), float(forward+3.0)]],
                "width_m": 3.5,
                "junction": True,
            })
    for forward in (7.0, 10.5, 14.0):
        for lateral in np.arange(-18.0, 18.0, 3.0):
            segments.append({
                "points": [[float(lateral), float(forward)],
                           [float(lateral+3.0), float(forward)]],
                "width_m": 3.5,
                "junction": True,
            })

    # Short curved turning connectors must not create extra street prongs.
    for angle in np.linspace(0, math.pi / 2, 12):
        next_angle = angle + math.pi / 24
        segments.append({
            "points": [
                [7.0+6*math.cos(angle), 18.0+6*math.sin(angle)],
                [7.0+6*math.cos(next_angle), 18.0+6*math.sin(next_angle)]
            ],
            "width_m": 3.5,
            "junction": True,
        })

    view = np.zeros((720, 1280, 3), dtype=np.uint8)
    project = lambda lat, fwd: (
        int(round(430 + lat * 10.0)),
        int(round(590 - fwd * 10.2)),
    )

    assert draw_road_corridors(view, segments, project)
    # No asphalt/ground/terrain fill: road interior remains transparent.
    assert not np.any(view[335, 430])
    assert not np.any(view[488, 430])
    # Left/right edge of long road visible outside the intersection.
    assert np.any(view[330:342, 374:384])
    assert np.any(view[330:342, 479:489])
    # Cross-street edge visible left and right of the intersection.
    assert np.any(view[424:437, 294:307])
    assert np.any(view[533:541, 570:585])
    # The boundary lines stop inside the junction rather than crossing it.
    assert not np.any(view[482:495, 374:384])
    # Tiny cap on total pixels catches regressions to filled surfaces.
    assert np.count_nonzero(view[:, :, 0]) < 15000

    empty = np.zeros_like(view)
    assert not draw_road_corridors(empty, [], project)
    print("LINE-ONLY ROAD CORRIDOR TEST PASSED")


if __name__ == "__main__":
    main()
