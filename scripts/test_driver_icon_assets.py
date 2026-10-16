"""Regression tests for SVG-derived Driver Mode icon assets (no CARLA)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.object_icon_assets import (
    _alpha_mask, draw_icon, resolve_icon,
)


def render(label: str, *, primary: bool = False,
           stale: bool = False) -> np.ndarray:
    canvas = np.full((96, 96, 3), 25, np.uint8)
    assert draw_icon(canvas, (48, 48), label,
                     size=17, primary=primary, stale=stale)
    assert np.array_equal(canvas[:, :, 0], canvas[:, :, 1])
    assert np.array_equal(canvas[:, :, 1], canvas[:, :, 2])
    return canvas


def main() -> None:
    for kind in (
        "EGO", "CAR", "TRUCK", "BUS", "MOTORCYCLE", "BICYCLE",
        "PEDESTRIAN", "PERSON", "MOTORBIKE",
        "SAFE", "UNCONFIRMED", "ROAD CLEAR",
    ):
        normal, stale = render(kind), render(kind, stale=True)
        assert np.count_nonzero(normal[:, :, 0] > 25) > 60, kind
        assert float(stale.mean()) < float(normal.mean()), kind

    assert resolve_icon("BUS") == "TRUCK"
    assert resolve_icon("BICYCLE") == "MOTORCYCLE"
    assert resolve_icon("PERSON") == "PEDESTRIAN"
    assert resolve_icon("MOTORBIKE") == "MOTORCYCLE"
    assert resolve_icon("ROAD CLEAR") == "SAFE"
    assert resolve_icon("ROAD_CLEAR") == "SAFE"
    assert np.array_equal(render("SAFE"), render("ROAD CLEAR"))
    assert not np.array_equal(render("SAFE"), render("UNCONFIRMED"))

    # Explicit semantic tint is reserved for the headline icon.
    highlighted = np.full((96, 96, 3), 25, np.uint8)
    assert draw_icon(
        highlighted, (48, 48), "SAFE", size=22,
        color=(135, 204, 121),
    )
    colored = highlighted.reshape(-1, 3)
    assert np.any((colored[:, 0] != colored[:, 1])
                  & (colored[:, 1] != colored[:, 2]))
    stale_tint = np.full((96, 96, 3), 25, np.uint8)
    assert draw_icon(
        stale_tint, (48, 48), "SAFE", size=22,
        stale=True, color=(135, 204, 121),
    )
    assert np.array_equal(stale_tint[:, :, 0], stale_tint[:, :, 1])
    assert np.array_equal(stale_tint[:, :, 1], stale_tint[:, :, 2])
    assert np.array_equal(render("BUS"), render("TRUCK"))
    assert np.array_equal(render("BICYCLE"), render("MOTORCYCLE"))
    assert np.array_equal(render("PERSON"), render("PEDESTRIAN"))
    assert np.array_equal(render("MOTORBIKE"), render("MOTORCYCLE"))
    assert not np.array_equal(render("CAR"), render("TRUCK"))
    assert not np.array_equal(render("PEDESTRIAN"), render("CAR"))

    # Primary ring is a visual-only state indicator.
    assert not np.array_equal(render("CAR", primary=True), render("CAR"))
    assert not np.array_equal(render("PEDESTRIAN", primary=True),
                              render("PEDESTRIAN"))
    assert _alpha_mask("CAR", 30).shape == (30, 30)

    # Compact Ego should not draw its old 31 px-radius halo.
    ego = render("EGO")
    assert np.all(ego[48, 48 + 29] == 25)
    assert np.any(ego[48, 48 + 23] != 25)

    unknown = np.full((96, 96, 3), 25, np.uint8)
    assert not draw_icon(unknown, (48, 48), "UNKNOWN")
    assert np.all(unknown == 25)

    # Clip safely when a tracked target is at the map boundary.
    edge = np.full((60, 60, 3), 25, np.uint8)
    assert draw_icon(edge, (0, 0), "TRUCK", primary=True)
    print("SVG SAFE/UNCONFIRMED ICON ASSET TEST PASSED")


if __name__ == "__main__":
    main()
