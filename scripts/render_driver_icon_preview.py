"""Render all seven Driver Mode SVG-derived icons without CARLA running."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.object_icon_assets import draw_icon

KINDS = [
    ("EGO", "OWN VEHICLE"),
    ("CAR", "PASSENGER CAR"),
    ("TRUCK", "HEAVY VEHICLE"),
    ("BUS", "USES TRUCK ICON"),
    ("MOTORCYCLE", "TWO-WHEELER"),
    ("BICYCLE", "USES MOTORCYCLE ICON"),
    ("PEDESTRIAN", "ROAD USER"),
]


def text(canvas, words, x, y, size=0.6, color=(210, 210, 210)):
    cv2.putText(canvas, words, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def main():
    board = np.full((620, 1280, 3), 16, np.uint8)
    text(board, "PERCEPTINET / DRIVER ICON SYSTEM", 45, 55, 0.91,
         (245, 245, 245))
    text(board, "USER-SUPPLIED SVG ARTWORK  /  GREYSCALE  /  NO CARLA REQUIRED",
         48, 81, 0.42, (160, 160, 160))

    for index, (kind, description) in enumerate(KINDS):
        col, row = index % 4, index // 4
        x, y = 38 + col * 310, 117 + row * 238
        cv2.rectangle(board, (x, y), (x+292, y+222),
                      (55, 55, 55), 1, cv2.LINE_AA)
        text(board, kind, x+15, y+28, 0.61, (242, 242, 242))
        text(board, description, x+15, y+47, 0.39, (150, 150, 150))
        for j, state in enumerate(("NORMAL", "PRIMARY", "STALE")):
            center = (x+52+j*91, y+112)
            draw_icon(board, center, kind, size=17,
                      primary=state == "PRIMARY", stale=state == "STALE")
            text(board, state, center[0]-29, y+168, 0.35,
                 (184, 184, 184))
    folder = ROOT / "results" / "dashboard_polish_preview"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "driver_mode_v70_icon_sheet.png"
    if not cv2.imwrite(str(output), board):
        raise RuntimeError(f"Cannot write {output}")
    print(f"DRIVER SVG ICON SHEET READY: {output}")


if __name__ == "__main__":
    main()
