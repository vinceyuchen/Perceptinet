"""Offline synthetic showcase of target-first Driver cards and urgency colors.

The preview is never a live CARLA frame. Road/object data is illustrative.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.app import PerceptinetDashboardApp
from test_dashboard_app import build_snapshot


def make_snapshot(level: str, target: str, *, emergency: bool = False,
                  stopped: bool = False):
    snapshot = copy.deepcopy(build_snapshot())
    snapshot["_preview_recorded"] = True
    snapshot["hazard"]["control_level"] = level
    snapshot["hazard"]["observed_level"] = level
    primary = snapshot["hazard"]["primary"]
    if level == "NORMAL":
        snapshot["hazard"]["primary"] = None
    else:
        primary["hazard_level"] = level
        primary["class_name"] = target.lower()
        snapshot["fusion"]["objects"][0]["class_name"] = target.lower()
    braking = snapshot["braking"]
    braking["emergency"] = emergency
    braking["active"] = level != "NORMAL"
    snapshot["ego"]["speed_mps"] = 0.0 if stopped else 3.5
    braking["target_speed_mps"] = 0.0 if stopped else 3.45
    return snapshot


def main():
    unconfirmed = make_snapshot("WARNING", "CAR")
    unconfirmed["hazard"]["primary"]["longitudinal_m"] = 0.0
    cases = [
        ("normal", make_snapshot("NORMAL", "CAR")),
        ("stopped", make_snapshot("NORMAL", "CAR", stopped=True)),
        ("caution", make_snapshot("CAUTION", "PEDESTRIAN")),
        ("warning", make_snapshot("WARNING", "TRUCK")),
        ("emergency", make_snapshot("CRITICAL", "PEDESTRIAN", emergency=True)),
        ("unconfirmed", unconfirmed),
    ]
    app = PerceptinetDashboardApp(default_mode="DRIVER")
    output = ROOT / "results" / "dashboard_polish_preview"
    output.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for name, snapshot in cases:
        frame = app.render(snapshot, stale=False)
        path = output / f"driver_mode_v71_{name}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Failed to save {path}")
        thumbs.append(cv2.resize(frame, (800, 450), interpolation=cv2.INTER_AREA))
    contact = np.vstack((
        np.hstack(thumbs[:2]),
        np.hstack(thumbs[2:4]),
        np.hstack(thumbs[4:6]),
    ))
    contact_path = output / "driver_mode_v71_status_comparison.png"
    if not cv2.imwrite(str(contact_path), contact):
        raise RuntimeError(f"Failed to save {contact_path}")
    print("SYNTHETIC DRIVER CARD V7.1 PREVIEWS READY")
    print("Non-live demonstration only; status colors and class icons are illustrative.")
    print(f"Overview: {contact_path}")


if __name__ == "__main__":
    main()
