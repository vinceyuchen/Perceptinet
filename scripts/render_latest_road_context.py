"""Render the latest actual CARLA road-context snapshot without CARLA running.

This is a recorded snapshot preview. It is not evidence of current live status.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.app import PerceptinetDashboardApp


def choose_snapshot(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    folder = ROOT / "results" / "validation"
    candidates = sorted(
        folder.glob("*/dashboard_snapshot.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        road = data.get("road_context") or {}
        if road.get("segments"):
            return path
    raise FileNotFoundError(
        "No saved dashboard_snapshot.json with road_context found in "
        "results/validation. Run a CARLA v6.1 demo first."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=None)
    args = parser.parse_args()
    path = choose_snapshot(args.snapshot)
    snapshot = json.loads(path.read_text(encoding="utf-8-sig"))
    road = snapshot.get("road_context") or {}
    app = PerceptinetDashboardApp(default_mode="DRIVER")
    output = ROOT / "results" / "dashboard_polish_preview"
    output.mkdir(parents=True, exist_ok=True)

    # Never portray a recorded snapshot as currently LIVE.
    stale_path = output / "driver_mode_v68_recorded.png"
    if not cv2.imwrite(str(stale_path), app.render(snapshot, stale=True)):
        raise RuntimeError(f"Failed to write {stale_path}")

    # This is an old snapshot. Render non-stale colors only to inspect
    # layout, with an unmistakable PREVIEW pill instead of LIVE.
    layout_snapshot = dict(snapshot)
    layout_snapshot["_preview_recorded"] = True
    layout_path = output / "driver_mode_v68_layout_preview.png"
    if not cv2.imwrite(
        str(layout_path), app.render(layout_snapshot, stale=False)
    ):
        raise RuntimeError(f"Failed to write {layout_path}")

    segments = road.get("segments") or []
    junctions = sum(bool(s.get("junction")) for s in segments)
    print("=== PERCEPTINET V6.8 ROAD VIEW ===")
    print(f"Recorded snapshot: {path}")
    print(f"Map: {road.get('map_name', 'unknown')}")
    print(f"Road segments: {len(segments)}")
    view = road.get("view_range_m") or {}
    print(
        "Snapshot coverage (recorded): "
        f"forward {view.get('forward', '?')} m, "
        f"behind {view.get('behind', '?')} m, "
        f"side +/-{view.get('side', '?')} m"
    )
    print(
        "Note: older saved snapshots retain their original CARLA "
        "sampling range. The expanded range requires a fresh run."
    )
    print(f"Junction-labelled segments: {junctions}")
    print(f"Recorded-data image: {stale_path}")
    print(f"Recorded layout illustration (PREVIEW): {layout_path}")
    print("ROAD VIEW RENDERED. Inspect both files; do not treat the")
    print("layout-only preview as live validation evidence.")
    print("Extended schematic road lines beyond recorded sampling ends;")
    print("extrapolated portions are illustrative, not observed CARLA roads.")


if __name__ == "__main__":
    main()
