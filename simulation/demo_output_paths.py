"""Isolated run artifacts for the live looping teacher demonstration.

A per-cycle directory prevents an Excel-open CSV or a previous demo
instance from blocking the next scene's metrics writes. The live dashboard
snapshot deliberately keeps its established repository-level location.
This module has no CARLA, YOLO, or OpenCV dependency.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple


def create_teacher_demo_paths(
    repo_root: Path, cycle_index: str | int
) -> Tuple[Path, Path]:
    """Return a unique metrics directory and the stable Dashboard snapshot."""
    index = str(cycle_index).strip()
    if not index.isdecimal() or int(index) < 1 or int(index) > 999999:
        raise ValueError("Teacher Demo cycle index must be 1..999999")

    output_root = Path(repo_root) / "results"
    runs_root = output_root / "teacher_demo" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    # mkdtemp atomically creates a unique directory even across launches.
    # Never overwrite historical evidence, and never require deleting a CSV.
    run_dir = Path(tempfile.mkdtemp(
        prefix=f"cycle_{int(index):04d}_", dir=str(runs_root)
    ))
    dashboard_snapshot = output_root / "dashboard_snapshot.json"
    return run_dir, dashboard_snapshot
