"""No-CARLA regression: teacher cycles cannot collide with opened CSVs."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.demo_output_paths import create_teacher_demo_paths


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        repo = Path(temp)
        old = repo / "results" / "occluded_pedestrian_world_v1.csv"
        old.parent.mkdir(parents=True)
        old.write_text("valuable existing result\n", encoding="utf-8")

        first, snapshot1 = create_teacher_demo_paths(repo, 1)
        second, snapshot2 = create_teacher_demo_paths(repo, 2)
        repeated, snapshot3 = create_teacher_demo_paths(repo, 2)

        assert first != second and second != repeated and first != repeated
        assert snapshot1 == snapshot2 == snapshot3
        assert snapshot1 == repo / "results" / "dashboard_snapshot.json"
        assert all(p.is_dir() for p in (first, second, repeated))
        assert first.parent == second.parent == repeated.parent
        assert first.name.startswith("cycle_0001_")
        assert second.name.startswith("cycle_0002_")
        assert old.read_text(encoding="utf-8") == "valuable existing result\n"

        for folder in (first, second, repeated):
            with (folder / "occluded_pedestrian_world_v1.csv").open(
                "w", encoding="utf-8", newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "source"])
                writer.writerow([7, "RSU-01"])
            (folder / "metrics_summary.json").write_text(
                "{}", encoding="utf-8"
            )
        assert old.read_text(encoding="utf-8") == "valuable existing result\n"

        for invalid in ("0", "-1", "foo", "", "1000000"):
            try:
                create_teacher_demo_paths(repo, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted invalid cycle: {invalid!r}")

    print("TEACHER DEMO PER-CYCLE OUTPUT PATHS TEST PASSED")


if __name__ == "__main__":
    main()
