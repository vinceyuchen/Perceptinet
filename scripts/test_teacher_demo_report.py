"""No-CARLA regression for the tracked report generator and missing-data case."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import generate_teacher_demo_report as report
from simulation.teacher_scenarios import (
    CROSSING_TRIGGER_DISTANCE_M, CROSSING_WALK_SPEED_MPS,
    CROSSING_FINISH_LATERAL_M, scenario_for_cycle,
)


def main():
    assert CROSSING_TRIGGER_DISTANCE_M == 11.5
    assert CROSSING_WALK_SPEED_MPS == 0.95
    assert CROSSING_FINISH_LATERAL_M == 2.4
    assert [scenario_for_cycle("alternate", i) for i in (1, 2)] == [
        "clear", "rain_v2i",
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runs = root / "runs"
        output = root / "report"
        # No run files must NOT invent safe stops or collisions.
        assert report.read_runs(runs) == []
        before = report.summarize([])
        assert before["clear"]["cycles"] == 0
        assert before["rain_v2i"]["verified_safe_stop_rate"] is None
        assert report.main([
            "--runs-root", str(runs), "--output", str(output),
        ]) == 0
        assert (output / "Perceptinet_Comparative_Data_Report.csv").is_file()

        clear_dir = runs / "cycle_0001_sample"
        clear_dir.mkdir(parents=True)
        (clear_dir / "scenario_result.json").write_text(json.dumps({
            "completed": True,
            "scenario": "clear",
            "reason": "controller_confirmed_stop",
            "safe_stop_verified": True,
            "crossing_started_sim_s": 4.0,
            "first_brake_s": 3.5,
            "pedestrian_motion_verified": True,
            "pedestrian_walked_m": 2.7,
            "minimum_center_distance_m": 6.3,
            "kinematic_contact_observed": False,
            "collision_sensor_available": True,
            "physical_collision_observed": False,
            "v2i": {"total_sent": 140, "total_dropped": 0},
        }), encoding="utf-8")

        rain_dir = runs / "cycle_0002_sample"
        rain_dir.mkdir(parents=True)
        (rain_dir / "scenario_result.json").write_text(json.dumps({
            "completed": True,
            "scenario": "rain_v2i",
            "reason": "kinematic_contact",
            "safe_stop_verified": False,
            "crossing_started_sim_s": 5.0,
            "first_brake_s": None,
            "pedestrian_motion_verified": True,
            "pedestrian_walked_m": 2.5,
            "minimum_center_distance_m": 1.0,
            "kinematic_contact_observed": True,
            "collision_sensor_available": False,
            "physical_collision_observed": None,
            "v2i": {"outage_dropped": 40, "random_dropped": 6},
        }), encoding="utf-8")
        rows = report.read_runs(runs)
        assert len(rows) == 2
        assert rows[0]["safe_stop_verified"] is True
        assert rows[0]["brake_before_crossing_s"] == 0.5
        assert rows[1]["kinematic_contact_observed"] is True
        assert rows[1]["physical_collision_observed"] is None
        summary = report.summarize(rows)
        assert summary["clear"]["verified_safe_stops"] == 1
        assert summary["rain_v2i"]["kinematic_contact_cycles"] == 1
        assert report.main([
            "--runs-root", str(runs), "--output", str(output),
        ]) == 0
        exported = json.loads(
            (output / "Perceptinet_Comparative_Data_Report.json")
            .read_text(encoding="utf-8")
        )
        assert exported["run_count"] == 2
        assert exported["by_scenario"]["clear"]["verified_safe_stops"] == 1
        assert exported["by_scenario"]["rain_v2i"]["kinematic_contact_cycles"] == 1

    print("TEACHER REPORT/COMPARATIVE WALKER CONFIG TEST PASSED")


if __name__ == "__main__":
    main()
