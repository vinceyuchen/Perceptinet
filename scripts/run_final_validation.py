from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.comparison import (
    build_suite_summary,
    compare_pair,
    load_json,
    write_pair_csv,
    write_report,
    write_suite_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run paired Perceptinet baseline/cooperative "
            "CARLA final-system validation."
        )
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of paired baseline/cooperative trials.",
    )
    parser.add_argument(
        "--max-sim-seconds",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--near-miss-m",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show the OpenCV perception preview during validation.",
    )
    parser.add_argument(
        "--suite-id",
        default=None,
    )

    return parser.parse_args()


def run_one(
    *,
    mode: str,
    run_id: str,
    suite_root: Path,
    max_sim_seconds: float,
    near_miss_m: float,
    show_preview: bool,
) -> Path:
    run_dir = suite_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    env = os.environ.copy()
    env.update(
        {
            "PERCEPTINET_VALIDATION": "1",
            "PERCEPTINET_VALIDATION_MODE": mode,
            "PERCEPTINET_VALIDATION_RUN_ID": run_id,
            "PERCEPTINET_VALIDATION_RESULTS_ROOT":
                str(suite_root),
            "PERCEPTINET_VALIDATION_AUTO_EXIT": "1",
            "PERCEPTINET_VALIDATION_HEADLESS":
                "0" if show_preview else "1",
            "PERCEPTINET_VALIDATION_MAX_SIM_SECONDS":
                str(max_sim_seconds),
            "PERCEPTINET_VALIDATION_NEAR_MISS_M":
                str(near_miss_m),
            "PYTHONUNBUFFERED": "1",
        }
    )

    command = [
        sys.executable,
        "-u",
        str(
            REPO_ROOT
            / "scripts"
            / "run_occluded_pedestrian_braking_v1.py"
        ),
    ]

    print()
    print(
        f"=== {run_id} | {mode.upper()} ==="
    )

    with stdout_path.open(
        "w",
        encoding="utf-8",
    ) as stdout_handle, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_handle:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )

    summary_path = run_dir / "validation_summary.json"

    if result.returncode != 0:
        tail = ""

        if stderr_path.exists():
            lines = stderr_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
            tail = "\n".join(lines[-30:])

        raise RuntimeError(
            f"{run_id} failed with exit code "
            f"{result.returncode}\n{tail}"
        )

    if not summary_path.exists():
        raise RuntimeError(
            f"{run_id} did not create "
            f"{summary_path.name}"
        )

    summary = load_json(summary_path)

    print(
        f"Outcome={summary.get('outcome')} | "
        f"min_clearance="
        f"{(summary.get('safety') or {}).get('minimum_clearance_m')} | "
        f"final_clearance="
        f"{(summary.get('safety') or {}).get('final_clearance_m')}"
    )

    return summary_path


def main():
    args = parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")

    suite_id = (
        args.suite_id
        or datetime.now().strftime(
            "final_validation_%Y%m%d_%H%M%S"
        )
    )

    suite_root = (
        REPO_ROOT
        / "results"
        / "validation"
        / suite_id
    )
    suite_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Perceptinet Final System Validation v1"
    )
    print("Suite:", suite_id)
    print("Trials:", args.trials)
    print("Output:", suite_root)
    print(
        "Order per trial: BASELINE control -> "
        "COOPERATIVE experimental"
    )

    pair_rows = []

    for trial in range(1, args.trials + 1):
        suffix = f"{trial:02d}"

        baseline_path = run_one(
            mode="baseline",
            run_id=f"baseline_{suffix}",
            suite_root=suite_root,
            max_sim_seconds=args.max_sim_seconds,
            near_miss_m=args.near_miss_m,
            show_preview=args.show_preview,
        )

        # Give CARLA a brief quiet window after actor cleanup.
        time.sleep(1.0)

        cooperative_path = run_one(
            mode="cooperative",
            run_id=f"cooperative_{suffix}",
            suite_root=suite_root,
            max_sim_seconds=args.max_sim_seconds,
            near_miss_m=args.near_miss_m,
            show_preview=args.show_preview,
        )

        time.sleep(1.0)

        pair = compare_pair(
            load_json(baseline_path),
            load_json(cooperative_path),
            pair_id=f"trial_{suffix}",
        )
        pair_rows.append(pair)

        print(
            f"PAIR {suffix} | "
            f"pass={pair['pair_pass']} | "
            f"clearance_gain="
            f"{pair['clearance_gain_m']} | "
            f"RSU lead="
            f"{pair['cooperative_rsu_detection_lead_s']}"
        )

    summary = build_suite_summary(
        pair_rows
    )

    csv_path = write_pair_csv(
        pair_rows,
        suite_root / "final_validation_results.csv",
    )
    json_path = write_suite_json(
        summary,
        suite_root / "final_validation_summary.json",
    )
    report_path = write_report(
        summary,
        suite_root / "final_validation_report.txt",
    )

    latest_pointer = (
        REPO_ROOT
        / "results"
        / "validation"
        / "latest_suite.json"
    )
    latest_pointer.write_text(
        json.dumps(
            {
                "suite_id": suite_id,
                "suite_root": str(suite_root),
                "summary": str(json_path),
                "results_csv": str(csv_path),
                "report": str(report_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 56)
    print("FINAL VALIDATION COMPLETE")
    print("=" * 56)
    print(
        f"Passed pairs: "
        f"{summary['passed_pairs']}/"
        f"{summary['pair_count']}"
    )
    print(
        "All pairs passed:",
        summary["all_pairs_passed"],
    )
    print(
        "Mean RSU detection lead:",
        summary["mean_rsu_detection_lead_s"],
        "s",
    )
    print(
        "Mean V2I latency:",
        summary["mean_v2i_latency_ms"],
        "ms",
    )
    print(
        "Mean E2E latency:",
        summary["mean_end_to_end_latency_ms"],
        "ms",
    )
    print(
        "Mean clearance gain:",
        summary["mean_clearance_gain_m"],
        "m",
    )
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("Report:", report_path)

    if not summary["all_pairs_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
