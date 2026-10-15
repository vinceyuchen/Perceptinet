from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import carla

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.state_validator import DashboardStateValidator


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Live semantic consistency smoke test for the "
            "Perceptinet dashboard data stream."
        )
    )

    parser.add_argument(
        "--max-sim-seconds",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
    )

    return parser.parse_args()


def wait_for_carla(
    host: str,
    port: int,
):
    client = carla.Client(host, port)
    client.set_timeout(8.0)
    world = client.get_world()
    return client, world


def safe_load(path: Path):
    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except (
        FileNotFoundError,
        PermissionError,
        json.JSONDecodeError,
    ):
        return None


def main():
    args = parse_args()

    wait_for_carla(
        args.host,
        args.port,
    )

    suite_id = datetime.now().strftime(
        "dashboard_state_%Y%m%d_%H%M%S"
    )

    suite_root = (
        REPO_ROOT
        / "results"
        / "dashboard_state_smoke"
        / suite_id
    )
    run_id = "cooperative_live_state"

    run_dir = suite_root / run_id
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_path = (
        run_dir
        / "dashboard_snapshot.json"
    )
    stdout_path = (
        run_dir
        / "stdout.log"
    )
    stderr_path = (
        run_dir
        / "stderr.log"
    )

    env = os.environ.copy()
    env.update(
        {
            "PERCEPTINET_VALIDATION": "1",
            "PERCEPTINET_VALIDATION_MODE":
                "cooperative",
            "PERCEPTINET_VALIDATION_RUN_ID":
                run_id,
            "PERCEPTINET_VALIDATION_RESULTS_ROOT":
                str(suite_root),
            "PERCEPTINET_VALIDATION_AUTO_EXIT":
                "0",
            "PERCEPTINET_VALIDATION_HEADLESS":
                "1",
            "PERCEPTINET_VALIDATION_MAX_SIM_SECONDS":
                str(args.max_sim_seconds),
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

    validator = DashboardStateValidator(
        cooperative_expected=True
    )

    print(
        "Perceptinet Live Dashboard State Smoke Test"
    )
    print("Output:", suite_root)

    with stdout_path.open(
        "w",
        encoding="utf-8",
    ) as stdout_handle, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )

        previous = None
        seen_frames = set()
        snapshots_checked = 0
        errors: List[Dict] = []
        warnings: List[Dict] = []

        evidence = {
            "saw_healthy": False,
            "saw_rsu_primary": False,
            "saw_v2i": False,
            "saw_warning_or_critical": False,
            "saw_braking": False,
            "saw_stopped": False,
        }

        wall_deadline = (
            time.perf_counter()
            + max(
                60.0,
                args.max_sim_seconds * 10.0,
            )
        )

        while (
            process.poll() is None
            and time.perf_counter() < wall_deadline
        ):
            snapshot = safe_load(
                snapshot_path
            )

            if snapshot is None:
                time.sleep(args.poll_seconds)
                continue

            frame_id = snapshot.get("frame_id")

            if frame_id in seen_frames:
                time.sleep(args.poll_seconds)
                continue

            seen_frames.add(frame_id)

            result = validator.validate(
                snapshot,
                previous=previous,
            )

            snapshots_checked += 1

            if result["errors"]:
                errors.append(
                    {
                        "frame_id": frame_id,
                        "errors": result["errors"],
                    }
                )

            if result["warnings"]:
                warnings.append(
                    {
                        "frame_id": frame_id,
                        "warnings": result["warnings"],
                    }
                )

            health = snapshot.get(
                "system_health"
            ) or {}
            hazard = snapshot.get("hazard") or {}
            braking = snapshot.get(
                "braking"
            ) or {}
            primary = hazard.get("primary") or {}
            latency = snapshot.get(
                "latency_ms"
            ) or {}

            if (
                str(health.get("status")).upper()
                == "HEALTHY"
            ):
                evidence["saw_healthy"] = True

            if any(
                str(source).upper().startswith(
                    "RSU"
                )
                for source in (
                    primary.get("sources") or []
                )
            ):
                evidence["saw_rsu_primary"] = True

            if (
                latency.get("v2i_latency_ms")
                is not None
            ):
                evidence["saw_v2i"] = True

            if (
                str(
                    hazard.get(
                        "control_level",
                        "NORMAL",
                    )
                ).upper()
                in {"WARNING", "CRITICAL"}
            ):
                evidence[
                    "saw_warning_or_critical"
                ] = True

            if bool(braking.get("active")):
                evidence["saw_braking"] = True

            if str(
                braking.get("state", "")
            ).upper().startswith("STOPPED"):
                evidence["saw_stopped"] = True

            previous = snapshot

            time.sleep(args.poll_seconds)

        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)

            errors.append(
                {
                    "frame_id": None,
                    "errors": [
                        "pipeline exceeded wall-time limit"
                    ],
                }
            )
        else:
            process.wait(timeout=10)

    stderr_text = stderr_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    stdout_text = stdout_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    fatal_patterns = [
        "traceback (most recent call last)",
        "failed to spawn",
        "runtimeerror:",
        "permissionerror:",
    ]

    fatals = [
        pattern
        for pattern in fatal_patterns
        if pattern in (
            stdout_text + "\n" + stderr_text
        ).lower()
    ]

    required_evidence = [
        "saw_healthy",
        "saw_rsu_primary",
        "saw_v2i",
        "saw_warning_or_critical",
        "saw_braking",
        "saw_stopped",
    ]

    missing_evidence = [
        key
        for key in required_evidence
        if not evidence[key]
    ]

    passed = (
        process.returncode == 0
        and snapshots_checked >= 10
        and not errors
        and not fatals
        and not missing_evidence
    )

    summary = {
        "suite_id": suite_id,
        "returncode": process.returncode,
        "snapshots_checked":
            snapshots_checked,
        "unique_frames": len(seen_frames),
        "evidence": evidence,
        "missing_evidence":
            missing_evidence,
        "semantic_errors": errors,
        "semantic_warnings": warnings,
        "fatal_patterns": fatals,
        "passed": passed,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }

    summary_path = (
        suite_root
        / "dashboard_state_smoke_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=== DASHBOARD STATE RESULT ===")
    print(
        "Snapshots checked:",
        snapshots_checked,
    )
    print(
        "Semantic errors:",
        len(errors),
    )
    print(
        "Semantic warnings:",
        len(warnings),
    )
    print(
        "Evidence:",
        evidence,
    )
    print(
        "Missing evidence:",
        missing_evidence,
    )
    print(
        "Fatal patterns:",
        fatals,
    )
    print(
        "PASS:",
        passed,
    )
    print(
        "Summary:",
        summary_path,
    )

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
