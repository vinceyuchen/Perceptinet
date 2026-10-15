from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import carla


REPO_ROOT = Path(__file__).resolve().parents[1]

TEST_ACTOR_PREFIXES = (
    "vehicle.",
    "walker.pedestrian.",
    "sensor.camera.",
    "controller.ai.walker",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Repeated CARLA startup/spawn/dashboard-data/cleanup smoke test "
            "for Perceptinet demo reliability."
        )
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-sim-seconds",
        type=float,
        default=3.0,
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


def connect(host: str, port: int):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    return client, world


def live_test_actors(world):
    return [
        actor
        for actor in world.get_actors()
        if actor.is_alive
        and actor.type_id.startswith(
            TEST_ACTOR_PREFIXES
        )
    ]


def verify_carla_clean(world):
    actors = live_test_actors(world)

    if actors:
        detail = ", ".join(
            f"{actor.id}:{actor.type_id}"
            for actor in actors[:8]
        )
        raise RuntimeError(
            "Stale CARLA actors remain after run: "
            + detail
        )

    settings = world.get_settings()

    if settings.synchronous_mode:
        raise RuntimeError(
            "CARLA remained in synchronous mode "
            "after child process cleanup"
        )


def run_one(
    *,
    run_number: int,
    mode: str,
    suite_root: Path,
    max_sim_seconds: float,
):
    run_id = f"smoke_{run_number:02d}_{mode}"
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
            "PERCEPTINET_VALIDATION_AUTO_EXIT": "0",
            "PERCEPTINET_VALIDATION_HEADLESS": "1",
            "PERCEPTINET_VALIDATION_MAX_SIM_SECONDS":
                str(max_sim_seconds),
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

    started = time.perf_counter()

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

    elapsed = time.perf_counter() - started

    stdout_text = stdout_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    stderr_text = stderr_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    required_markers = [
        "CARLA spawn state settled",
        "Ego Vehicle spawned:",
        "Occluding Vehicle spawned:",
        "Pedestrian spawned:",
        "Scenario settled |",
        "Test cleaned up.",
    ]

    missing = [
        marker
        for marker in required_markers
        if marker not in stdout_text
    ]

    dashboard_path = (
        run_dir / "dashboard_snapshot.json"
    )
    summary_path = (
        run_dir / "validation_summary.json"
    )

    dashboard_ok = False

    if dashboard_path.exists():
        try:
            snapshot = json.loads(
                dashboard_path.read_text(
                    encoding="utf-8"
                )
            )
            dashboard_ok = (
                isinstance(snapshot, dict)
                and snapshot.get("frame_id") is not None
            )
        except Exception:
            dashboard_ok = False

    summary_ok = False

    if summary_path.exists():
        try:
            summary = json.loads(
                summary_path.read_text(
                    encoding="utf-8"
                )
            )
            summary_ok = (
                isinstance(summary, dict)
                and summary.get("run_id") == run_id
            )
        except Exception:
            summary_ok = False

    fatal_text = (
        stdout_text + "\n" + stderr_text
    ).lower()

    fatal_patterns = [
        "failed to spawn",
        "access is denied",
        "traceback (most recent call last)",
        "runtimeerror:",
    ]

    fatals = [
        pattern
        for pattern in fatal_patterns
        if pattern in fatal_text
    ]

    return {
        "run_id": run_id,
        "mode": mode,
        "returncode": result.returncode,
        "elapsed_wall_s": elapsed,
        "missing_markers": missing,
        "dashboard_ok": dashboard_ok,
        "summary_ok": summary_ok,
        "fatal_patterns": fatals,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def main():
    args = parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    client, world = connect(
        args.host,
        args.port,
    )

    suite_id = datetime.now().strftime(
        "reliability_smoke_%Y%m%d_%H%M%S"
    )

    suite_root = (
        REPO_ROOT
        / "results"
        / "reliability_smoke"
        / suite_id
    )
    suite_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Perceptinet Demo Reliability Smoke Test")
    print("Runs:", args.runs)
    print("Output:", suite_root)
    print(
        "Modes alternate cooperative / baseline "
        "to exercise both data paths."
    )

    results = []

    for index in range(1, args.runs + 1):
        mode = (
            "cooperative"
            if index % 2 == 1
            else "baseline"
        )

        print()
        print(
            f"=== RUN {index}/{args.runs} | "
            f"{mode.upper()} ==="
        )

        result = run_one(
            run_number=index,
            mode=mode,
            suite_root=suite_root,
            max_sim_seconds=args.max_sim_seconds,
        )

        # Re-read the current world after the child has restored settings.
        client, world = connect(
            args.host,
            args.port,
        )

        carla_clean = True
        cleanup_error = None

        try:
            verify_carla_clean(world)
        except Exception as exc:
            carla_clean = False
            cleanup_error = str(exc)

        result["carla_clean"] = carla_clean
        result["cleanup_error"] = cleanup_error

        passed = (
            result["returncode"] == 0
            and not result["missing_markers"]
            and result["dashboard_ok"]
            and result["summary_ok"]
            and not result["fatal_patterns"]
            and result["carla_clean"]
        )

        result["passed"] = passed
        results.append(result)

        print(
            f"return={result['returncode']} | "
            f"dashboard={result['dashboard_ok']} | "
            f"summary={result['summary_ok']} | "
            f"cleanup={result['carla_clean']} | "
            f"wall={result['elapsed_wall_s']:.1f}s | "
            f"PASS={passed}"
        )

        if result["missing_markers"]:
            print(
                "Missing markers:",
                result["missing_markers"],
            )

        if result["fatal_patterns"]:
            print(
                "Fatal patterns:",
                result["fatal_patterns"],
            )

        if cleanup_error:
            print(
                "Cleanup error:",
                cleanup_error,
            )

        if not passed:
            print(
                "Stopping smoke test at first failure "
                "to preserve diagnostics."
            )
            break

        time.sleep(0.75)

    passed_count = sum(
        1 for result in results
        if result.get("passed")
    )

    summary = {
        "suite_id": suite_id,
        "requested_runs": args.runs,
        "completed_runs": len(results),
        "passed_runs": passed_count,
        "all_passed": (
            len(results) == args.runs
            and passed_count == args.runs
        ),
        "runs": results,
    }

    summary_path = (
        suite_root
        / "reliability_smoke_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 58)
    print(
        f"RELIABILITY RESULT: "
        f"{passed_count}/{args.runs} PASSED"
    )
    print("Summary:", summary_path)
    print("=" * 58)

    if not summary["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
