"""Perceptinet live looping CARLA + YOLO26n + RSU + Dashboard teacher demo.

On Windows, double-click this .py file (with Python 3.12 associated).
CARLA's graphical Unreal window shows the spectator scene, the existing
braking script shows the ONBOARD + RSU-01 camera feeds with detections, and
a separate process shows the live four-card Driver/Pro Dashboard.

Every cycle runs the *real* existing perception/control pipeline. A cycle
restarts only after the controller confirms STOPPED and holds it briefly.
No recorded video, invented detections, or synthetic Dashboard values.

Exit: press Q in the camera window, Q in the launcher console, or close
the Dashboard. The launcher requests clean CARLA actor cleanup first.

Requirements: Windows desktop session, CARLA server, yolo26n model,
and the dependencies already needed by the existing project.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
SCENARIO = ROOT / "scripts" / "run_occluded_pedestrian_braking_v1.py"
DASHBOARD = ROOT / "scripts" / "run_dashboard.py"
SIGNALS = ROOT / "results" / "teacher_demo"
CYCLE_FILE = SIGNALS / "cycle_complete.json"
STOP_FILE = SIGNALS / "stop.signal"
SNAPSHOT = ROOT / "results" / "dashboard_snapshot.json"
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000


def connected(host: str = CARLA_HOST, port: int = CARLA_PORT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def executable_candidates() -> list[Path]:
    """Look for common Windows CARLA installations, without guessing success."""
    hints = [
        os.environ.get("PERCEPTINET_CARLA_EXE", ""),
        r"C:\CARLA\CarlaUE4.exe",
        r"C:\CARLA\CarlaUnreal.exe",
        r"C:\CARLA\WindowsNoEditor\CarlaUE4.exe",
        r"C:\CARLA\WindowsNoEditor\CarlaUnreal.exe",
        r"C:\CARLA\WindowsNoEditor\CarlaUE5.exe",
        r"C:\CARLA\CarlaUE5.exe",
    ]
    return [Path(x) for x in hints if x]


def find_carla_exe(requested: str | None) -> Path | None:
    candidates = [Path(requested)] if requested else executable_candidates()
    return next((p for p in candidates if p.is_file()), None)


def launch_carla_if_needed(args) -> subprocess.Popen | None:
    if connected():
        print("CARLA server detected on 127.0.0.1:2000.")
        return None
    exe = find_carla_exe(args.carla_exe)
    if not exe:
        raise RuntimeError(
            "CARLA is not running on 127.0.0.1:2000 and its Windows "
            "executable was not found. Start CARLA manually, or supply "
            "--carla-exe C:\\path\\to\\CarlaUE4.exe. "
            "You may also set PERCEPTINET_CARLA_EXE."
        )
    print(f"Starting CARLA: {exe}")
    server = subprocess.Popen(
        [str(exe), "-windowed", "-ResX=900", "-ResY=520",
         "-quality-level=Low"], cwd=str(exe.parent)
    )
    deadline = time.monotonic() + args.carla_start_timeout
    while time.monotonic() < deadline:
        if connected():
            print("CARLA port is open. Waiting briefly for map startup...")
            time.sleep(5)
            return server
        if server.poll() is not None:
            raise RuntimeError(
                f"CARLA executable exited early (code {server.returncode})."
            )
        time.sleep(1)
    raise TimeoutError(
        "CARLA did not open port 2000 within "
        f"{args.carla_start_timeout} seconds."
    )


def check_project() -> None:
    for f in (SCENARIO, DASHBOARD):
        if not f.is_file():
            raise FileNotFoundError(
                f"Missing project file: {f}. Put this launcher at the "
                "Perceptinet repository root."
            )
    modules = ("carla", "cv2", "ultralytics", "numpy")
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(
            f"Missing Python modules: {', '.join(missing)}. Use the same "
            "Python environment that runs the original CARLA demo."
        )


def keyboard_quit() -> bool:
    """Non-blocking console Q on Windows; camera Q works independently."""
    if os.name != "nt" or not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        import msvcrt
        if msvcrt.kbhit():
            key = msvcrt.getwch().lower()
            return key in ("q", "\x1b")
    except (OSError, ImportError):
        pass
    return False


def stop_scenario_gracefully(
    process: subprocess.Popen | None, timeout: float = 20.0
) -> None:
    if process is None or process.poll() is not None:
        return
    STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOP_FILE.write_text("stop", encoding="utf-8")
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("Scenario did not clean up in time; terminating it.")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def request_dashboard_close(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    # Closing a read-only dashboard cannot affect simulated CARLA control.
    if os.name == "nt":
        try:
            user32 = ctypes.windll.user32
            handle = user32.FindWindowW(None, "Perceptinet Dashboard")
            if handle:
                user32.PostMessageW(handle, 0x0010, 0, 0)  # WM_CLOSE
                process.wait(timeout=3)
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Loop the real CARLA pedestrian braking scenario "
                    "alongside onboard/RSU camera and live dashboard windows."
    )
    parser.add_argument("--carla-exe", default=None,
                        help="CARLA Windows server executable (if not running).")
    parser.add_argument("--carla-start-timeout", type=float, default=120.0)
    parser.add_argument("--hold-seconds", type=float, default=3.0,
                        help="Show confirmed STOPPED state before each reset.")
    parser.add_argument("--max-round-seconds", type=float, default=300.0,
                        help="Stop with an error if a run never stops.")
    parser.add_argument("--cycles", type=int, default=0,
                        help="0 = loop until Q, positive = number of cycles.")
    parser.add_argument("--dashboard-width", type=int, default=1152)
    parser.add_argument("--dashboard-height", type=int, default=648)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate file locations/commands without CARLA.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cycles < 0 or args.max_round_seconds < 5:
        raise ValueError("Cycles must be >= 0, max round seconds >= 5.")
    if not 1.0 <= args.hold_seconds <= 15.0:
        raise ValueError("hold-seconds must be within 1..15.")
    if not SCENARIO.is_file() or not DASHBOARD.is_file():
        raise FileNotFoundError("Launcher must be saved at repo root: " + str(ROOT))
    if args.dry_run:
        print("DRY RUN: these are the real, existing pipeline entry points.")
        print("Scenario:", SCENARIO)
        print("Dashboard:", DASHBOARD)
        print("Snapshot:", SNAPSHOT)
        print("Scenario exits for restart ONLY on a confirmed physical stop.")
        print("Requirements: CARLA server 127.0.0.1:2000, Python carla/cv2/YOLO.")
        return 0

    check_project()
    SIGNALS.mkdir(parents=True, exist_ok=True)
    STOP_FILE.unlink(missing_ok=True)
    CYCLE_FILE.unlink(missing_ok=True)

    session_dir = Path(tempfile.mkdtemp(
        prefix="session_", dir=str(SIGNALS)
    ))
    live_snapshot = session_dir / "dashboard_snapshot.json"
    print("Demo snapshot:", live_snapshot)

    server = None
    scenario = None
    dashboard = None
    rounds = 0
    try:
        server = launch_carla_if_needed(args)
        dashboard = subprocess.Popen(
            [sys.executable, str(DASHBOARD),
             "--snapshot", str(live_snapshot),
             "--width", str(args.dashboard_width),
             "--height", str(args.dashboard_height),
             "--mode", "driver", "--stale-seconds", "2.5"],
            cwd=str(ROOT)
        )
        print("\n=== PERCEPTINET TEACHER DEMO: REAL CARLA PIPELINE ===")
        print("CARLA spectator scene: Unreal CARLA window")
        print("RSU + Onboard camera feeds: Automatic Braking Demo window")
        print("Road Map / statuses: Perceptinet Dashboard window")
        print("Click Driver Mode / Pro Mode in Dashboard to switch.")
        print("Press Q in camera window or in THIS console to stop.")
        print("Closing the Dashboard also stops the launcher.\n", flush=True)

        while args.cycles == 0 or rounds < args.cycles:
            if dashboard.poll() is not None:
                print("Dashboard closed. Ending demonstration.")
                break
            STOP_FILE.unlink(missing_ok=True)
            CYCLE_FILE.unlink(missing_ok=True)
            env = os.environ.copy()
            # Demo lifecycle overrides any inherited evaluation/validation mode.
            env.pop("PERCEPTINET_VALIDATION", None)
            env.pop("PERCEPTINET_VALIDATION_MODE", None)
            env.pop("PERCEPTINET_VALIDATION_RUN_ID", None)
            env.pop("PERCEPTINET_VALIDATION_HEADLESS", None)
            env.pop("PERCEPTINET_VALIDATION_AUTO_EXIT", None)
            env["PERCEPTINET_TEACHER_DEMO"] = "1"
            # Unique per-cycle CSV and metrics, with a private live
            # snapshot shared by this launcher's Dashboard and scenario.
            env["PERCEPTINET_DEMO_CYCLE_INDEX"] = str(rounds + 1)
            env["PERCEPTINET_DEMO_SNAPSHOT_FILE"] = str(live_snapshot)
            env["PERCEPTINET_DEMO_STOP_FILE"] = str(STOP_FILE)
            env["PERCEPTINET_DEMO_CYCLE_FILE"] = str(CYCLE_FILE)
            env["PERCEPTINET_DEMO_STOP_HOLD_SECONDS"] = str(args.hold_seconds)

            print(f"\n=== REAL SCENARIO CYCLE {rounds + 1} ===", flush=True)
            scenario = subprocess.Popen(
                [sys.executable, str(SCENARIO)],
                cwd=str(ROOT), env=env
            )
            started = time.monotonic()
            interrupted = False
            timed_out = False

            while scenario.poll() is None:
                if keyboard_quit() or dashboard.poll() is not None:
                    print("Operator exit requested.")
                    interrupted = True
                    stop_scenario_gracefully(scenario)
                    break
                if time.monotonic() - started > args.max_round_seconds:
                    print("No confirmed stop in time. Demo will not "
                          "fake a successful cycle.")
                    timed_out = True
                    stop_scenario_gracefully(scenario)
                    break
                time.sleep(0.25)
            code = scenario.wait()
            scenario = None
            if interrupted:
                break
            if timed_out:
                raise TimeoutError("CARLA demo cycle exceeded time limit.")
            if code != 0:
                raise RuntimeError(f"Scenario failed (exit code {code}).")

            # The marker is created only after the existing controller's
            # stop_reported=True and the requested STOPPED hold duration.
            if not CYCLE_FILE.is_file():
                print("Scenario ended without a confirmed completed-stop "
                      "marker (for example, camera window Q). Stopping.")
                break
            report = json.loads(CYCLE_FILE.read_text(encoding="utf-8"))
            if (report.get("completed") is not True or
                    report.get("reason") != "controller_confirmed_stop"):
                raise RuntimeError("Invalid completion marker; refusing restart.")
            rounds += 1
            print(f"Confirmed stop cycle #{rounds}. CARLA actors cleaned. "
                  "Recreating scene for the next run...", flush=True)
            if args.cycles and rounds >= args.cycles:
                break
            for _ in range(6):
                if keyboard_quit() or dashboard.poll() is not None:
                    print("Operator exit requested.")
                    return 0
                time.sleep(0.25)
        print(f"Demo ended normally. Confirmed completed cycles: {rounds}")
        return 0
    except KeyboardInterrupt:
        print("\nDemo interrupted. Asking the CARLA scene to clean up...")
        return 130
    finally:
        stop_scenario_gracefully(scenario)
        request_dashboard_close(dashboard)
        STOP_FILE.unlink(missing_ok=True)
        CYCLE_FILE.unlink(missing_ok=True)
        if server is not None and server.poll() is None:
            print("CARLA graphical server was started by this launcher; "
                  "leaving it running for inspection. Close it manually.")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\nTEACHER DEMO ERROR: {exc}", file=sys.stderr)
        print("Verify CARLA is ready, the YOLO26n model exists, "
              "and the Python environment matches your tested setup.")
        if sys.stdin and sys.stdin.isatty():
            try:
                input("Press Enter to close this window...")
            except EOFError:
                pass
        sys.exit(1)
