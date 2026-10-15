import argparse
import ctypes
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.app import PerceptinetDashboardApp


DEFAULT_SNAPSHOT = (
    REPO_ROOT / "results" / "dashboard_snapshot.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perceptinet Unified Dashboard"
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1600,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=952,
    )
    parser.add_argument(
        "--stale-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--mode",
        choices=["driver", "pro"],
        default="driver",
    )
    return parser.parse_args()


def enable_high_dpi():
    if sys.platform != "win32":
        return

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main():
    args = parse_args()
    enable_high_dpi()

    app = PerceptinetDashboardApp(
        width=args.width,
        height=args.height,
        default_mode=args.mode,
    )

    window_name = "Perceptinet Dashboard"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )
    cv2.resizeWindow(
        window_name,
        args.width,
        args.height,
    )

    latest_snapshot = {"value": None}

    def on_mouse(event, x, y, flags, param):
        del flags, param

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        changed = app.handle_click(x, y)

        if changed is not None:
            print(f"Dashboard mode: {changed}")

    cv2.setMouseCallback(
        window_name,
        on_mouse,
    )

    print("=== Perceptinet Unified Dashboard ===")
    print(f"Snapshot: {args.snapshot}")
    print("Click DRIVER MODE / PRO MODE to switch.")
    print("Keyboard: D=Driver, P=Pro, TAB=Pro source filter, Q/Esc=quit")

    last_frame_id = None
    last_update_wall_time = 0.0

    try:
        while True:
            try:
                snapshot = app.read_snapshot(
                    args.snapshot
                )

                frame_id = snapshot.get("frame_id")

                if frame_id != last_frame_id:
                    last_frame_id = frame_id
                    last_update_wall_time = time.monotonic()

                latest_snapshot["value"] = snapshot

            except (
                FileNotFoundError,
                PermissionError,
                ValueError,
            ):
                snapshot = latest_snapshot["value"]

            if snapshot is None:
                time.sleep(0.05)
                continue

            stale = (
                time.monotonic()
                - last_update_wall_time
                > args.stale_seconds
            )

            frame = app.render(
                snapshot,
                stale=stale,
            )

            cv2.imshow(
                window_name,
                frame,
            )

            key = cv2.waitKey(33) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (ord("d"), ord("D")):
                app.set_mode(
                    PerceptinetDashboardApp.DRIVER_MODE
                )
                print("Dashboard mode: DRIVER")

            if key in (ord("p"), ord("P")):
                app.set_mode(
                    PerceptinetDashboardApp.PRO_MODE
                )
                print("Dashboard mode: PRO")

            if (
                key == 9
                and app.mode
                == PerceptinetDashboardApp.PRO_MODE
            ):
                source = app.pro_renderer.cycle_source(
                    snapshot
                )
                print(f"Pro source filter: {source}")

            if cv2.getWindowProperty(
                window_name,
                cv2.WND_PROP_VISIBLE,
            ) < 1:
                break

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
