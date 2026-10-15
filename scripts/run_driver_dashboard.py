import argparse
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.driver_mode import DriverModeRenderer


DEFAULT_SNAPSHOT = (
    REPO_ROOT / "results" / "dashboard_snapshot.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perceptinet Driver Mode v1"
    )

    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Path to live dashboard_snapshot.json",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
    )

    parser.add_argument(
        "--stale-seconds",
        type=float,
        default=1.0,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    renderer = DriverModeRenderer(
        width=args.width,
        height=args.height,
    )

    window_name = "Perceptinet - Driver Mode v1"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        args.width,
        args.height,
    )

    print("=== Perceptinet Driver Mode v1 ===")
    print(f"Snapshot: {args.snapshot}")
    print("Press Q or Esc to close.")

    last_snapshot = None
    last_frame_id = None
    last_update_wall_time = 0.0

    try:
        while True:
            try:
                snapshot = renderer.read_snapshot(
                    args.snapshot
                )

                frame_id = snapshot.get("frame_id")

                if frame_id != last_frame_id:
                    last_frame_id = frame_id
                    last_update_wall_time = time.monotonic()

                last_snapshot = snapshot

            except (
                FileNotFoundError,
                PermissionError,
                ValueError,
            ):
                snapshot = last_snapshot

            if snapshot is None:
                time.sleep(0.05)
                continue

            stale = (
                time.monotonic() - last_update_wall_time
                > args.stale_seconds
            )

            frame = renderer.render(
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

            if cv2.getWindowProperty(
                window_name,
                cv2.WND_PROP_VISIBLE,
            ) < 1:
                break

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
