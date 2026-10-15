from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control.braking_controller import (
    BrakingConfig,
    BrakingController,
)


def main():

    controller = BrakingController(
        BrakingConfig(
            caution_decel_mps2=0.0,
            warning_decel_mps2=0.5,
            critical_decel_mps2=6.0,
        )
    )

    dt = 0.05
    initial_speed = 4.0

    print(
        "=== Perceptinet Braking Controller v1 ==="
    )

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    normal = controller.update(
        initial_speed,
        "NORMAL",
        dt,
    )

    print(
        f'NORMAL   | '
        f'speed={normal["target_speed_mps"]:.2f}m/s | '
        f'decel={normal["deceleration_mps2"]:.2f}m/s^2'
    )

    assert normal["target_speed_mps"] == initial_speed
    assert normal["braking_active"] is False

    # --------------------------------------------------------
    # CAUTION
    # --------------------------------------------------------

    caution = controller.update(
        initial_speed,
        "CAUTION",
        dt,
    )

    print(
        f'CAUTION  | '
        f'speed={caution["target_speed_mps"]:.2f}m/s | '
        f'decel={caution["deceleration_mps2"]:.2f}m/s^2'
    )

    assert caution["target_speed_mps"] == initial_speed
    assert caution["braking_active"] is False

    # --------------------------------------------------------
    # WARNING
    # --------------------------------------------------------

    warning = controller.update(
        initial_speed,
        "WARNING",
        dt,
    )

    print(
        f'WARNING  | '
        f'speed={warning["target_speed_mps"]:.2f}m/s | '
        f'decel={warning["deceleration_mps2"]:.2f}m/s^2'
    )

    assert (
        warning["target_speed_mps"]
        < caution["target_speed_mps"]
    )
    assert warning["braking_active"] is True

    # --------------------------------------------------------
    # CRITICAL emergency braking simulation
    # --------------------------------------------------------

    speed = initial_speed
    travelled = 0.0
    elapsed = 0.0
    steps = 0

    first_command = controller.update(
        speed,
        "CRITICAL",
        dt,
    )

    expected_stopping_distance = (
        first_command[
            "estimated_stopping_distance_m"
        ]
    )

    while speed > 0.0 and steps < 200:

        command = controller.update(
            speed,
            "CRITICAL",
            dt,
        )

        next_speed = command[
            "target_speed_mps"
        ]

        # Trapezoidal integration of travelled distance.
        travelled += (
            (speed + next_speed)
            * 0.5
            * dt
        )

        speed = next_speed
        elapsed += dt
        steps += 1

    print(
        f'CRITICAL | '
        f'stopped={speed:.2f}m/s | '
        f'time={elapsed:.2f}s | '
        f'distance={travelled:.2f}m | '
        f'analytic='
        f'{expected_stopping_distance:.2f}m'
    )

    assert first_command["emergency_braking"] is True
    assert speed == 0.0

    # For 4 m/s at 6 m/s^2:
    # ideal braking distance ≈ 1.33 m.
    assert 1.2 < travelled < 1.5
    assert elapsed < 1.0

    assert abs(
        travelled
        - expected_stopping_distance
    ) < 0.1

    print("BRAKING TEST PASSED")


if __name__ == "__main__":
    main()
