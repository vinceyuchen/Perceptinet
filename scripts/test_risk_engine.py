from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from risk.risk_engine import RiskEngine


def main():

    # Synthetic Ego state:
    # moving along +X toward the occluded pedestrian.
    ego_state = {
        "world_position": {
            "x": -60.0,
            "y": 25.23,
            "z": 0.95,
        },
        "yaw_deg": 0.0,
        "speed_mps": 8.0,
    }

    fused_world = {
        "object_count": 1,
        "objects": [
            {
                "class_name": "pedestrian",
                "actor_id": 46,
                "confidence": 0.73,
                "sources": ["RSU-01"],
                "world_position": {
                    "x": -44.65,
                    "y": 25.23,
                    "z": 0.95,
                },
            }
        ],
    }

    engine = RiskEngine()

    result = engine.assess_world(
        ego_state,
        fused_world,
    )

    risk = result["risks"][0]

    print("=== Perceptinet Risk Assessment v1 ===")
    print(
        f'Pedestrian actor={risk["actor_id"]}'
    )
    print(
        f'Sources: {risk["sources"]}'
    )
    print(
        f'Distance: {risk["distance_m"]:.2f} m'
    )
    print(
        f'Longitudinal: '
        f'{risk["longitudinal_m"]:.2f} m'
    )
    print(
        f'Lateral: {risk["lateral_m"]:.2f} m'
    )
    print(
        f'Closing speed: '
        f'{risk["closing_speed_mps"]:.2f} m/s'
    )
    print(
        f'TTC: {risk["ttc_s"]:.2f} s'
    )
    print(
        f'Hazard: {risk["hazard_level"]}'
    )
    print(
        f'Highest hazard: '
        f'{result["highest_hazard"]}'
    )

    assert risk["in_path"] is True

    assert (
        1.8
        < risk["ttc_s"]
        < 2.1
    )

    assert risk["hazard_level"] == "WARNING"

    assert result["highest_hazard"] == "WARNING"

    assert risk["sources"] == ["RSU-01"]

    print("RISK TEST PASSED")


if __name__ == "__main__":
    main()
