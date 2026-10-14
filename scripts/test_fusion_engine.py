from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion.fusion_engine import FusionEngine


def detection(
    class_name,
    confidence,
    actor_id,
    x,
    y,
    z,
):
    return {
        "class_name": class_name,
        "confidence": confidence,
        "world_match": {
            "actor_id": actor_id,
            "world_position": {
                "x": x,
                "y": y,
                "z": z,
            },
        },
    }


def main():

    onboard_packet = {
        "source": "onboard",
        "detections": [
            detection(
                "car",
                0.82,
                10,
                -50.0,
                22.0,
                0.8,
            )
        ],
    }

    rsu_packet = {
        "source": "rsu_01",
        "v2i": {
            "source_id": "RSU-01",
            "latency_ms": 25.4,
        },
        "detections": [
            detection(
                "pedestrian",
                0.75,
                41,
                -44.65,
                25.23,
                0.95,
            ),
            detection(
                "car",
                0.79,
                10,
                -50.0,
                22.0,
                0.8,
            ),
        ],
    }

    engine = FusionEngine(
        fusion_distance_m=2.0
    )

    result = engine.fuse(
        [
            onboard_packet,
            rsu_packet,
        ]
    )

    print("=== Perceptinet Fusion v1 Test ===")

    for obj in result["objects"]:
        print(
            f'{obj["class_name"]} | '
            f'actor={obj["actor_id"]} | '
            f'conf={obj["confidence"]:.2f} | '
            f'sources={obj["sources"]} | '
            f'world={obj["world_position"]}'
        )

    print(
        f'Fused object count: '
        f'{result["object_count"]}'
    )

    pedestrian = next(
        obj
        for obj in result["objects"]
        if obj["class_name"] == "pedestrian"
    )

    car = next(
        obj
        for obj in result["objects"]
        if obj["class_name"] == "car"
    )

    assert pedestrian["sources"] == ["RSU-01"]
    assert pedestrian["actor_id"] == 41

    assert car["source_count"] == 2
    assert "onboard" in car["sources"]
    assert "RSU-01" in car["sources"]

    assert result["object_count"] == 2

    print("FUSION TEST PASSED")


if __name__ == "__main__":
    main()
