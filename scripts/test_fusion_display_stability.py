from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion.fusion_engine import FusionEngine


def detection(confidence, source_x):
    return {
        "class_name": "pedestrian",
        "confidence": confidence,
        "world_match": {
            "actor_id": 77,
            "world_position": {
                "x": source_x,
                "y": 5.0,
                "z": 0.95,
            },
        },
    }


def packet(frame_id, source, detections):
    result = {
        "source": source,
        "frame_id": frame_id,
        "detections": detections,
    }

    if source == "rsu_01":
        result["v2i"] = {
            "source_id": "RSU-01",
            "latency_ms": 25.0,
        }

    return result


def pedestrian(world):
    return next(
        obj
        for obj in world["objects"]
        if obj["class_name"] == "pedestrian"
    )


def main():
    engine = FusionEngine(
        temporal_max_missed_frames=2,
        display_position_ema_alpha=0.45,
        display_source_hold_frames=2,
    )

    first = engine.fuse(
        [
            packet(
                10,
                "rsu_01",
                [detection(0.80, 10.0)],
            )
        ]
    )
    p1 = pedestrian(first)

    assert p1["display_sources"] == ["RSU-01"]
    assert p1["display_world_position"]["x"] == 10.0

    # RSU drops for one frame while Onboard observes the same actor.
    second = engine.fuse(
        [
            packet(
                11,
                "onboard",
                [detection(0.72, 10.4)],
            )
        ]
    )
    p2 = pedestrian(second)

    assert p2["sources"] == ["onboard"]
    assert "RSU-01" in p2["display_sources"]
    assert "onboard" in p2["display_sources"]

    # Display coordinate should move less than the raw 0.4 m jump.
    raw_x = p2["world_position"]["x"]
    display_x = p2["display_world_position"]["x"]

    assert raw_x == 10.4
    assert 10.0 < display_x < 10.4

    # Second RSU miss is still held for presentation.
    third = engine.fuse(
        [
            packet(
                12,
                "onboard",
                [detection(0.70, 10.5)],
            )
        ]
    )
    p3 = pedestrian(third)

    assert "RSU-01" in p3["display_sources"]

    # Third consecutive RSU miss removes only the display contribution.
    fourth = engine.fuse(
        [
            packet(
                13,
                "onboard",
                [detection(0.69, 10.6)],
            )
        ]
    )
    p4 = pedestrian(fourth)

    assert p4["sources"] == ["onboard"]
    assert p4["display_sources"] == ["onboard"]
    assert "RSU-01" in p4["source_history"]

    print("=== Perceptinet Fusion Display Stability Test ===")
    print("Raw source truth preserved: PASS")
    print("Contributor 2-frame hold: PASS")
    print("Source history retained: PASS")
    print("Display position EMA smoothing: PASS")
    print("Risk world position remains raw: PASS")
    print("FUSION DISPLAY STABILITY TEST PASSED")


if __name__ == "__main__":
    main()
