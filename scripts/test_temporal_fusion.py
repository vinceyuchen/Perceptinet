from __future__ import annotations

import sys
from pathlib import Path

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


def packet(
    frame_id,
    source,
    detections,
):
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
        (
            obj
            for obj in world["objects"]
            if obj["class_name"] == "pedestrian"
        ),
        None,
    )


def main():
    engine = FusionEngine(
        fusion_distance_m=2.0,
        temporal_max_missed_frames=2,
        confidence_ema_alpha=0.65,
    )

    frame_1 = engine.fuse(
        [
            packet(
                100,
                "rsu_01",
                [
                    detection(
                        "pedestrian",
                        0.72,
                        41,
                        -44.65,
                        25.23,
                        0.95,
                    )
                ],
            )
        ]
    )

    p1 = pedestrian(frame_1)

    assert p1 is not None
    assert p1["track_id"] == "pedestrian:actor:41"
    assert p1["coasted"] is False
    assert p1["missed_frames"] == 0
    assert p1["track_hits"] == 1

    # One-frame detector dropout must not make the fused object disappear.
    frame_2 = engine.fuse(
        [
            packet(
                101,
                "rsu_01",
                [],
            )
        ]
    )

    p2 = pedestrian(frame_2)

    assert p2 is not None
    assert p2["track_id"] == p1["track_id"]
    assert p2["coasted"] is True
    assert p2["missed_frames"] == 1
    assert p2["temporal_status"] == "COASTED"

    # Re-observation must recover the same track, not create a new identity.
    frame_3 = engine.fuse(
        [
            packet(
                102,
                "onboard",
                [
                    detection(
                        "pedestrian",
                        0.51,
                        41,
                        -44.64,
                        25.24,
                        0.95,
                    )
                ],
            ),
            packet(
                102,
                "rsu_01",
                [
                    detection(
                        "pedestrian",
                        0.80,
                        41,
                        -44.65,
                        25.23,
                        0.95,
                    )
                ],
            ),
        ]
    )

    p3 = pedestrian(frame_3)

    assert p3 is not None
    assert p3["track_id"] == p1["track_id"]
    assert p3["coasted"] is False
    assert p3["missed_frames"] == 0
    assert p3["track_hits"] == 2
    assert p3["source_count"] == 2
    assert "onboard" in p3["source_history"]
    assert "RSU-01" in p3["source_history"]

    # Two missing frames are tolerated.
    frame_4 = engine.fuse(
        [packet(103, "rsu_01", [])]
    )
    frame_5 = engine.fuse(
        [packet(104, "rsu_01", [])]
    )

    assert pedestrian(frame_4) is not None
    assert pedestrian(frame_5) is not None
    assert (
        pedestrian(frame_5)["missed_frames"]
        == 2
    )

    # The third consecutive miss expires the track.
    frame_6 = engine.fuse(
        [packet(105, "rsu_01", [])]
    )

    assert pedestrian(frame_6) is None
    assert frame_6["object_count"] == 0

    print("=== Perceptinet Temporal Fusion Test ===")
    print("Stable actor identity: PASS")
    print("One-frame dropout tolerance: PASS")
    print("Two-frame coast window: PASS")
    print("Re-observation keeps same track_id: PASS")
    print("Track expiry after tolerance: PASS")
    print("TEMPORAL FUSION TEST PASSED")


if __name__ == "__main__":
    main()
