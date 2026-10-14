from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2i.channel import V2IChannel, V2IConfig


def main():
    config = V2IConfig(
        base_latency_ms=25.0,
        jitter_ms=5.0,
        packet_loss_rate=0.0,
        random_seed=3163,
    )

    channel = V2IChannel(config)

    rsu_packet = {
        "source": "rsu_01",
        "frame_id": 18543,
        "timestamp": 100.000,
        "inference_ms": 12.4,
        "detections": [
            {
                "class_name": "pedestrian",
                "confidence": 0.74,
                "bbox": [
                    629.1,
                    324.9,
                    655.7,
                    411.2,
                ],
                "world_match": {
                    "actor_id": 36,
                    "world_position": {
                        "x": -44.65,
                        "y": 25.23,
                        "z": 0.95,
                    },
                },
            }
        ],
    }

    received = channel.transmit(
        rsu_packet,
        source_id="RSU-01",
    )

    if received is None:
        raise RuntimeError("Unexpected V2I packet drop")

    v2i = received["v2i"]

    print("=== Perceptinet V2I v1 Test ===")
    print(f"Source: {v2i['source_id']}")
    print(f"Frame: {received['frame_id']}")
    print(
        f"Capture timestamp: "
        f"{v2i['capture_timestamp_s']:.6f}s"
    )
    print(
        f"Send timestamp: "
        f"{v2i['send_timestamp_s']:.6f}s"
    )
    print(
        f"Receive timestamp: "
        f"{v2i['receive_timestamp_s']:.6f}s"
    )
    print(
        f"V2I latency: "
        f"{v2i['latency_ms']:.2f} ms"
    )
    print(
        "Pedestrian world position:",
        received["detections"][0]
        ["world_match"]["world_position"],
    )
    print("Stats:", channel.stats())

    assert 20.0 <= v2i["latency_ms"] <= 30.0
    assert v2i["source_id"] == "RSU-01"
    assert received["frame_id"] == 18543
    assert channel.total_received == 1

    print("V2I TEST PASSED")


if __name__ == "__main__":
    main()
