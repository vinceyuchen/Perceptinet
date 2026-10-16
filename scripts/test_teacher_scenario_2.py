"""CARLA-free Scenario 2 tests: actual delayed delivery and honest labels."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2i.channel import V2IConfig
from v2i.impaired_delivery import ImpairedV2IDelivery
from simulation.teacher_scenarios import (
    CLEAR, RAIN_V2I, get_teacher_scenario, scenario_for_cycle,
)
from dashboard.presentation_state import derive_presentation_state
from dashboard.driver_mode import DriverModeRenderer, ALERT_YELLOW


def packet(t: float, pedestrian: bool = True):
    return {
        "source": "rsu_01",
        "frame_id": int(round(t * 20)),
        "timestamp": t,
        "detections": (
            [{"class_name": "pedestrian", "confidence": 0.95}]
            if pedestrian else []
        ),
    }


def main() -> None:
    assert [scenario_for_cycle("alternate", i) for i in range(1, 5)] == [
        CLEAR, RAIN_V2I, CLEAR, RAIN_V2I
    ]
    assert scenario_for_cycle("clear", 2) == CLEAR
    assert scenario_for_cycle("rain_v2i", 1) == RAIN_V2I
    rain = get_teacher_scenario(RAIN_V2I)
    assert rain.precipitation > 80
    assert rain.base_latency_ms >= 1000
    assert 0 < rain.packet_loss_rate < 1
    for bad in ("wrong", ""):
        try:
            get_teacher_scenario(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid preset accepted")

    queue = ImpairedV2IDelivery(
        V2IConfig(base_latency_ms=500, jitter_ms=0,
                  packet_loss_rate=0, random_seed=3163),
        outage_after_detection_s=1.0, max_age_s=2.0
    )
    queue.offer(packet(10.0), sim_time_s=10.0)
    queue.offer(packet(10.5), sim_time_s=10.5)
    assert queue.stats()["outage_dropped"] == 2
    assert queue.receive_ready(sim_time_s=10.9) is None
    queue.offer(packet(11.1), sim_time_s=11.1)
    assert queue.receive_ready(sim_time_s=11.5) is None
    delivered = queue.receive_ready(sim_time_s=11.6)
    assert delivered and delivered["frame_id"] == packet(11.1)["frame_id"]
    assert delivered["v2i"]["latency_ms"] == 500
    assert queue.stats()["delivered"] == 1

    # Packet transport genuinely stalls; it is not metadata-only latency.
    delayed = ImpairedV2IDelivery(
        V2IConfig(base_latency_ms=1200, jitter_ms=0,
                  packet_loss_rate=0, random_seed=3163),
        outage_after_detection_s=0.0, max_age_s=2.0,
    )
    delayed.offer(packet(20), sim_time_s=20.0)
    assert delayed.receive_ready(sim_time_s=21.19) is None
    assert delayed.receive_ready(sim_time_s=21.2)["frame_id"] == 400

    expired = ImpairedV2IDelivery(
        V2IConfig(base_latency_ms=1500, jitter_ms=0,
                  packet_loss_rate=0, random_seed=3163),
        max_age_s=0.5, outage_after_detection_s=0,
    )
    expired.offer(packet(22), sim_time_s=22)
    assert expired.receive_ready(sim_time_s=23.5) is None
    assert expired.stats()["expired"] == 1

    lost = ImpairedV2IDelivery(
        V2IConfig(base_latency_ms=25, jitter_ms=0,
                  packet_loss_rate=1, random_seed=3163),
        outage_after_detection_s=0,
    )
    lost.offer(packet(25), sim_time_s=25.0)
    assert lost.receive_ready(sim_time_s=30) is None
    assert lost.stats()["random_dropped"] == 1

    snapshot = {
        "ego": {"speed_mps": 4.0},
        "braking": {"target_speed_mps": 4.0, "active": False},
        "hazard": {"control_level": "NORMAL", "primary": None},
        "system_health": {"status": "DATA LIMITED"},
        "experiment": {"scenario": RAIN_V2I, "fault_injected": True},
    }
    state = derive_presentation_state(snapshot)
    assert state["data_status"] == "V2I DEGRADED"
    assert state["perception_text"] == "COOPERATIVE LIMITED"
    assert state["display_health"] == "DATA LIMITED"
    renderer = DriverModeRenderer()
    assert renderer._unified_status_color(snapshot, stale=False) == ALERT_YELLOW
    assert renderer._unified_status_color(snapshot, stale=True) != ALERT_YELLOW

    print("HEAVY RAIN / SIMULATED V2I DELIVERY TEST PASSED")


if __name__ == "__main__":
    main()
