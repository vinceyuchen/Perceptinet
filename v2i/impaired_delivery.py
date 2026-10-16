"""Simulation-time V2I delivery for reproducible teacher-demo fault injection.

The original V2IChannel annotates a delay but returns immediately. This
adapter actually queues packets until their simulated receive timestamp,
with simulated packet loss and a deterministic detection-triggered outage.
It is for experiments, not a calibrated rain/RF propagation model.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from v2i.channel import V2IChannel, V2IConfig


class ImpairedV2IDelivery:
    """Deliver only due RSU packets and reject expired observations."""

    def __init__(
        self,
        config: Optional[V2IConfig] = None,
        *,
        max_age_s: float = 2.0,
        outage_after_detection_s: float = 3.0,
    ):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        if not math.isfinite(outage_after_detection_s) or outage_after_detection_s < 0:
            raise ValueError("outage_after_detection_s must be nonnegative")
        self.channel = V2IChannel(config or V2IConfig())
        self.max_age_s = float(max_age_s)
        self.outage_after_detection_s = float(outage_after_detection_s)
        self.outage_started_s: Optional[float] = None
        self.queue: List[Dict[str, Any]] = []
        self.outage_dropped = 0
        self.expired = 0
        self.superseded = 0
        self.delivered = 0
        self.last_latency_ms: Optional[float] = None

    @staticmethod
    def _has_pedestrian(packet: Dict[str, Any]) -> bool:
        return any(
            str(d.get("class_name", "")).lower() in ("person", "pedestrian")
            for d in packet.get("detections", [])
        )

    def offer(self, packet: Dict[str, Any], *, sim_time_s: float) -> None:
        """Queue RSU detection at the simulation time of this tick."""
        now = float(sim_time_s)
        if not math.isfinite(now):
            raise ValueError("Invalid CARLA simulation timestamp")
        if self.outage_started_s is None and self._has_pedestrian(packet):
            self.outage_started_s = now
        # Inject a bounded V2I blackout beginning at first RSU detection.
        if (self.outage_started_s is not None
                and now - self.outage_started_s < self.outage_after_detection_s):
            self.outage_dropped += 1
            return
        transmitted = self.channel.transmit(packet, source_id="RSU-01")
        if transmitted is not None:
            self.queue.append(transmitted)

    def receive_ready(self, *, sim_time_s: float) -> Optional[Dict[str, Any]]:
        """Pop most recent packet whose receive time has passed.

        Older due packets are superseded, while future ones remain queued.
        The receiver never receives a packet before simulated delivery.
        """
        now = float(sim_time_s)
        ready, future = [], []
        for packet in self.queue:
            metadata = packet.get("v2i") or {}
            if float(metadata.get("receive_timestamp_s", math.inf)) <= now:
                ready.append(packet)
            else:
                future.append(packet)
        self.queue = future
        if not ready:
            return None
        valid = [
            packet for packet in ready
            if 0 <= now - float(packet.get("timestamp", -math.inf)) <= self.max_age_s
        ]
        self.expired += len(ready) - len(valid)
        if not valid:
            return None
        newest = max(valid, key=lambda p: float(p.get("timestamp", -math.inf)))
        self.superseded += len(valid) - 1
        self.delivered += 1
        self.last_latency_ms = float(newest["v2i"]["latency_ms"])
        return newest

    def stats(self) -> Dict[str, Any]:
        return {
            "fault_injected": True,
            "total_sent": self.channel.total_sent + self.outage_dropped,
            "outage_dropped": self.outage_dropped,
            "random_dropped": self.channel.total_dropped,
            "expired": self.expired,
            "superseded": self.superseded,
            "delivered": self.delivered,
            "pending": len(self.queue),
            "last_latency_ms": self.last_latency_ms,
            "outage_triggered": self.outage_started_s is not None,
        }
