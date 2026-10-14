from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class V2IConfig:
    """
    Perceptinet V2I v1 configuration.

    V1 simulates communication latency and packet loss.
    It does not yet model a real wireless protocol stack.
    """

    base_latency_ms: float = 25.0
    jitter_ms: float = 5.0
    packet_loss_rate: float = 0.0
    random_seed: int = 3163


class V2IChannel:
    """
    Simulated V2I communication channel.

    Input:
        RSU detection packet

    Output:
        Packet received by the Ego/Fusion side with V2I metadata.

    Timestamp convention:
        capture_timestamp_s:
            original perception timestamp

        send_timestamp_s:
            when the RSU packet is considered transmitted

        receive_timestamp_s:
            send_timestamp + simulated V2I latency
    """

    def __init__(self, config: Optional[V2IConfig] = None):
        self.config = config or V2IConfig()
        self._rng = random.Random(self.config.random_seed)

        self.total_sent = 0
        self.total_received = 0
        self.total_dropped = 0

    def _sample_latency_ms(self) -> float:
        jitter = self._rng.uniform(
            -self.config.jitter_ms,
            self.config.jitter_ms,
        )

        return max(
            0.0,
            self.config.base_latency_ms + jitter,
        )

    def transmit(
        self,
        packet: Dict[str, Any],
        source_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        self.total_sent += 1

        if self._rng.random() < self.config.packet_loss_rate:
            self.total_dropped += 1
            return None

        transmitted = copy.deepcopy(packet)

        packet_source = (
            source_id
            or transmitted.get("source")
            or "unknown"
        )

        capture_timestamp = float(
            transmitted.get("timestamp", 0.0)
        )

        send_timestamp = capture_timestamp
        latency_ms = self._sample_latency_ms()

        receive_timestamp = (
            send_timestamp
            + latency_ms / 1000.0
        )

        transmitted["v2i"] = {
            "source_id": packet_source,
            "capture_timestamp_s": capture_timestamp,
            "send_timestamp_s": send_timestamp,
            "receive_timestamp_s": receive_timestamp,
            "latency_ms": latency_ms,
            "dropped": False,
        }

        self.total_received += 1

        return transmitted

    def stats(self) -> Dict[str, Any]:
        return {
            "total_sent": self.total_sent,
            "total_received": self.total_received,
            "total_dropped": self.total_dropped,
        }
