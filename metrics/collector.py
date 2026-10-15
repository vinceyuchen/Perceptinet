from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class _DetectionState:
    consecutive_frames: int = 0
    streak_start_time_s: Optional[float] = None
    streak_start_frame_id: Optional[int] = None
    first_stable_time_s: Optional[float] = None
    first_stable_frame_id: Optional[int] = None


class MetricsCollector:
    """
    Perceptinet Metrics v1.

    Collects latency, stable cooperative-detection timing, hazard/braking
    events and final stopping performance without depending on the UI.

    A detection is considered "stable" only after N consecutive frames.
    The reported stable-detection timestamp is the first frame in the
    qualifying streak, which avoids treating one-frame YOLO flicker as a
    meaningful early detection.
    """

    LATENCY_KEYS = (
        "onboard_inference_ms",
        "rsu_inference_ms",
        "v2i_latency_ms",
        "fusion_latency_ms",
        "risk_latency_ms",
        "control_latency_ms",
        "end_to_end_latency_ms",
    )

    def __init__(self, stable_detection_frames: int = 3):
        if stable_detection_frames < 1:
            raise ValueError("stable_detection_frames must be >= 1")

        self.stable_detection_frames = int(stable_detection_frames)
        self._latencies: Dict[str, List[float]] = {
            key: [] for key in self.LATENCY_KEYS
        }
        self._detections: Dict[str, _DetectionState] = {}
        self._events: List[Dict[str, Any]] = []
        self._brake_trigger: Optional[Dict[str, Any]] = None
        self._stop: Optional[Dict[str, Any]] = None

    @staticmethod
    def _canonical_source(source: Optional[str]) -> str:
        if not source:
            return "unknown"

        value = str(source).strip()

        if value.lower() in {"onboard", "onboard_camera", "ego"}:
            return "onboard"

        if value.lower().startswith("rsu"):
            return value.replace("_", "-").upper()

        return value

    @staticmethod
    def _valid_number(value: Any) -> Optional[float]:
        if value is None:
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(number):
            return None

        return number

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> Optional[float]:
        if not values:
            return None

        ordered = sorted(values)

        if len(ordered) == 1:
            return float(ordered[0])

        rank = (len(ordered) - 1) * percentile
        lower = math.floor(rank)
        upper = math.ceil(rank)

        if lower == upper:
            return float(ordered[lower])

        weight = rank - lower

        return float(
            ordered[lower] * (1.0 - weight)
            + ordered[upper] * weight
        )

    @classmethod
    def _stats(cls, values: List[float]) -> Dict[str, Any]:
        if not values:
            return {
                "count": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "min": None,
                "max": None,
            }

        return {
            "count": len(values),
            "mean": float(statistics.fmean(values)),
            "p50": cls._percentile(values, 0.50),
            "p95": cls._percentile(values, 0.95),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    def record_latency(self, metric: str, value_ms: Any) -> None:
        if metric not in self._latencies:
            raise KeyError(f"Unknown latency metric: {metric}")

        value = self._valid_number(value_ms)

        if value is not None and value >= 0.0:
            self._latencies[metric].append(value)

    def observe_detection(
        self,
        source: str,
        detected: bool,
        sim_time_s: float,
        frame_id: Optional[int] = None,
    ) -> None:
        source_id = self._canonical_source(source)
        state = self._detections.setdefault(
            source_id,
            _DetectionState(),
        )

        if detected:
            if state.consecutive_frames == 0:
                state.streak_start_time_s = float(sim_time_s)
                state.streak_start_frame_id = (
                    None if frame_id is None else int(frame_id)
                )

            state.consecutive_frames += 1

            if (
                state.first_stable_time_s is None
                and state.consecutive_frames
                >= self.stable_detection_frames
            ):
                state.first_stable_time_s = state.streak_start_time_s
                state.first_stable_frame_id = state.streak_start_frame_id

                self.record_event(
                    "stable_detection",
                    sim_time_s=float(sim_time_s),
                    frame_id=frame_id,
                    source=source_id,
                    streak_frames=state.consecutive_frames,
                    stable_since_s=state.first_stable_time_s,
                )

        else:
            state.consecutive_frames = 0
            state.streak_start_time_s = None
            state.streak_start_frame_id = None

    def record_perception_packet(
        self,
        packet: Dict[str, Any],
    ) -> None:
        source = self._canonical_source(
            packet.get("v2i", {}).get("source_id")
            or packet.get("source")
        )

        inference_ms = packet.get("inference_ms")

        if source == "onboard":
            self.record_latency(
                "onboard_inference_ms",
                inference_ms,
            )
        elif source.startswith("RSU"):
            self.record_latency(
                "rsu_inference_ms",
                inference_ms,
            )

        v2i = packet.get("v2i") or {}

        if "latency_ms" in v2i:
            self.record_latency(
                "v2i_latency_ms",
                v2i.get("latency_ms"),
            )

        detections = packet.get("detections", [])
        pedestrian_detected = any(
            det.get("class_name") == "pedestrian"
            for det in detections
        )

        self.observe_detection(
            source=source,
            detected=pedestrian_detected,
            sim_time_s=float(packet.get("timestamp", 0.0)),
            frame_id=packet.get("frame_id"),
        )

    def record_pipeline_timing(
        self,
        *,
        fusion_ms: Any = None,
        risk_ms: Any = None,
        control_ms: Any = None,
        end_to_end_ms: Any = None,
    ) -> None:
        values = {
            "fusion_latency_ms": fusion_ms,
            "risk_latency_ms": risk_ms,
            "control_latency_ms": control_ms,
            "end_to_end_latency_ms": end_to_end_ms,
        }

        for metric, value in values.items():
            self.record_latency(metric, value)

    def record_event(
        self,
        event_type: str,
        *,
        sim_time_s: float,
        frame_id: Optional[int] = None,
        **details: Any,
    ) -> None:
        event = {
            "event_type": str(event_type),
            "sim_time_s": float(sim_time_s),
            "frame_id": (
                None if frame_id is None else int(frame_id)
            ),
        }
        event.update(details)
        self._events.append(event)

    def record_hazard_transition(
        self,
        *,
        old_level: str,
        new_level: str,
        sim_time_s: float,
        frame_id: Optional[int] = None,
    ) -> None:
        self.record_event(
            "hazard_transition",
            sim_time_s=sim_time_s,
            frame_id=frame_id,
            old_level=old_level,
            new_level=new_level,
        )

    def record_brake_trigger(
        self,
        *,
        sim_time_s: float,
        frame_id: Optional[int],
        distance_m: Any,
        ttc_s: Any,
        speed_mps: Any,
        hazard_level: str,
    ) -> None:
        if self._brake_trigger is not None:
            return

        self._brake_trigger = {
            "sim_time_s": float(sim_time_s),
            "frame_id": (
                None if frame_id is None else int(frame_id)
            ),
            "distance_m": self._valid_number(distance_m),
            "ttc_s": self._valid_number(ttc_s),
            "speed_mps": self._valid_number(speed_mps),
            "hazard_level": str(hazard_level),
        }

        self.record_event(
            "brake_trigger",
            sim_time_s=sim_time_s,
            frame_id=frame_id,
            **{
                key: value
                for key, value in self._brake_trigger.items()
                if key not in {"sim_time_s", "frame_id"}
            },
        )

    def record_stop(
        self,
        *,
        sim_time_s: float,
        frame_id: Optional[int],
        actual_braking_distance_m: Any,
        final_clearance_m: Any,
        hazard_level: str,
    ) -> None:
        if self._stop is not None:
            return

        self._stop = {
            "sim_time_s": float(sim_time_s),
            "frame_id": (
                None if frame_id is None else int(frame_id)
            ),
            "actual_braking_distance_m": self._valid_number(
                actual_braking_distance_m
            ),
            "final_clearance_m": self._valid_number(
                final_clearance_m
            ),
            "hazard_level": str(hazard_level),
        }

        self.record_event(
            "vehicle_stopped",
            sim_time_s=sim_time_s,
            frame_id=frame_id,
            **{
                key: value
                for key, value in self._stop.items()
                if key not in {"sim_time_s", "frame_id"}
            },
        )

    def _detection_summary(self) -> Dict[str, Any]:
        sources = {}

        for source, state in sorted(self._detections.items()):
            sources[source] = {
                "first_stable_time_s": state.first_stable_time_s,
                "first_stable_frame_id": state.first_stable_frame_id,
            }

        onboard_time = None

        if "onboard" in self._detections:
            onboard_time = self._detections[
                "onboard"
            ].first_stable_time_s

        rsu_candidates = [
            (
                source,
                state.first_stable_time_s,
                state.first_stable_frame_id,
            )
            for source, state in self._detections.items()
            if (
                source.startswith("RSU")
                and state.first_stable_time_s is not None
            )
        ]

        earliest_rsu_source = None
        earliest_rsu_time = None
        earliest_rsu_frame = None

        if rsu_candidates:
            (
                earliest_rsu_source,
                earliest_rsu_time,
                earliest_rsu_frame,
            ) = min(rsu_candidates, key=lambda item: item[1])

        rsu_lead_s = None

        if (
            onboard_time is not None
            and earliest_rsu_time is not None
        ):
            rsu_lead_s = float(onboard_time - earliest_rsu_time)

        return {
            "stable_detection_frames": self.stable_detection_frames,
            "sources": sources,
            "earliest_rsu_source": earliest_rsu_source,
            "earliest_rsu_stable_time_s": earliest_rsu_time,
            "earliest_rsu_stable_frame_id": earliest_rsu_frame,
            "onboard_stable_time_s": onboard_time,
            "rsu_detection_lead_s": rsu_lead_s,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "latency": {
                key: self._stats(values)
                for key, values in self._latencies.items()
            },
            "detection": self._detection_summary(),
            "braking": {
                "trigger": self._brake_trigger,
                "stop": self._stop,
            },
            "event_count": len(self._events),
        }

    @property
    def events(self) -> List[Dict[str, Any]]:
        return [dict(event) for event in self._events]

    def export_summary_json(self, path: Any) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                self.summary(),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return output_path

    def export_events_csv(self, path: Any) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        keys = ["event_type", "sim_time_s", "frame_id"]
        detail_keys = sorted(
            {
                key
                for event in self._events
                for key in event.keys()
                if key not in keys
            }
        )
        fieldnames = keys + detail_keys

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for event in self._events:
                writer.writerow(event)

        return output_path
