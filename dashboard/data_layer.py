from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


HAZARD_PRIORITY = {
    "NORMAL": 0,
    "CAUTION": 1,
    "WARNING": 2,
    "CRITICAL": 3,
}


class DashboardDataLayer:
    """
    Shared Dashboard Data Layer v1 for Perceptinet.

    Driver Mode and Pro Mode consume the same normalized snapshot.
    Presentation policy belongs to the UI layer, not here.
    """

    def __init__(self, max_timeline_events: int = 20):
        self.max_timeline_events = max(1, int(max_timeline_events))

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        if value in (None, "", "None"):
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clean_sources(sources: Any) -> List[str]:
        if sources is None:
            return []

        if isinstance(sources, str):
            value = sources.strip()

            if not value:
                return []

            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]

            parts = [
                part.strip().strip("'\"")
                for part in value.split(",")
                if part.strip()
            ]
            return parts

        if isinstance(sources, Iterable):
            return [str(source) for source in sources]

        return [str(sources)]

    @staticmethod
    def _primary_risk(
        risk_result: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not risk_result:
            return None

        risks = risk_result.get("risks") or []

        if not risks:
            return None

        return max(
            risks,
            key=lambda risk: HAZARD_PRIORITY.get(
                str(risk.get("hazard_level", "NORMAL")),
                0,
            ),
        )

    @staticmethod
    def _latency_value(
        metrics_summary: Optional[Dict[str, Any]],
        key: str,
        stat: str = "mean",
    ) -> Optional[float]:
        if not metrics_summary:
            return None

        latency = metrics_summary.get("latency") or {}
        metric = latency.get(key) or {}
        value = metric.get(stat)

        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _system_health(
        *,
        v2i_latency_ms: Optional[float],
        fused_world: Dict[str, Any],
    ) -> Dict[str, Any]:
        issues: List[str] = []

        if v2i_latency_ms is not None and v2i_latency_ms > 50.0:
            issues.append("V2I_HIGH_LATENCY")

        if "objects" not in fused_world:
            issues.append("FUSION_DATA_MISSING")

        return {
            "status": "HEALTHY" if not issues else "DEGRADED",
            "issues": issues,
        }

    @classmethod
    def significant_timeline(
        cls,
        events: Optional[Iterable[Dict[str, Any]]],
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Keep raw metrics untouched, but remove low-value CAUTION/NORMAL
        chatter from the dashboard timeline.
        """

        if not events:
            return []

        result: List[Dict[str, Any]] = []

        for event in events:
            event_type = str(event.get("event_type", ""))

            if event_type == "stable_detection":
                result.append(dict(event))
                continue

            if event_type in {
                "brake_trigger",
                "vehicle_stopped",
                "hazard_release",
            }:
                result.append(dict(event))
                continue

            if event_type == "hazard_transition":
                new_level = str(event.get("new_level", "NORMAL"))

                if new_level in {"WARNING", "CRITICAL"}:
                    result.append(dict(event))

        return result[-max(1, int(limit)):]

    def build_snapshot(
        self,
        *,
        frame_id: Optional[int],
        sim_time_s: Optional[float],
        ego_state: Dict[str, Any],
        fused_world: Dict[str, Any],
        risk_result: Optional[Dict[str, Any]],
        braking_command: Optional[Dict[str, Any]],
        braking_state: str,
        control_hazard: Optional[str] = None,
        metrics_summary: Optional[Dict[str, Any]] = None,
        metrics_events: Optional[Iterable[Dict[str, Any]]] = None,
        live_latencies: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        primary_risk = self._primary_risk(risk_result)

        observed_hazard = (
            str(risk_result.get("highest_hazard", "NORMAL"))
            if risk_result
            else "NORMAL"
        )

        control_level = str(
            control_hazard
            or (
                braking_command.get("hazard_level")
                if braking_command
                else observed_hazard
            )
        )

        live_latencies = live_latencies or {}

        v2i_latency_ms = self._number(
            live_latencies.get("v2i_latency_ms")
        )

        if v2i_latency_ms is None:
            v2i_latency_ms = self._latency_value(
                metrics_summary,
                "v2i_latency_ms",
            )

        latency = {
            "onboard_inference_ms": self._number(
                live_latencies.get("onboard_inference_ms")
            ),
            "rsu_inference_ms": self._number(
                live_latencies.get("rsu_inference_ms")
            ),
            "v2i_latency_ms": v2i_latency_ms,
            "fusion_latency_ms": self._number(
                live_latencies.get("fusion_latency_ms")
            ),
            "risk_latency_ms": self._number(
                live_latencies.get("risk_latency_ms")
            ),
            "control_latency_ms": self._number(
                live_latencies.get("control_latency_ms")
            ),
            "end_to_end_latency_ms": self._number(
                live_latencies.get("end_to_end_latency_ms")
            ),
        }

        if metrics_summary:
            for key in list(latency):
                if latency[key] is None:
                    latency[key] = self._latency_value(
                        metrics_summary,
                        key,
                    )

        primary_hazard = None

        if primary_risk is not None:
            primary_hazard = {
                "class_name": primary_risk.get("class_name"),
                "actor_id": primary_risk.get("actor_id"),
                "sources": self._clean_sources(
                    primary_risk.get("sources")
                ),
                "distance_m": self._number(
                    primary_risk.get("distance_m")
                ),
                "longitudinal_m": self._number(
                    primary_risk.get("longitudinal_m")
                ),
                "lateral_m": self._number(
                    primary_risk.get("lateral_m")
                ),
                "ttc_s": self._number(
                    primary_risk.get("ttc_s")
                ),
                "hazard_level": primary_risk.get(
                    "hazard_level",
                    "NORMAL",
                ),
                "world_position": primary_risk.get(
                    "world_position"
                ),
            }

        braking = {
            "state": str(braking_state),
            "active": bool(
                braking_command
                and braking_command.get("braking_active")
            ),
            "emergency": bool(
                braking_command
                and braking_command.get("emergency_braking")
            ),
            "deceleration_mps2": self._number(
                (
                    braking_command
                    or {}
                ).get("deceleration_mps2")
            ),
            "target_speed_mps": self._number(
                (
                    braking_command
                    or {}
                ).get("target_speed_mps")
            ),
            "estimated_stopping_distance_m": self._number(
                (
                    braking_command
                    or {}
                ).get("estimated_stopping_distance_m")
            ),
        }

        metrics = metrics_summary or {}

        return {
            "schema_version": "dashboard-v1",
            "frame_id": frame_id,
            "sim_time_s": sim_time_s,
            "system_health": self._system_health(
                v2i_latency_ms=v2i_latency_ms,
                fused_world=fused_world,
            ),
            "ego": {
                "speed_mps": self._number(
                    ego_state.get("speed_mps")
                ),
                "yaw_deg": self._number(
                    ego_state.get("yaw_deg")
                ),
                "world_position": ego_state.get(
                    "world_position"
                ),
            },
            "hazard": {
                "observed_level": observed_hazard,
                "control_level": control_level,
                "primary": primary_hazard,
            },
            "braking": braking,
            "fusion": {
                "object_count": int(
                    fused_world.get(
                        "object_count",
                        len(fused_world.get("objects", [])),
                    )
                ),
                "objects": fused_world.get("objects", []),
            },
            "latency_ms": latency,
            "metrics": {
                "detection": metrics.get("detection", {}),
                "braking": metrics.get("braking", {}),
            },
            "timeline": self.significant_timeline(
                metrics_events,
                limit=self.max_timeline_events,
            ),
        }

    @staticmethod
    def write_snapshot_json(
        snapshot: Dict[str, Any],
        path: Any,
    ) -> Path:
        """
        Atomically publish the latest live dashboard snapshot.

        The temporary-file replace prevents a UI reader from
        observing a partially written JSON document.
        """
        snapshot_path = Path(path)
        snapshot_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = snapshot_path.with_suffix(
            snapshot_path.suffix + ".tmp"
        )

        temporary_path.write_text(
            json.dumps(
                snapshot,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        last_error = None

        # On Windows, os.replace() can transiently fail with
        # WinError 5 while the dashboard reader has the current
        # snapshot open. Retry briefly instead of crashing the
        # live CARLA pipeline.
        for attempt in range(40):
            try:
                temporary_path.replace(snapshot_path)
                return snapshot_path
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.01)

        if last_error is not None:
            raise last_error

        return snapshot_path

    @staticmethod
    def load_metrics_summary(path: Any) -> Dict[str, Any]:
        metrics_path = Path(path)

        if not metrics_path.exists():
            return {}

        return json.loads(
            metrics_path.read_text(encoding="utf-8")
        )

    @staticmethod
    def load_metrics_events(path: Any) -> List[Dict[str, Any]]:
        events_path = Path(path)

        if not events_path.exists():
            return []

        with events_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as handle:
            return list(csv.DictReader(handle))

    def build_snapshot_from_files(
        self,
        *,
        summary_path: Any,
        events_path: Any,
        frame_id: Optional[int],
        sim_time_s: Optional[float],
        ego_state: Dict[str, Any],
        fused_world: Dict[str, Any],
        risk_result: Optional[Dict[str, Any]],
        braking_command: Optional[Dict[str, Any]],
        braking_state: str,
        control_hazard: Optional[str] = None,
        live_latencies: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.build_snapshot(
            frame_id=frame_id,
            sim_time_s=sim_time_s,
            ego_state=ego_state,
            fused_world=fused_world,
            risk_result=risk_result,
            braking_command=braking_command,
            braking_state=braking_state,
            control_hazard=control_hazard,
            metrics_summary=self.load_metrics_summary(
                summary_path
            ),
            metrics_events=self.load_metrics_events(
                events_path
            ),
            live_latencies=live_latencies,
        )
