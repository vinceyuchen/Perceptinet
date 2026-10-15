from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List, Optional


class FusionEngine:
    """
    Perceptinet late-fusion engine with lightweight temporal stability.

    Spatial association:
      1. matching CARLA actor_id in the simulation prototype
      2. same class + world-space distance threshold as fallback

    Temporal behavior:
      - stable track_id across frames
      - short dropout tolerance
      - confidence EMA for fused-world presentation
      - explicit observed/coasted metadata

    actor_id remains simulation-assisted and is not intended to be the final
    real-world identity mechanism.
    """

    def __init__(
        self,
        fusion_distance_m: float = 2.0,
        temporal_max_missed_frames: int = 2,
        confidence_ema_alpha: float = 0.65,
    ):
        self.fusion_distance_m = float(
            fusion_distance_m
        )
        self.temporal_max_missed_frames = max(
            0,
            int(temporal_max_missed_frames),
        )
        self.confidence_ema_alpha = min(
            1.0,
            max(
                0.0,
                float(confidence_ema_alpha),
            ),
        )

        self._tracks: Dict[str, Dict[str, Any]] = {}
        self._anonymous_track_counter = 0

    @staticmethod
    def _world_position(
        detection: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        match = detection.get("world_match")

        if not match:
            return None

        return match.get("world_position")

    @staticmethod
    def _actor_id(
        detection: Dict[str, Any]
    ) -> Optional[int]:
        match = detection.get("world_match")

        if not match:
            return None

        return match.get("actor_id")

    @staticmethod
    def _distance(
        a: Dict[str, float],
        b: Dict[str, float],
    ) -> float:
        return math.sqrt(
            (a["x"] - b["x"]) ** 2
            + (a["y"] - b["y"]) ** 2
            + (a["z"] - b["z"]) ** 2
        )

    @staticmethod
    def _object_distance(
        a: Dict[str, Any],
        b: Dict[str, Any],
    ) -> Optional[float]:
        pos_a = a.get("world_position")
        pos_b = b.get("world_position")

        if pos_a is None or pos_b is None:
            return None

        return FusionEngine._distance(
            pos_a,
            pos_b,
        )

    def reset(self) -> None:
        self._tracks.clear()
        self._anonymous_track_counter = 0

    def _same_detection(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
    ) -> bool:
        if a["class_name"] != b["class_name"]:
            return False

        actor_a = self._actor_id(a)
        actor_b = self._actor_id(b)

        if (
            actor_a is not None
            and actor_b is not None
        ):
            return actor_a == actor_b

        pos_a = self._world_position(a)
        pos_b = self._world_position(b)

        if pos_a is None or pos_b is None:
            return False

        return (
            self._distance(pos_a, pos_b)
            <= self.fusion_distance_m
        )

    def _fuse_current_frame(
        self,
        packets: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        fused_objects: List[Dict[str, Any]] = []

        for packet in packets:
            source = (
                packet.get("v2i", {}).get("source_id")
                or packet.get("source")
                or "unknown"
            )

            for detection in packet.get("detections", []):
                position = self._world_position(
                    detection
                )

                if position is None:
                    continue

                existing = None

                for obj in fused_objects:
                    if self._same_detection(
                        detection,
                        obj["_representative"],
                    ):
                        existing = obj
                        break

                confidence = float(
                    detection["confidence"]
                )

                if existing is None:
                    fused_objects.append(
                        {
                            "class_name":
                                detection["class_name"],
                            "confidence": confidence,
                            "confidence_raw": confidence,
                            "actor_id":
                                self._actor_id(detection),
                            "world_position":
                                deepcopy(position),
                            "sources": [source],
                            "source_count": 1,
                            "_representative":
                                detection,
                        }
                    )
                    continue

                if source not in existing["sources"]:
                    existing["sources"].append(source)

                existing["source_count"] = len(
                    existing["sources"]
                )

                existing["confidence"] = max(
                    existing["confidence"],
                    confidence,
                )
                existing["confidence_raw"] = max(
                    existing["confidence_raw"],
                    confidence,
                )

        for obj in fused_objects:
            obj.pop("_representative", None)

        return fused_objects

    def _actor_track_id(
        self,
        obj: Dict[str, Any],
    ) -> Optional[str]:
        actor_id = obj.get("actor_id")

        if actor_id is None:
            return None

        return (
            f'{obj["class_name"]}:actor:{int(actor_id)}'
        )

    def _find_track_for_object(
        self,
        obj: Dict[str, Any],
        claimed_track_ids: set[str],
    ) -> Optional[str]:
        actor_track_id = self._actor_track_id(obj)

        if actor_track_id is not None:
            if (
                actor_track_id in self._tracks
                and actor_track_id not in claimed_track_ids
            ):
                return actor_track_id

            # In the CARLA prototype, a real actor_id is authoritative.
            # Never merge two different actor IDs merely because they are
            # spatially close.
            return None

        best_track_id = None
        best_distance = None

        for track_id, track in self._tracks.items():
            if track_id in claimed_track_ids:
                continue

            if (
                track["class_name"]
                != obj["class_name"]
            ):
                continue

            distance = self._object_distance(
                obj,
                track,
            )

            if distance is None:
                continue

            if distance > self.fusion_distance_m:
                continue

            if (
                best_distance is None
                or distance < best_distance
            ):
                best_distance = distance
                best_track_id = track_id

        return best_track_id

    def _new_track_id(
        self,
        obj: Dict[str, Any],
    ) -> str:
        actor_track_id = self._actor_track_id(obj)

        if actor_track_id is not None:
            return actor_track_id

        self._anonymous_track_counter += 1

        return (
            f'{obj["class_name"]}:track:'
            f'{self._anonymous_track_counter:04d}'
        )

    @staticmethod
    def _frame_id(
        packets: List[Dict[str, Any]],
    ) -> Optional[int]:
        frame_ids = []

        for packet in packets:
            value = packet.get("frame_id")

            if value is None:
                continue

            try:
                frame_ids.append(int(value))
            except (TypeError, ValueError):
                pass

        return max(frame_ids) if frame_ids else None

    def _create_track(
        self,
        track_id: str,
        obj: Dict[str, Any],
        frame_id: Optional[int],
    ) -> Dict[str, Any]:
        track = deepcopy(obj)

        track.update(
            {
                "track_id": track_id,
                "track_age_frames": 1,
                "track_hits": 1,
                "missed_frames": 0,
                "coasted": False,
                "temporal_status": "OBSERVED",
                "first_seen_frame_id": frame_id,
                "last_seen_frame_id": frame_id,
                "source_history": list(
                    obj.get("sources", [])
                ),
            }
        )

        return track

    def _update_track(
        self,
        track: Dict[str, Any],
        obj: Dict[str, Any],
        frame_id: Optional[int],
    ) -> None:
        alpha = self.confidence_ema_alpha

        previous_confidence = float(
            track.get(
                "confidence",
                obj["confidence"],
            )
        )
        current_confidence = float(
            obj["confidence"]
        )

        track["confidence"] = (
            alpha * current_confidence
            + (1.0 - alpha)
            * previous_confidence
        )
        track["confidence_raw"] = float(
            obj.get(
                "confidence_raw",
                current_confidence,
            )
        )

        track["actor_id"] = obj.get("actor_id")
        track["world_position"] = deepcopy(
            obj["world_position"]
        )
        track["sources"] = list(
            obj.get("sources", [])
        )
        track["source_count"] = len(
            track["sources"]
        )

        history = list(
            track.get("source_history", [])
        )

        for source in track["sources"]:
            if source not in history:
                history.append(source)

        track["source_history"] = history
        track["track_age_frames"] = int(
            track.get("track_age_frames", 0)
        ) + 1
        track["track_hits"] = int(
            track.get("track_hits", 0)
        ) + 1
        track["missed_frames"] = 0
        track["coasted"] = False
        track["temporal_status"] = "OBSERVED"
        track["last_seen_frame_id"] = frame_id

    def _coast_track(
        self,
        track: Dict[str, Any],
    ) -> None:
        track["track_age_frames"] = int(
            track.get("track_age_frames", 0)
        ) + 1
        track["missed_frames"] = int(
            track.get("missed_frames", 0)
        ) + 1
        track["coasted"] = True
        track["temporal_status"] = "COASTED"

    def _apply_temporal_tracking(
        self,
        current_objects: List[Dict[str, Any]],
        frame_id: Optional[int],
    ) -> List[Dict[str, Any]]:
        claimed_track_ids: set[str] = set()

        for obj in current_objects:
            track_id = self._find_track_for_object(
                obj,
                claimed_track_ids,
            )

            if track_id is None:
                track_id = self._new_track_id(obj)

                self._tracks[track_id] = (
                    self._create_track(
                        track_id,
                        obj,
                        frame_id,
                    )
                )
            else:
                self._update_track(
                    self._tracks[track_id],
                    obj,
                    frame_id,
                )

            claimed_track_ids.add(track_id)

        expired = []

        for track_id, track in self._tracks.items():
            if track_id in claimed_track_ids:
                continue

            self._coast_track(track)

            if (
                track["missed_frames"]
                > self.temporal_max_missed_frames
            ):
                expired.append(track_id)

        for track_id in expired:
            del self._tracks[track_id]

        return [
            deepcopy(track)
            for track in self._tracks.values()
        ]

    def fuse(
        self,
        packets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        current_objects = self._fuse_current_frame(
            packets
        )

        frame_id = self._frame_id(packets)

        stable_objects = (
            self._apply_temporal_tracking(
                current_objects,
                frame_id,
            )
        )

        stable_objects.sort(
            key=lambda obj: str(
                obj.get("track_id", "")
            )
        )

        return {
            "object_count": len(stable_objects),
            "raw_object_count": len(current_objects),
            "objects": stable_objects,
            "temporal_max_missed_frames":
                self.temporal_max_missed_frames,
        }
