from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List, Optional


class FusionEngine:
    """
    Perceptinet late-fusion engine v1.

    Primary association in CARLA prototype:
        matching CARLA actor_id

    Fallback association:
        same class + world-space distance threshold

    actor_id is simulation-assisted and is not intended to be
    the final real-world identity mechanism.
    """

    def __init__(self, fusion_distance_m: float = 2.0):
        self.fusion_distance_m = fusion_distance_m

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

    def _same_object(
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
            and actor_a == actor_b
        ):
            return True

        pos_a = self._world_position(a)
        pos_b = self._world_position(b)

        if pos_a is None or pos_b is None:
            return False

        return (
            self._distance(pos_a, pos_b)
            <= self.fusion_distance_m
        )

    def fuse(
        self,
        packets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        fused_objects: List[Dict[str, Any]] = []

        for packet in packets:
            source = (
                packet.get("v2i", {}).get("source_id")
                or packet.get("source")
                or "unknown"
            )

            for detection in packet.get("detections", []):

                position = self._world_position(detection)

                # Fusion v1 works in common world coordinates.
                if position is None:
                    continue

                existing = None

                for obj in fused_objects:
                    if self._same_object(
                        detection,
                        obj["_representative"],
                    ):
                        existing = obj
                        break

                if existing is None:
                    fused_objects.append(
                        {
                            "class_name": detection["class_name"],
                            "confidence": float(
                                detection["confidence"]
                            ),
                            "actor_id": self._actor_id(detection),
                            "world_position": deepcopy(position),
                            "sources": [source],
                            "source_count": 1,
                            "_representative": detection,
                        }
                    )

                else:
                    if source not in existing["sources"]:
                        existing["sources"].append(source)

                    existing["source_count"] = len(
                        existing["sources"]
                    )

                    # Keep strongest confidence for Fusion v1.
                    existing["confidence"] = max(
                        existing["confidence"],
                        float(detection["confidence"]),
                    )

        for obj in fused_objects:
            obj.pop("_representative", None)

        return {
            "object_count": len(fused_objects),
            "objects": fused_objects,
        }
