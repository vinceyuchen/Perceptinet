"""CARLA waypoint geometry for the read-only Perceptinet Driver Road View.

This module deliberately does not import CARLA: only a CARLA map / waypoint
interface is needed, so the sampling, coordinate math and shape can be tested
with lightweight waypoint doubles on machines without the simulator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


class RoadContextSampler:
    """Pre-index map waypoint links once, then export a local ego-relative view.

    Export format (v1): lane line segments, expressed as [lateral, forward]
    metres in the same ego frame as RiskEngine. Geometry is *visual only*.
    """

    def __init__(
        self,
        carla_map: Any,
        *,
        sample_step_m: float = 3.0,
        grid_size_m: float = 15.0,
        max_segments: int = 320,
    ) -> None:
        self.map_name = str(getattr(carla_map, "name", "unknown"))
        self.sample_step_m = max(1.0, float(sample_step_m))
        self.grid_size_m = max(5.0, float(grid_size_m))
        self.max_segments = max(1, int(max_segments))
        self._segments: List[Tuple[float, float, float, float, float, bool]] = []
        self._grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        for waypoint in carla_map.generate_waypoints(self.sample_step_m):
            if not self._drivable(waypoint):
                continue
            start = waypoint.transform.location
            x1, y1 = float(start.x), float(start.y)
            next_waypoints = waypoint.next(self.sample_step_m)
            for nxt in next_waypoints:
                if not self._drivable(nxt):
                    continue
                end = nxt.transform.location
                x2, y2 = float(end.x), float(end.y)
                distance = math.hypot(x2 - x1, y2 - y1)
                if not (0.1 < distance < self.sample_step_m * 2.25):
                    continue
                width = min(7.0, max(1.5, float(waypoint.lane_width)))
                is_junction = bool(waypoint.is_junction or nxt.is_junction)
                idx = len(self._segments)
                self._segments.append((x1, y1, x2, y2, width, is_junction))
                gx = int(math.floor(((x1+x2)*0.5) / self.grid_size_m))
                gy = int(math.floor(((y1+y2)*0.5) / self.grid_size_m))
                self._grid[(gx, gy)].append(idx)

    @staticmethod
    def _drivable(waypoint: Any) -> bool:
        lane_type = str(getattr(waypoint, "lane_type", "Driving"))
        return "Driving" in lane_type or lane_type == "1"

    @staticmethod
    def _finite(value: Any) -> Optional[float]:
        try:
            v = float(value)
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    def snapshot(
        self,
        ego_state: Dict[str, Any],
        *,
        forward_m: float = 40.0,
        behind_m: float = 8.0,
        side_m: float = 18.0,
    ) -> Optional[Dict[str, Any]]:
        pose = ego_state.get("world_position") or {}
        x = self._finite(pose.get("x"))
        y = self._finite(pose.get("y"))
        heading = self._finite(ego_state.get("yaw_deg"))
        if x is None or y is None or heading is None:
            return None
        heading_rad = math.radians(heading)
        cosine, sine = math.cos(heading_rad), math.sin(heading_rad)

        def local(wx: float, wy: float) -> Tuple[float, float]:
            dx, dy = wx-x, wy-y
            return -dx*sine+dy*cosine, dx*cosine+dy*sine

        # Index lookup uses a conservative circumscribed radius, not an
        # assumption that world XY axes equal ego local axes.
        radius = math.hypot(max(behind_m, forward_m), side_m) + 12.0
        cx = int(math.floor(x / self.grid_size_m))
        cy = int(math.floor(y / self.grid_size_m))
        grid_radius = int(math.ceil(radius / self.grid_size_m))
        candidates = set()
        for gx in range(cx-grid_radius, cx+grid_radius+1):
            for gy in range(cy-grid_radius, cy+grid_radius+1):
                candidates.update(self._grid.get((gx, gy), ()))

        selected = []
        for idx in candidates:
            x1, y1, x2, y2, width, junction = self._segments[idx]
            lat1, fwd1 = local(x1, y1)
            lat2, fwd2 = local(x2, y2)
            if (max(fwd1, fwd2) < -behind_m-5.0 or
                    min(fwd1, fwd2) > forward_m+5.0 or
                    min(lat1, lat2) > side_m+5.0 or
                    max(lat1, lat2) < -side_m-5.0):
                continue
            closeness = ((fwd1+fwd2)*0.5-12.0)**2 + ((lat1+lat2)*0.5)**2
            selected.append((closeness, idx, lat1, fwd1, lat2, fwd2,
                             width, junction))
        selected.sort(key=lambda item: (item[0], item[1]))

        segments = [
            {"points": [[round(lat1, 2), round(fwd1, 2)],
                        [round(lat2, 2), round(fwd2, 2)]],
             "width_m": round(width, 2),
             "junction": junction}
            for _, _, lat1, fwd1, lat2, fwd2, width, junction
            in selected[:self.max_segments]
        ]
        return {
            "version": 1,
            "source": "carla_waypoints",
            "coordinate_frame": "ego_relative_lateral_forward_m",
            "map_name": self.map_name,
            "view_range_m": {
                "forward": float(forward_m),
                "behind": float(behind_m),
                "side": float(side_m),
            },
            "segments": segments,
        }
