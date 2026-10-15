"""CARLA-free smoke checks for simulation.road_context."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.road_context import RoadContextSampler


class Waypoint:
    def __init__(self, x, y, junction=False):
        self.transform = SimpleNamespace(
            location=SimpleNamespace(x=x, y=y)
        )
        self.lane_width = 3.5
        self.lane_type = "Driving"
        self.is_junction = junction
        self.followers = []

    def next(self, distance):
        return self.followers


class MockMap:
    name = "TownTest"

    def __init__(self, points):
        self.points = points
        self.calls = 0

    def generate_waypoints(self, step):
        self.calls += 1
        return self.points


def main():
    # Straight section with a perpendicular junction branch.
    a, b = Waypoint(0, 0), Waypoint(3, 0, junction=True)
    c, d = Waypoint(3, -3, junction=True), Waypoint(3, 0, junction=True)
    a.followers = [b]
    c.followers = [d]
    world_map = MockMap([a, c])
    sampler = RoadContextSampler(world_map, sample_step_m=3.0)

    east = sampler.snapshot({
        "world_position": {"x": 0, "y": 0}, "yaw_deg": 0
    })
    assert east["source"] == "carla_waypoints"
    assert east["coordinate_frame"] == "ego_relative_lateral_forward_m"
    assert len(east["segments"]) == 2
    assert any(s["junction"] for s in east["segments"])
    assert any(
        s["points"] == [[0.0, 0.0], [0.0, 3.0]]
        for s in east["segments"]
    )

    north = sampler.snapshot({
        "world_position": {"x": 0, "y": 0}, "yaw_deg": 90
    })
    assert any(
        s["points"] == [[0.0, 0.0], [-3.0, 0.0]]
        for s in north["segments"]
    )
    assert sampler.snapshot({
        "world_position": {"x": 0}, "yaw_deg": 0
    }) is None
    assert world_map.calls == 1, "waypoint extraction must be cached"
    print("ROAD CONTEXT TEST PASSED")


if __name__ == "__main__":
    main()
