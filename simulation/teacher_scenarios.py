"""Teacher demo scenario presets. The rain-associated V2I fault is injected.

CARLA WeatherParameters change scene visuals; they do not generate a
wireless channel impairment. Results must be described as an injected,
reproducible fault, not a measured meteorological propagation effect.
"""

from __future__ import annotations

from dataclasses import dataclass


CLEAR = "clear"
RAIN_V2I = "rain_v2i"


@dataclass(frozen=True)
class TeacherScenario:
    key: str
    label: str
    precipitation: float
    cloudiness: float
    wetness: float
    fog_density: float
    deposits: float
    wind_intensity: float
    sun_altitude: float
    base_latency_ms: float
    jitter_ms: float
    packet_loss_rate: float
    blackout_s: float


PRESETS = {
    CLEAR: TeacherScenario(
        CLEAR, "CLEAR WEATHER / COOPERATIVE BASELINE",
        0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 75.0,
        25.0, 5.0, 0.0, 0.0,
    ),
    RAIN_V2I: TeacherScenario(
        RAIN_V2I, "HEAVY RAIN / INJECTED V2I DEGRADATION",
        90.0, 95.0, 100.0, 20.0, 85.0, 60.0, 25.0,
        1200.0, 200.0, 0.30, 3.0,
    ),
}


def get_teacher_scenario(name: str) -> TeacherScenario:
    key = str(name).strip().lower()
    if key not in PRESETS:
        raise ValueError(f"Unknown teacher scenario: {name!r}")
    return PRESETS[key]


def scenario_for_cycle(choice: str, cycle_index: int) -> str:
    if cycle_index < 1:
        raise ValueError("cycle_index must be >= 1")
    choice = str(choice).strip().lower()
    if choice == "alternate":
        return CLEAR if cycle_index % 2 else RAIN_V2I
    if choice in PRESETS:
        return choice
    raise ValueError("Expected clear, rain_v2i, or alternate")
