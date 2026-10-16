# Perceptinet Teacher Demo: Comparative Scenario 1 & Scenario 2

## Launch on the Windows CARLA machine

1. Update the repository from GitHub `main`.
2. Run `py -3.12 scripts/test_teacher_scenario_2.py` and `py -3.12 scripts/test_teacher_demo_launcher.py`.
3. Ensure CARLA on `127.0.0.1:2000` responds to `get_world()` and `get_map()`.
4. Double-click `START_COMPARISON_DEMO.bat` for alternating scenarios.
5. Double-click `START_RAIN_SCENARIO_DEMO.bat` to loop **Scenario 2 only**.
6. Existing `START_TEACHER_DEMO.bat` remains **Scenario 1 only**.

Alternating rounds: **1 = Clear cooperative baseline**, **2 = Heavy rain with simulated V2I degradation**, then repeat. Each run recreates the CARLA scene, and the dashboard consumes a session-private live snapshot. Quit with `Q` in the camera window or launcher console, or close the dashboard.

## Scenario 2 fault mechanism

CARLA weather is configured for heavy precipitation, clouds, road wetness and fog. **Weather is visual only** in this project. Separately, an experiment-only impairment is injected into the RSU-to-Ego channel. CARLA rain **does not automatically cause these packet delays**.

| Parameter | Scenario 1 | Scenario 2 |
| --- | --- | --- |
| Weather | ClearNoon | Heavy rain |
| V2I latency model | 25 ± 5 ms (original simulated metadata) | 1200 ± 200 ms *actual sim-time queued delivery* |
| Random packet loss | 0% | 30%, fixed RNG seed 3163 |
| Deterministic blackout | None | 3.0 s following first RSU pedestrian detection |
| RSU / Onboard YOLO26n | Both active | Both active |
| Ego controller / Risk | Existing logic | Same existing logic |
| Scene geometry and initial speed | Original | Same original |

Under the injected failure, dropped packets are not fed to Fusion; future messages are not delivered early; packets older than 2.0 s are rejected. Onboard detections still proceed. The Dashboard shows **COOPERATIVE LIMITED**, **V2I DEGRADED**, and **UNCONFIRMED** where relevant.

## Honest outcomes

Each cycle produces an isolated `results/teacher_demo/runs/cycle_*/scenario_result.json` including:

- First RSU pedestrian observation, first fused pedestrian, and first braking trigger (CARLA sim times).
- Whether the controller confirmed stopping or the kinematic motion guard ended the scene.
- Minimum Ego-to-pedestrian **centre-point** separation and V2I drops/deliveries.
- An explicit `collision_sensor_available: false` and `physical_collision_observed: null`.

**No physical collision claim can be made from this demo.** In the existing scenario vehicle physics is disabled and the Ego is moved by deterministic transforms. A close approach / guard-trigger is a kinematic proximity event, not measured CARLA impact. For physical collisions a separate physics-enabled validation scene with a collision sensor would be needed.

This is a stress test and controlled fault injection, **not a calibrated meteorological propagation model**. Scenario 2 may still stop safely if enough valid perceptions arrive. Never label the observed outcome as failure/collision without the measured evidence.

To run one variant from the CLI:

```powershell
py -3.12 .\Perceptinet_Teacher_Demo.py --scenario rain_v2i --cycles 1
py -3.12 .\Perceptinet_Teacher_Demo.py --scenario clear --cycles 1
```
