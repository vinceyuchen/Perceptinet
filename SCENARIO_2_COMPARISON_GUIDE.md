# Perceptinet Teacher Demo: Comparative Scenario 1 & Scenario 2

## Launch on the Windows CARLA machine

1. Update the repository from GitHub `main`.
2. Run `py -3.12 scripts/test_teacher_scenario_2.py` and `py -3.12 scripts/test_teacher_demo_launcher.py`.
3. Ensure CARLA on `127.0.0.1:2000` responds to `get_world()` and `get_map()`.
4. Double-click `START_COMPARISON_DEMO.bat` for alternating scenarios.
5. Double-click `START_RAIN_SCENARIO_DEMO.bat` to loop **Scenario 2 only**.
6. Existing `START_TEACHER_DEMO.bat` remains **Scenario 1 only**.

**Both Scenario 1 and Scenario 2 now use the same moving CARLA pedestrian.**
CARLA `WalkerControl` begins crossing when the pedestrian is approximately
11.5 m longitudinally ahead of Ego, walking at 0.95 m/s toward the opposite
side of the road. Both scenes use identical initial geometry, walker speed,
trigger distance, YOLO26n and Risk/Braking logic; only the weather and V2I
treatment differ. Scenario 1 is a normal-communication early-braking test,
not an automatically guaranteed successful stop. Per-cycle
`safe_stop_verified` is true only if the vehicle stopped, the walker truly
moved, no contact was observed, and minimum measured centre distance exceeds
the model contact envelope plus 0.3 m.

Both teacher-demo scenes attach a `sensor.other.collision` to Ego.
A CARLA walker collision sensor event is recorded separately from a
2D bounding-footprint overlap. Because Ego is still moved kinematically,
an envelope overlap is marked `kinematic_contact` / `CONTACT (SIM)`,
**not** a physically simulated crash. The Dashboard says `CARLA CONTACT`
only if CARLA reports an actual walker collision event.

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
- A collision-sensor availability flag and any observed CARLA collision events.
- An independent `kinematic_contact_observed` flag for 2D footprint overlap.
- `pedestrian_crossing_started`, `pedestrian_motion_verified`, and
  `pedestrian_walked_m` from actual CARLA pedestrian world-position updates.
- `kinematic_contact` completion when the 2D model envelopes overlap.
- `safe_stop_verified`, `crossing_started_sim_s`, and
  `brake_before_crossing_s` (positive means braking began before walking).
- `physical_collision_observed` is `null` when no sensor is available; with a
  sensor, it reflects actual reported walker collision **events**, not
  fabricated visual overlap.

**Do not equate a footprint overlap with a physical collision.**
The Ego still uses deterministic kinematic transforms with vehicle physics
disabled, even though the walker can now walk under CARLA control. A CARLA
collision sensor may emit an event in this configuration, but full impact
dynamics are not simulated. A real vehicle-physics collision demonstration
requires a separate physics-enabled Ego and validated sensor behavior.

This is a stress test and controlled fault injection, **not a calibrated meteorological propagation model**. Scenario 2 may still stop safely if enough valid perceptions arrive. Never label the observed outcome as failure/collision without the measured evidence.

To generate a measured PDF and CSV report after completing test cycles:

```powershell
py -3.12 -m pip install reportlab
py -3.12 .\scripts\generate_teacher_demo_report.py
```

Outputs are written to `results/teacher_demo/report`. The reporter is now
tracked in GitHub; no separate downloaded script is needed. With no run
data, fields remain pending and no collision or success is fabricated.

To run one variant from the CLI:

```powershell
py -3.12 .\Perceptinet_Teacher_Demo.py --scenario rain_v2i --cycles 1
py -3.12 .\Perceptinet_Teacher_Demo.py --scenario clear --cycles 1
```
