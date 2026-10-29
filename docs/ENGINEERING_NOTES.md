# Engineering notes

A source-linked walkthrough of the current Perceptinet research prototype. For an overview and video demonstrations, return to the [README](../README.md).

## End-to-end processing path

| Stage | Implementation | Key engineering detail |
| --- | --- | --- |
| 1. Simulated inputs | [CARLA scenario runner](../scripts/run_occluded_pedestrian_braking_v1.py) | Synchronized onboard and roadside RGB frames from CARLA; moving walkers and occluders |
| 2. Object detection | [YOLO26n perception](../perception/yolo26.py) | Road-user class filtering: pedestrians, bicycles, cars, motorcycles, buses, trucks |
| 3. Spatial localization | [World-coordinate projection](../perception/world_coordinates.py) | Camera intrinsics/extrinsics and CARLA actor matching; simulator-assisted ground truth |
| 4. V2I communication | [Channel](../v2i/channel.py), [impaired delivery](../v2i/impaired_delivery.py) | Timestamped RSU packet metadata, queue delivery, packet loss, blackouts and stale observation rejection |
| 5. Cooperative fusion | [Fusion engine](../fusion/fusion_engine.py) | Actor-ID association where available, spatial-distance fallback, temporal tracks and confidence smoothing |
| 6. Hazard estimation | [Risk Engine](../risk/risk_engine.py) | Relative longitudinal/lateral position, closing speed, TTC, lane-path relevance |
| 7. Control | [Braking controller](../control/braking_controller.py) | Hazard-to-deceleration mapping and commanded target speed |
| 8. Observability | [Metrics](../metrics/collector.py), [dashboard](../dashboard/app.py) | Latency and braking events, Driver Mode / Pro Mode, per-cycle JSON and CSV |

## Risk model and braking response

The [Risk Engine](../risk/risk_engine.py) uses object positions in the ego frame, a path-width test, and relative closing speed. For an in-path target with positive closing speed, its prototype TTC estimate is:

```text
TTC = longitudinal distance / closing speed
```

The current prototype hazard thresholds are:

| Risk category | TTC in the demo | Controller behavior |
| --- | --- | --- |
| NORMAL | Not threatened / no valid TTC above risk thresholds | No braking |
| CAUTION | ≤ 5.0 s | Advisory; 0 m/s² commanded deceleration |
| WARNING | ≤ 3.0 s | 1.5 m/s² commanded deceleration |
| CRITICAL | ≤ 1.5 s | 6.0 m/s² commanded deceleration |

These thresholds and deceleration values are **demo configuration**, not automotive safety-certified parameters.

## Communication model

The baseline [V2IChannel](../v2i/channel.py) attaches a nominal **25 ± 5 ms** sampled latency to packets; this original path is primarily metadata-based.

The degraded-link experiment uses [ImpairedV2IDelivery](../v2i/impaired_delivery.py), which **withholds delivery** until the packet's simulated receive timestamp. Its controlled configuration is:

- **1,200 ± 200 ms** injected latency
- **30%** random packet loss, deterministic RNG seed **3163**
- **3 seconds** of dropped RSU messages beginning with the first detected pedestrian
- **2-second** maximum observation age (stale delivered packets rejected)
- Freshest valid due packet selected where multiple messages become available

This tests the consequences of reduced cooperative data availability. **Weather itself does not inject RF interference in CARLA.**

## Comparative experiment and evidence

The two scenarios share the same [walker crossing controller](../simulation/pedestrian_crossing.py) configuration, with a trigger distance of **11.5 m** and walking speed **0.95 m/s**. They differ in CARLA weather visuals and V2I delivery reliability, and are launched through [Perceptinet_Teacher_Demo.py](../Perceptinet_Teacher_Demo.py).

Results are recorded per cycle under `results/teacher_demo/runs/cycle_*/`:

- `scenario_result.json`: per-scenario outcome and contact observations
- `metrics_summary.json`: measured latency and detection/braking summaries
- `metrics_events.csv`: timestamped event log
- `occluded_pedestrian_world_v1.csv`: matched detections and world coordinates

The [report generator](../scripts/generate_teacher_demo_report.py) aggregates available completed cycles into PDF/CSV/JSON. **Missing runs are not treated as successes or failures.**

### Contact, stopping, and safety claims

- `controller_confirmed_stop`: the deterministic controller commanded/confirmed a stop in the kinematic ego model.
- `kinematic_contact`: actor footprint intersection in 2D; **not** evidence of physically simulated impact.
- `collision_sensor_event`: separate CARLA sensor observation when available; physics-disabled ego motion limits realism.
- `safe_stop_verified`: a conservative per-cycle predicate involving reported stop, walker movement, minimum measured centre separation, and no observed contact.

**No public road deployment, physical braking validation, calibrated wireless channel, or statistically established collision-avoidance claim is implied.**

## Reproducibility

```powershell
# Basic CARLA-independent checks (Python 3.12)
py -3.12 .\scripts\test_braking_controller.py
py -3.12 .\scripts\test_risk_engine.py
py -3.12 .\scripts\test_v2i_channel.py
py -3.12 .\scripts\test_metrics_collector.py
py -3.12 .\scripts\test_teacher_demo_report.py

# Local Windows CARLA demo with two alternating cycles
py -3.12 .\Perceptinet_Teacher_Demo.py --scenario alternate --cycles 2

# Report on real recorded results
py -3.12 -m pip install reportlab
py -3.12 .\scripts\generate_teacher_demo_report.py
```

The repository's [GitHub Actions workflow](../.github/workflows/core-checks.yml) runs the CARLA-independent subset on Python 3.12. Full end-to-end testing still requires CARLA, camera rendering, model weights and the configured inference device.
