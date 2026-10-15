# Perceptinet Final System Validation v1

This validation suite implements the project's baseline-vs-cooperative demo as
a repeatable paired experiment.

## Test A - Baseline control

- The Ego Vehicle receives only the Onboard Camera perception packet.
- RSU transmission is disabled.
- The RSU detector may run as a shadow evaluator, but RSU data is never passed
  into the Ego-side FusionEngine, RiskEngine, or BrakingController.
- The expected failure mode is late local visibility / near-miss.

## Test B - Cooperative perception

- Onboard and RSU perception both run.
- The RSU packet crosses the simulated V2I channel.
- Post-V2I RSU data is fused with Onboard detections.
- Risk assessment and braking consume the unified world model.
- The expected success mode is early warning and a safe stop.

## Primary metrics

- RSU stable-detection lead time.
- Onboard pedestrian detection rate during the scenario-defined occlusion
  window.
- RSU pedestrian detection rate during the same window.
- V2I mean and p95 latency.
- End-to-end mean and p95 latency.
- Brake-trigger distance and TTC.
- Actual braking distance.
- Baseline minimum pedestrian clearance.
- Cooperative final pedestrian clearance.
- Clearance gain:
  cooperative final clearance - baseline minimum clearance.
- Travel distance saved:
  baseline Ego travel - cooperative Ego travel.

## Evaluation-only ground truth

CARLA actor distance is used only for validation bookkeeping and the simulation
motion guard. It does **not** feed RiskEngine or BrakingController decisions.

The occlusion-rate denominator is a deterministic scenario window: frames are
counted as occluded while the occluding truck centre remains more than the
configured release distance ahead of the Ego Vehicle along the frozen path.

## Safety disclaimer

All TTC, clearance and deceleration thresholds are prototype/demo parameters.
They are not validated production automotive safety limits.

## Run

From the Windows CARLA VM, with CARLA already running:

```powershell
py -3.12 .\scripts\run_final_validation.py --trials 3
```

Outputs are written under:

```text
results/validation/final_validation_<timestamp>/
```

The suite generates:

- `final_validation_results.csv`
- `final_validation_summary.json`
- `final_validation_report.txt`
- one folder per baseline/cooperative run containing raw logs and run metrics.
