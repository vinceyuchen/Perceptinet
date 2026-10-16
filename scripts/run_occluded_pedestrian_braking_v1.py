import sys
import csv
import json
import os
import math
import threading
import time
from pathlib import Path

import carla
import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# Make repository root importable
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.world_coordinates import enrich_detection_packet
from v2i.channel import V2IChannel, V2IConfig
from v2i.impaired_delivery import ImpairedV2IDelivery
from simulation.teacher_scenarios import get_teacher_scenario, RAIN_V2I
from fusion.fusion_engine import FusionEngine
from risk.risk_engine import RiskEngine, RiskConfig
from control.braking_controller import BrakingController, BrakingConfig
from metrics.collector import MetricsCollector
from dashboard.data_layer import DashboardDataLayer
from simulation.road_context import RoadContextSampler
from simulation.demo_output_paths import create_teacher_demo_paths
from validation.runtime import ValidationConfig, ValidationRunTracker


# Deterministic kinematic Ego motion for Risk Transition v1.
# Physics remains disabled in this stage.
EGO_TEST_SPEED_MPS = 4.0
EGO_STOP_DISTANCE_M = 1.5

# Deterministic visual grounding. Actors remain physics-disabled and are
# translated only in Z so the lowest face of their CARLA bounding box sits
# just above the projected road surface.
SCENE_GROUND_CLEARANCE_M = 0.03
SCENE_GROUND_MAX_Z_ADJUST_M = 1.50

# Once the Ego Vehicle is fully stopped, require a short run
# of stable NORMAL observations before releasing the safety latch.
HAZARD_RELEASE_NORMAL_FRAMES = 10



# ============================================================
# Perceptinet - Automatic Braking Demo v1
# ============================================================

HOST = "127.0.0.1"
PORT = 2000
TM_PORT = 8000

SIM_FPS = 20
FIXED_DELTA = 1.0 / SIM_FPS

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOV = 90
CAMERA_TICK = FIXED_DELTA

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360

YOLO_MODEL_PATH = r"C:\CARLA\yolo26n.pt"
YOLO_DEVICE = 0
YOLO_IMAGE_SIZE = 1280
YOLO_CONFIDENCE = 0.25

TARGET_CLASS_IDS = [0, 1, 2, 3, 5, 7]

CLASS_NAME_OVERRIDES = {
    "person": "pedestrian",
}

OCCLUDER_FORWARD_DISTANCES = [17.0, 18.0, 16.0, 19.0]

# Keep the occluding truck beside the Ego centre path rather
# than directly inside it. The pedestrian remains behind the
# inner edge of the truck from the Onboard Camera viewpoint.
OCCLUDER_LATERAL_OFFSET_M = 2.7

PEDESTRIAN_FORWARD_FROM_OCCLUDER = 3.0

# Relative to the truck. With a +2.7 m truck offset these
# produce pedestrian lateral positions around +1.7 to +2.3 m
# from the Ego path, still inside RiskEngine's 2.5 m corridor.
PEDESTRIAN_LATERAL_OFFSETS = [-0.6, -0.8, -0.4, -1.0]

TRUCK_PATH_CLEARANCE_MIN_M = 0.10

RSU_FORWARD = 20.0
RSU_LATERAL = -10.0
RSU_HEIGHT = 7.0

latest_frames = {
    "onboard": None,
    "rsu_01": None,
}

latest_frame_ids = {
    "onboard": None,
    "rsu_01": None,
}

latest_timestamps = {
    "onboard": None,
    "rsu_01": None,
}

frame_lock = threading.Lock()


def carla_image_to_bgr(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3].copy()


def camera_callback(source, image):
    frame = carla_image_to_bgr(image)

    with frame_lock:
        latest_frames[source] = frame
        latest_frame_ids[source] = int(image.frame)
        latest_timestamps[source] = float(image.timestamp)


def add_header(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def add_status(frame, text, y=70):
    cv2.rectangle(frame, (8, y - 26), (610, y + 10), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def find_blueprints(blueprints, patterns):
    found = []
    seen = set()

    for pattern in patterns:
        for bp in blueprints.filter(pattern):
            if bp.id not in seen:
                found.append(bp)
                seen.add(bp.id)

    return found


def relative_location(base_transform, forward=0.0, lateral=0.0, vertical=0.0):
    forward_vec = base_transform.get_forward_vector()
    right_vec = base_transform.get_right_vector()

    return carla.Location(
        x=base_transform.location.x
        + forward_vec.x * forward
        + right_vec.x * lateral,
        y=base_transform.location.y
        + forward_vec.y * forward
        + right_vec.y * lateral,
        z=base_transform.location.z + vertical,
    )


REMOVABLE_TEST_ACTOR_PREFIXES = (
    "vehicle.",
    "walker.pedestrian.",
    "sensor.camera.",
    "controller.ai.walker",
)


def get_live_test_actors(world):
    return [
        actor
        for actor in world.get_actors()
        if actor.is_alive
        and actor.type_id.startswith(
            REMOVABLE_TEST_ACTOR_PREFIXES
        )
    ]


def _wait_for_actor_ids_gone(
    world,
    actor_ids,
    timeout_s=3.0,
):
    pending = {int(actor_id) for actor_id in actor_ids}
    deadline = time.perf_counter() + float(timeout_s)

    while pending and time.perf_counter() < deadline:
        current_ids = {
            actor.id
            for actor in get_live_test_actors(world)
        }

        pending.intersection_update(current_ids)

        if not pending:
            return True

        try:
            if world.get_settings().synchronous_mode:
                world.tick()
            else:
                world.wait_for_tick(0.5)
        except Exception:
            time.sleep(0.10)

    return not pending


def clear_stale_test_actors(
    client,
    world,
    *,
    max_attempts=3,
):
    """
    Remove dynamic actors left behind by interrupted demo/validation runs.

    CARLA actor destruction is asynchronous from the application's point of
    view. A successful DestroyActor command does not guarantee that a new
    actor can immediately reuse the same geometry on the very next Python
    line. This helper therefore destroys, waits, verifies, and retries before
    allowing scenario construction to continue.
    """

    total_removed = 0

    for attempt in range(1, int(max_attempts) + 1):
        stale_actors = get_live_test_actors(world)

        if not stale_actors:
            print(
                "Stale actor cleanup verified | "
                f"attempt={attempt} | remaining=0"
            )
            return total_removed

        stale_ids = [actor.id for actor in stale_actors]

        print(
            "Stale actor cleanup | "
            f"attempt={attempt}/{max_attempts} | "
            f"found={len(stale_ids)}"
        )

        responses = client.apply_batch_sync(
            [
                carla.command.DestroyActor(actor_id)
                for actor_id in stale_ids
            ],
            False,
        )

        failures = [
            response.error
            for response in responses
            if response.error
        ]

        removed_this_attempt = (
            len(stale_ids) - len(failures)
        )
        total_removed += removed_this_attempt

        if failures:
            print(
                "Stale actor cleanup command warnings | "
                + "; ".join(failures[:3])
            )

        _wait_for_actor_ids_gone(
            world,
            stale_ids,
            timeout_s=2.0,
        )

    remaining = get_live_test_actors(world)

    if remaining:
        detail = ", ".join(
            f"{actor.id}:{actor.type_id}"
            for actor in remaining[:8]
        )

        raise RuntimeError(
            "CARLA cleanup verification failed; "
            f"{len(remaining)} test actors remain: {detail}"
        )

    print(
        "Stale actor cleanup verified | "
        "remaining=0"
    )

    return total_removed


def destroy_tracked_actors(
    client,
    world,
    actors,
):
    """
    Stop sensors first, batch-destroy the tracked scenario actors, and verify
    that they are gone before returning control to the next run.
    """

    unique_actors = []
    seen_ids = set()

    for actor in reversed(actors):
        if actor is None:
            continue

        try:
            actor_id = int(actor.id)
        except Exception:
            continue

        if actor_id in seen_ids:
            continue

        seen_ids.add(actor_id)
        unique_actors.append(actor)

    for actor in unique_actors:
        try:
            if (
                actor.is_alive
                and actor.type_id.startswith("sensor.")
            ):
                actor.stop()
        except Exception:
            pass

    live_ids = []

    for actor in unique_actors:
        try:
            if actor.is_alive:
                live_ids.append(int(actor.id))
        except Exception:
            pass

    if not live_ids:
        print("Scenario cleanup verified | tracked_remaining=0")
        return True

    try:
        responses = client.apply_batch_sync(
            [
                carla.command.DestroyActor(actor_id)
                for actor_id in live_ids
            ],
            bool(world.get_settings().synchronous_mode),
        )

        failures = [
            response.error
            for response in responses
            if response.error
        ]

        if failures:
            print(
                "Scenario cleanup command warnings | "
                + "; ".join(failures[:3])
            )
    except Exception as exc:
        print(
            "Scenario cleanup batch warning | "
            f"{exc}"
        )

    gone = _wait_for_actor_ids_gone(
        world,
        live_ids,
        timeout_s=2.0,
    )

    if gone:
        print(
            "Scenario cleanup verified | "
            "tracked_remaining=0"
        )
        return True

    remaining_ids = {
        actor.id
        for actor in get_live_test_actors(world)
    }.intersection(live_ids)

    print(
        "Scenario cleanup warning | "
        f"tracked_remaining={sorted(remaining_ids)}"
    )

    return False


def spawn_first_available(
    world,
    blueprints,
    transforms,
    label,
    *,
    retry_rounds=3,
):
    if not blueprints:
        raise RuntimeError(
            f"No blueprints supplied for {label}"
        )

    if not transforms:
        raise RuntimeError(
            f"No transforms supplied for {label}"
        )

    for round_index in range(1, int(retry_rounds) + 1):
        for blueprint in blueprints:
            for transform in transforms:
                actor = world.try_spawn_actor(
                    blueprint,
                    transform,
                )

                if actor is not None:
                    print(
                        f"{label} spawned: {actor.id} "
                        f"({blueprint.id}) at "
                        f"x={transform.location.x:.1f}, "
                        f"y={transform.location.y:.1f}, "
                        f"z={transform.location.z:.1f} | "
                        f"spawn_round={round_index}"
                    )
                    return actor, transform

        print(
            f"{label} spawn retry | "
            f"round={round_index}/{retry_rounds} | "
            f"blueprints={len(blueprints)} | "
            f"transforms={len(transforms)}"
        )

        try:
            if world.get_settings().synchronous_mode:
                world.tick()
            else:
                world.wait_for_tick(0.5)
        except Exception:
            time.sleep(0.10)

    raise RuntimeError(
        f"Failed to spawn {label} after "
        f"{retry_rounds} retry rounds "
        f"({len(blueprints)} blueprints x "
        f"{len(transforms)} transforms)"
    )


def snap_actor_bbox_to_road(
    actor,
    world_map,
    label,
    *,
    reference_transform=None,
    clearance_m=SCENE_GROUND_CLEARANCE_M,
):
    """
    Ground one static demo actor without enabling CARLA physics.

    CARLA vehicle coordinates are normally close to the lowest face of the
    vehicle bounding box in Z, while pedestrian coordinates are normally near
    the middle of their bounding box. Using the actor's actual world-space
    bounding-box vertices handles both conventions consistently.
    """

    # Actors spawned during the current synchronous frame can briefly report
    # the identity transform until CARLA advances. Use the known spawn
    # transform as the authoritative XY/yaw reference when supplied.
    transform = (
        reference_transform
        if reference_transform is not None
        else actor.get_transform()
    )

    waypoint = world_map.get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if waypoint is None:
        raise RuntimeError(
            f"Could not project {label} onto driving surface"
        )

    road_z = float(
        waypoint.transform.location.z
    )

    vertices = actor.bounding_box.get_world_vertices(
        transform
    )

    if not vertices:
        raise RuntimeError(
            f"No bounding-box vertices for {label}"
        )

    bbox_bottom_z = min(
        float(vertex.z)
        for vertex in vertices
    )

    target_bottom_z = (
        road_z + float(clearance_m)
    )

    z_adjust_m = (
        target_bottom_z - bbox_bottom_z
    )

    if abs(z_adjust_m) > SCENE_GROUND_MAX_Z_ADJUST_M:
        raise RuntimeError(
            f"Ground snap for {label} is implausibly large: "
            f"dz={z_adjust_m:+.3f}m"
        )

    grounded = carla.Transform(
        carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z + z_adjust_m,
        ),
        transform.rotation,
    )

    actor.set_transform(grounded)

    print(
        f"Road snap | {label} | "
        f"road_z={road_z:.3f}m | "
        f"bbox_bottom={bbox_bottom_z:.3f}m | "
        f"dz={z_adjust_m:+.3f}m"
    )

    return grounded


def normalize_name(name):
    return CLASS_NAME_OVERRIDES.get(name, name)


def parse_result(result):
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

        detections.append(
            {
                "class_id": class_id,
                "class_name": normalize_name(result.names[class_id]),
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            }
        )

    return detections


def draw_detections(frame, detections):
    output = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = f'{det["class_name"]} {det["confidence"]:.2f}'

        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def get_pedestrian_detection(packet):
    for det in packet.get("detections", []):
        if det.get("class_name") == "pedestrian":
            return det

    return None


def world_position_text(detection):
    if detection is None:
        return "NO PEDESTRIAN"

    match = detection.get("world_match")

    if not match:
        return "PEDESTRIAN DETECTED | WORLD MATCH FAILED"

    pos = match["world_position"]

    return (
        f'PEDESTRIAN WORLD '
        f'({pos["x"]:.2f}, {pos["y"]:.2f}, {pos["z"]:.2f})'
    )


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)

    # CARLA Unreal can temporarily pause while loading or recovering
    # actors after a previous cycle. Retry only for the teacher-demo
    # launcher; preserve standalone and validation behavior.
    if os.environ.get("PERCEPTINET_TEACHER_DEMO") == "1":
        for attempt in range(3):
            try:
                world = client.get_world()
                break
            except RuntimeError as exc:
                message = str(exc).lower()
                if "time-out" not in message and "timeout" not in message:
                    raise
                if attempt == 2:
                    raise RuntimeError(
                        "CARLA get_world RPC remained unresponsive after "
                        "three attempts. Verify Unreal CARLA is not frozen "
                        "and manually restart the simulator if necessary."
                    ) from exc
                print(
                    "TEACHER DEMO | CARLA world RPC temporarily stalled "
                    f"(attempt {attempt + 1}/3). Retrying...",
                    flush=True,
                )
                time.sleep(3.0)
    else:
        world = client.get_world()

    print("Connected to CARLA")
    print("Map:", world.get_map().name)

    # Recover the dedicated test world if a previous validation process was
    # force-killed while CARLA was still in synchronous mode.
    startup_settings = world.get_settings()

    if startup_settings.synchronous_mode:
        startup_settings.synchronous_mode = False
        startup_settings.fixed_delta_seconds = None
        world.apply_settings(startup_settings)

        try:
            client.get_trafficmanager(
                TM_PORT
            ).set_synchronous_mode(False)
        except Exception:
            pass

        print("Recovered stale synchronous CARLA state")

    # Hard-killed validation runs can also leave old Ego/truck/walker/camera
    # actors behind. Remove them before constructing a fresh scenario.
    clear_stale_test_actors(client, world)

    original_settings = world.get_settings()
    original_weather = world.get_weather()
    actors = []
    traffic_manager = None

    validation_config = ValidationConfig.from_environment(
        REPO_ROOT
    )
    validation_tracker = ValidationRunTracker(
        validation_config
    )

    # Teacher Demo keeps the live Dashboard snapshot at its original
    # location, but must never overwrite an old CSV/metrics artifact.
    # Excel and other viewers can hold Windows write locks on those files.
    teacher_demo = (
        os.environ.get("PERCEPTINET_TEACHER_DEMO") == "1"
        and not validation_config.enabled
    )
    teacher_scenario = (
        get_teacher_scenario(
            os.environ.get("PERCEPTINET_DEMO_SCENARIO", "clear")
        )
        if teacher_demo else None
    )
    rain_demo = bool(
        teacher_scenario and teacher_scenario.key == RAIN_V2I
    )
    if teacher_scenario:
        print("TEACHER DEMO |", teacher_scenario.label)
    if validation_config.enabled:
        results_dir = validation_config.run_dir
        dashboard_snapshot_path = results_dir / "dashboard_snapshot.json"
    elif teacher_demo:
        results_dir, dashboard_snapshot_path = create_teacher_demo_paths(
            REPO_ROOT,
            os.environ.get("PERCEPTINET_DEMO_CYCLE_INDEX", "1"),
        )
        print("TEACHER DEMO | Isolated cycle metrics:", results_dir)
        dashboard_snapshot_path = Path(os.environ.get(
            "PERCEPTINET_DEMO_SNAPSHOT_FILE",
            str(dashboard_snapshot_path),
        ))
        dashboard_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        print("TEACHER DEMO | Live dashboard:", dashboard_snapshot_path)
    else:
        results_dir = REPO_ROOT / "results"
        dashboard_snapshot_path = results_dir / "dashboard_snapshot.json"

    results_dir.mkdir(parents=True, exist_ok=True)

    # Optional teacher-demo lifecycle. Real CARLA / YOLO / V2I / Fusion /
    # Risk / Braking pipeline remains unchanged.
    demo_stop_file = Path(os.environ.get(
        "PERCEPTINET_DEMO_STOP_FILE",
        str(results_dir / "teacher_demo_stop.signal"),
    ))
    demo_cycle_file = Path(os.environ.get(
        "PERCEPTINET_DEMO_CYCLE_FILE",
        str(results_dir / "teacher_demo_cycle.json"),
    ))
    try:
        demo_hold_s = max(1.0, min(15.0, float(
            os.environ.get("PERCEPTINET_DEMO_STOP_HOLD_SECONDS", "3.0")
        )))
    except ValueError:
        demo_hold_s = 3.0
    demo_stop_since = None
    dashboard_publish_failures = 0

    csv_path = results_dir / "occluded_pedestrian_world_v1.csv"
    metrics_summary_path = results_dir / "metrics_summary.json"
    metrics_events_path = results_dir / "metrics_events.csv"

    metrics_collector = MetricsCollector(
        stable_detection_frames=3
    )

    dashboard_data_layer = DashboardDataLayer(
        max_timeline_events=20
    )

    # Read-only road geometry for the Driver dashboard, separate from
    # V2I / perception / risk / braking. Failure leaves the demo intact.
    road_context_sampler = None
    try:
        road_context_sampler = RoadContextSampler(
            world.get_map(), max_segments=320
        )
        print("CARLA road-context geometry cache ready")
    except Exception as exc:
        print(f"Road-context visualisation unavailable: {exc}")

    model_path = Path(YOLO_MODEL_PATH)
    model = YOLO(str(model_path) if model_path.exists() else "yolo26n.pt")

    print("YOLO26n loaded")

    v2i_channel = V2IChannel(
        V2IConfig(
            base_latency_ms=25.0,
            jitter_ms=5.0,
            packet_loss_rate=0.0,
            random_seed=3163,
        )
    )
    degraded_v2i = (
        ImpairedV2IDelivery(
            V2IConfig(
                base_latency_ms=teacher_scenario.base_latency_ms,
                jitter_ms=teacher_scenario.jitter_ms,
                packet_loss_rate=teacher_scenario.packet_loss_rate,
                random_seed=3163,
            ),
            max_age_s=2.0,
            outage_after_detection_s=teacher_scenario.blackout_s,
        )
        if rain_demo else None
    )
    if rain_demo:
        print(
            "V2I FAULT INJECTION | sim-time delivery queue | "
            f"latency={teacher_scenario.base_latency_ms:.0f}ms +/- "
            f"{teacher_scenario.jitter_ms:.0f}ms | "
            f"loss={teacher_scenario.packet_loss_rate:.0%} | "
            f"first-pedestrian blackout={teacher_scenario.blackout_s:.1f}s"
        )
    else:
        print("V2I channel: 25 ms base latency +/- 5 ms jitter")

    fusion_engine = FusionEngine(
        fusion_distance_m=2.0,
        temporal_max_missed_frames=2,
        confidence_ema_alpha=0.65,
        display_position_ema_alpha=0.45,
        display_source_hold_frames=2,
    )

    print(
        "Fusion engine enabled | "
        "temporal coast=2 frames | "
        "stable track IDs enabled | "
        "display position EMA=0.45 | "
        "source hold=2 frames"
    )

    risk_engine = RiskEngine(
        RiskConfig(
            lane_half_width_m=2.5,
            caution_ttc_s=5.0,
            warning_ttc_s=3.0,
            critical_ttc_s=1.5,
        )
    )

    print("Risk engine enabled")
    print("Dashboard data layer enabled")

    braking_controller = BrakingController(
        BrakingConfig(
            caution_decel_mps2=0.0,
            warning_decel_mps2=1.5,
            critical_decel_mps2=6.0,
        )
    )

    # Persistent kinematic speed command.
    ego_command_speed_mps = EGO_TEST_SPEED_MPS

    # Cooperative braking hazard state.
    #
    # CAUTION is advisory and is never permanently latched.
    # WARNING and CRITICAL are safety-relevant braking states.
    # They remain latched through the active braking event so a
    # one-frame perception dropout cannot suddenly release braking.
    # After the vehicle has stopped, a stable NORMAL window releases
    # the latch and leaves the vehicle in a stopped/hold condition.
    latched_pedestrian_hazard = "NORMAL"
    last_observed_pedestrian_hazard = "NORMAL"
    normal_release_frames = 0
    braking_state = "DRIVING"

    hazard_priority = {
        "NORMAL": 0,
        "CAUTION": 1,
        "WARNING": 2,
        "CRITICAL": 3,
    }

    # Metrics for later Dashboard / evaluation.
    ego_travelled_m = 0.0
    brake_started = False
    brake_start_speed_mps = None
    brake_start_ttc_s = None
    brake_start_distance_m = None
    brake_start_travelled_m = None
    brake_estimated_stop_distance_m = None
    stop_reported = False
    motion_guard_triggered = False
    # Explicit kinematic safety/proximity observations, not fabricated
    # physical collision measurements (actor physics is disabled here).
    min_true_separation_m = math.inf
    first_rsu_pedestrian_time_s = None
    first_fused_pedestrian_time_s = None
    first_brake_time_s = None
    first_sim_time_s = None
    scenario_end_reason = None
    scenario_hold_since = None

    print(
        f"Automatic braking Ego cruise speed: "
        f"{EGO_TEST_SPEED_MPS:.1f} m/s"
    )
    print(
        "Braking controller: "
        "CAUTION=0.0, WARNING=1.5, CRITICAL=6.0 m/s^2"
    )
    print("World-coordinate association enabled")

    if validation_config.cooperative_enabled:
        print("V2I channel enabled")
    else:
        print("V2I channel disabled for BASELINE control run")

    if validation_config.enabled:
        print(
            "Final validation | "
            f"mode={validation_config.mode.upper()} | "
            f"run_id={validation_config.run_id} | "
            f"headless={validation_config.headless} | "
            f"auto_exit={validation_config.auto_exit}"
        )

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(
        [
            "frame_id",
            "timestamp",
            "source",
            "class_name",
            "confidence",
            "actor_id",
            "actor_type",
            "world_x",
            "world_y",
            "world_z",
            "camera_depth_m",
            "inference_ms",
        ]
    )

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(TM_PORT)
        traffic_manager.set_synchronous_mode(True)

        if rain_demo:
            world.set_weather(carla.WeatherParameters(
                cloudiness=teacher_scenario.cloudiness,
                precipitation=teacher_scenario.precipitation,
                precipitation_deposits=teacher_scenario.deposits,
                wetness=teacher_scenario.wetness,
                wind_intensity=teacher_scenario.wind_intensity,
                fog_density=teacher_scenario.fog_density,
                sun_altitude_angle=teacher_scenario.sun_altitude,
            ))
            print("CARLA WEATHER | HEAVY RAIN | visual weather only")
        else:
            world.set_weather(carla.WeatherParameters.ClearNoon)

        # Ensure the mode transition and startup cleanup are visible to the
        # synchronous simulation before any spawn collision checks occur.
        for _ in range(2):
            world.tick()

        print("CARLA spawn state settled")

        blueprints = world.get_blueprint_library()

        ego_candidates = find_blueprints(
            blueprints,
            ["vehicle.tesla.model3", "vehicle.*"],
        )

        occluder_candidates = find_blueprints(
            blueprints,
            [
                "vehicle.carlamotors.carlacola",
                "vehicle.carlamotors.european_hgv",
                "vehicle.*truck*",
                "vehicle.*bus*",
                "vehicle.*",
            ],
        )

        walker_candidates = find_blueprints(
            blueprints,
            ["walker.pedestrian.*"],
        )

        if not ego_candidates:
            raise RuntimeError("No ego vehicle blueprint found")

        if not occluder_candidates:
            raise RuntimeError("No occluding vehicle blueprint found")

        if not walker_candidates:
            raise RuntimeError("No pedestrian blueprint found")

        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))
        camera_bp.set_attribute("sensor_tick", str(CAMERA_TICK))

        spawn_points = world.get_map().get_spawn_points()

        ego_vehicle, ego_transform = spawn_first_available(
            world,
            ego_candidates,
            spawn_points,
            "Ego Vehicle",
            retry_rounds=3,
        )

        actors.append(ego_vehicle)
        ego_vehicle.set_simulate_physics(False)

        occluder_transforms = []

        for distance in OCCLUDER_FORWARD_DISTANCES:
            loc = relative_location(
                ego_transform,
                forward=distance,
                lateral=OCCLUDER_LATERAL_OFFSET_M,
                vertical=0.2,
            )

            occluder_transforms.append(
                carla.Transform(
                    loc,
                    carla.Rotation(
                        pitch=ego_transform.rotation.pitch,
                        yaw=ego_transform.rotation.yaw,
                        roll=ego_transform.rotation.roll,
                    ),
                )
            )

        occluder_vehicle, occluder_transform = spawn_first_available(
            world,
            occluder_candidates,
            occluder_transforms,
            "Occluding Vehicle",
        )

        actors.append(occluder_vehicle)
        occluder_vehicle.set_simulate_physics(False)

        # ----------------------------------------------------
        # Verify that the truck no longer intersects the
        # deterministic Ego centre path.
        #
        # Both actors use the same road heading in this demo,
        # therefore CARLA bounding-box half widths can be used
        # directly for this lateral clearance check.
        # ----------------------------------------------------

        ego_half_width_m = float(
            ego_vehicle.bounding_box.extent.y
        )

        truck_half_width_m = float(
            occluder_vehicle.bounding_box.extent.y
        )

        ego_right = ego_transform.get_right_vector()

        truck_dx = (
            occluder_transform.location.x
            - ego_transform.location.x
        )
        truck_dy = (
            occluder_transform.location.y
            - ego_transform.location.y
        )

        truck_lateral_m = (
            truck_dx * ego_right.x
            + truck_dy * ego_right.y
        )

        truck_path_clearance_m = (
            abs(truck_lateral_m)
            - ego_half_width_m
            - truck_half_width_m
        )

        print(
            "Truck path geometry | "
            f"lateral={truck_lateral_m:.2f}m | "
            f"ego_half_width={ego_half_width_m:.2f}m | "
            f"truck_half_width={truck_half_width_m:.2f}m | "
            f"clearance={truck_path_clearance_m:.2f}m"
        )

        if (
            truck_path_clearance_m
            < TRUCK_PATH_CLEARANCE_MIN_M
        ):
            raise RuntimeError(
                "Occluding truck intersects Ego path: "
                f"clearance={truck_path_clearance_m:.2f}m"
            )

        pedestrian_base = carla.Transform(
            occluder_transform.location,
            occluder_transform.rotation,
        )

        pedestrian_transforms = []

        for lateral in PEDESTRIAN_LATERAL_OFFSETS:
            loc = relative_location(
                pedestrian_base,
                forward=PEDESTRIAN_FORWARD_FROM_OCCLUDER,
                lateral=lateral,
                vertical=0.3,
            )

            pedestrian_transforms.append(
                carla.Transform(
                    loc,
                    carla.Rotation(
                        yaw=(ego_transform.rotation.yaw + 180.0) % 360.0
                    ),
                )
            )

        pedestrian, pedestrian_transform = spawn_first_available(
            world,
            walker_candidates,
            pedestrian_transforms,
            "Pedestrian",
        )

        actors.append(pedestrian)

        # ----------------------------------------------------
        # Deterministic visual grounding pass
        #
        # Never enable physics here. The validated demo uses deterministic
        # kinematic motion, so allowing CARLA physics to settle these actors
        # can change XY geometry unpredictably. Instead, move each actor only
        # in Z until its world-space bounding-box bottom is just above the
        # projected road surface.
        # ----------------------------------------------------

        pedestrian.set_simulate_physics(False)

        world_map = world.get_map()

        print("Ground snapping scene to road surface...")

        ego_spawn_transform = ego_transform
        truck_spawn_transform = occluder_transform
        pedestrian_spawn_transform = pedestrian_transform

        ego_transform = snap_actor_bbox_to_road(
            ego_vehicle,
            world_map,
            "Ego Vehicle",
            reference_transform=ego_spawn_transform,
        )
        occluder_transform = snap_actor_bbox_to_road(
            occluder_vehicle,
            world_map,
            "Occluding Vehicle",
            reference_transform=truck_spawn_transform,
        )
        pedestrian_transform = snap_actor_bbox_to_road(
            pedestrian,
            world_map,
            "Pedestrian",
            reference_transform=pedestrian_spawn_transform,
        )

        # Let transform updates become visible to sensors/snapshots.
        world.tick()

        ego_transform = ego_vehicle.get_transform()
        occluder_transform = occluder_vehicle.get_transform()
        pedestrian_transform = pedestrian.get_transform()

        print(
            "Road snap verified | "
            f"ego=({ego_transform.location.x:.2f}, "
            f"{ego_transform.location.y:.2f}, "
            f"{ego_transform.location.z:.2f}) | "
            f"truck=({occluder_transform.location.x:.2f}, "
            f"{occluder_transform.location.y:.2f}, "
            f"{occluder_transform.location.z:.2f}) | "
            f"pedestrian=({pedestrian_transform.location.x:.2f}, "
            f"{pedestrian_transform.location.y:.2f}, "
            f"{pedestrian_transform.location.z:.2f})"
        )

        def xy_error(actual, expected):
            return math.hypot(
                actual.location.x - expected.location.x,
                actual.location.y - expected.location.y,
            )

        ego_snap_xy_error_m = xy_error(
            ego_transform,
            ego_spawn_transform,
        )
        truck_snap_xy_error_m = xy_error(
            occluder_transform,
            truck_spawn_transform,
        )
        pedestrian_snap_xy_error_m = xy_error(
            pedestrian_transform,
            pedestrian_spawn_transform,
        )

        print(
            "Road snap XY preservation | "
            f"ego={ego_snap_xy_error_m:.4f}m | "
            f"truck={truck_snap_xy_error_m:.4f}m | "
            f"pedestrian={pedestrian_snap_xy_error_m:.4f}m"
        )

        if max(
            ego_snap_xy_error_m,
            truck_snap_xy_error_m,
            pedestrian_snap_xy_error_m,
        ) > 0.05:
            raise RuntimeError(
                "Road grounding changed scenario XY geometry: "
                f"ego={ego_snap_xy_error_m:.3f}m, "
                f"truck={truck_snap_xy_error_m:.3f}m, "
                f"pedestrian={pedestrian_snap_xy_error_m:.3f}m"
            )

        # Rebuild the Ego right vector from the final grounded heading so all
        # geometry checks use the same frame as motion and risk.
        ego_right = ego_transform.get_right_vector()

        settled_truck_dx = (
            occluder_transform.location.x
            - ego_transform.location.x
        )
        settled_truck_dy = (
            occluder_transform.location.y
            - ego_transform.location.y
        )
        settled_truck_lateral_m = (
            settled_truck_dx * ego_right.x
            + settled_truck_dy * ego_right.y
        )
        settled_truck_clearance_m = (
            abs(settled_truck_lateral_m)
            - ego_half_width_m
            - truck_half_width_m
        )

        print(
            "Grounded truck path geometry | "
            f"lateral={settled_truck_lateral_m:.2f}m | "
            f"clearance={settled_truck_clearance_m:.2f}m"
        )

        if (
            settled_truck_clearance_m
            < TRUCK_PATH_CLEARANCE_MIN_M
        ):
            raise RuntimeError(
                "Grounded occluding truck intersects Ego path: "
                f"clearance={settled_truck_clearance_m:.2f}m"
            )

        pedestrian_dx = (
            pedestrian_transform.location.x
            - ego_transform.location.x
        )
        pedestrian_dy = (
            pedestrian_transform.location.y
            - ego_transform.location.y
        )

        pedestrian_lateral_m = (
            pedestrian_dx * ego_right.x
            + pedestrian_dy * ego_right.y
        )

        print(
            "Pedestrian path geometry | "
            f"lateral={pedestrian_lateral_m:.2f}m | "
            f"risk_corridor="
            f"+/-{RiskConfig().lane_half_width_m:.2f}m"
        )

        # ----------------------------------------------------
        # Horizontal occlusion check
        #
        # Project the Ego->pedestrian line of sight onto the
        # longitudinal plane occupied by the truck.
        #
        # If that LOS passes through the truck's lateral bbox,
        # the truck can still occlude the pedestrian while
        # remaining physically clear of the Ego path.
        # ----------------------------------------------------

        ego_forward_geometry = (
            ego_transform.get_forward_vector()
        )

        truck_forward_m = (
            truck_dx * ego_forward_geometry.x
            + truck_dy * ego_forward_geometry.y
        )

        pedestrian_forward_m = (
            pedestrian_dx * ego_forward_geometry.x
            + pedestrian_dy * ego_forward_geometry.y
        )

        if pedestrian_forward_m <= 0.0:
            raise RuntimeError(
                "Pedestrian is not ahead of Ego"
            )

        los_lateral_at_truck_m = (
            pedestrian_lateral_m
            * truck_forward_m
            / pedestrian_forward_m
        )

        los_offset_from_truck_center_m = abs(
            los_lateral_at_truck_m
            - truck_lateral_m
        )

        horizontal_occlusion_margin_m = (
            truck_half_width_m
            - los_offset_from_truck_center_m
        )

        print(
            "Occlusion geometry | "
            f"truck_forward={truck_forward_m:.2f}m | "
            f"ped_forward={pedestrian_forward_m:.2f}m | "
            f"LOS_at_truck={los_lateral_at_truck_m:.2f}m | "
            f"truck_center={truck_lateral_m:.2f}m | "
            f"margin={horizontal_occlusion_margin_m:.2f}m"
        )

        if horizontal_occlusion_margin_m <= 0.0:
            raise RuntimeError(
                "Truck no longer horizontally occludes "
                "the pedestrian"
            )

        if (
            abs(pedestrian_lateral_m)
            > RiskConfig().lane_half_width_m
        ):
            raise RuntimeError(
                "Pedestrian is outside RiskEngine path corridor: "
                f"lateral={pedestrian_lateral_m:.2f}m"
            )

        # ----------------------------------------------------
        # Deterministic Ego path frame
        #
        # The scenario itself was constructed relative to the
        # original Ego transform. Freeze that same direction
        # for both motion and risk assessment.
        # ----------------------------------------------------

        raw_path_forward = ego_transform.get_forward_vector()

        path_norm = math.hypot(
            raw_path_forward.x,
            raw_path_forward.y,
        )

        if path_norm <= 1e-6:
            raise RuntimeError("Invalid Ego path direction")

        ego_path_forward = carla.Vector3D(
            x=raw_path_forward.x / path_norm,
            y=raw_path_forward.y / path_norm,
            z=0.0,
        )

        # Make absolutely sure the frozen direction points
        # toward the pedestrian rather than away from it.
        to_pedestrian_x = (
            pedestrian_transform.location.x
            - ego_transform.location.x
        )
        to_pedestrian_y = (
            pedestrian_transform.location.y
            - ego_transform.location.y
        )

        path_dot = (
            ego_path_forward.x * to_pedestrian_x
            + ego_path_forward.y * to_pedestrian_y
        )

        if path_dot < 0.0:
            ego_path_forward.x *= -1.0
            ego_path_forward.y *= -1.0

        # RiskEngine uses the mathematical heading generated
        # from this exact XY movement vector.
        ego_path_yaw_deg = math.degrees(
            math.atan2(
                ego_path_forward.y,
                ego_path_forward.x,
            )
        )

        # Preserve the original visual orientation for the
        # vehicle and attached Onboard Camera.
        ego_path_rotation = carla.Rotation(
            pitch=ego_transform.rotation.pitch,
            yaw=ego_transform.rotation.yaw,
            roll=ego_transform.rotation.roll,
        )

        initial_path_distance = (
            ego_transform.location.distance(
                pedestrian_transform.location
            )
        )

        print(
            "Ego path locked | "
            f"forward=("
            f"{ego_path_forward.x:.3f}, "
            f"{ego_path_forward.y:.3f}) | "
            f"risk_yaw={ego_path_yaw_deg:.2f}deg | "
            f"pedestrian_distance="
            f"{initial_path_distance:.2f}m"
        )

        onboard_camera = world.spawn_actor(
            camera_bp,
            carla.Transform(
                carla.Location(x=1.5, z=1.7),
                carla.Rotation(pitch=-5.0),
            ),
            attach_to=ego_vehicle,
        )

        actors.append(onboard_camera)
        onboard_camera.listen(
            lambda image: camera_callback("onboard", image)
        )

        rsu_location = relative_location(
            ego_transform,
            forward=RSU_FORWARD,
            lateral=RSU_LATERAL,
            vertical=RSU_HEIGHT,
        )

        direction = pedestrian_transform.location - rsu_location

        rsu_yaw = math.degrees(math.atan2(direction.y, direction.x))
        horizontal = math.sqrt(direction.x ** 2 + direction.y ** 2)
        rsu_pitch = math.degrees(math.atan2(direction.z, horizontal))

        rsu_camera = world.spawn_actor(
            camera_bp,
            carla.Transform(
                rsu_location,
                carla.Rotation(
                    pitch=rsu_pitch,
                    yaw=rsu_yaw,
                    roll=0.0,
                ),
            ),
        )

        actors.append(rsu_camera)
        rsu_camera.listen(
            lambda image: camera_callback("rsu_01", image)
        )

        spectator = world.get_spectator()

        spectator.set_transform(
            carla.Transform(
                relative_location(
                    ego_transform,
                    forward=12.0,
                    vertical=16.0,
                ),
                carla.Rotation(
                    pitch=-55.0,
                    yaw=ego_transform.rotation.yaw,
                    roll=0.0,
                ),
            )
        )

        # ----------------------------------------------------
        # Settle CARLA actor state before the control loop.
        #
        # In synchronous mode, actors spawned in the current
        # frame may report stale transforms until the world
        # has advanced. Motion/risk logic must never consume
        # those un-synchronised transforms.
        # ----------------------------------------------------

        print()
        print("Settling CARLA scene...")

        for _ in range(2):
            world.tick()

        settled_ego = ego_vehicle.get_transform()
        settled_pedestrian = pedestrian.get_transform()

        settled_distance_m = (
            settled_ego.location.distance(
                settled_pedestrian.location
            )
        )

        print(
            "Scenario settled | "
            f"ego=("
            f"{settled_ego.location.x:.2f}, "
            f"{settled_ego.location.y:.2f}) | "
            f"pedestrian=("
            f"{settled_pedestrian.location.x:.2f}, "
            f"{settled_pedestrian.location.y:.2f}) | "
            f"distance={settled_distance_m:.2f}m"
        )

        # Fail loudly instead of silently zeroing Ego speed
        # if CARLA actor state is still inconsistent.
        if settled_distance_m < 5.0:
            raise RuntimeError(
                "Invalid settled scenario geometry: "
                f"Ego-pedestrian distance="
                f"{settled_distance_m:.2f}m"
            )

        print()
        print("World Coordinate Test running")
        print("Expected:")
        print("  Onboard: pedestrian not detected")
        print("  RSU-01 : pedestrian detected + world position")
        print()
        print(f"CSV: {csv_path}")
        print("Press Q / Esc or Ctrl+C to stop.")

        last_processed_frame = None

        while True:
            loop_start = time.perf_counter()

            # ------------------------------------------------
            # Automatic Braking v1
            #
            # Vehicle physics remains disabled in this stage.
            # ego_command_speed_mps is updated by the braking
            # controller after each Risk Engine assessment.
            # ------------------------------------------------

            # Speed commanded for THIS simulation tick.
            # Keep this separate from the speed command that
            # will be calculated for the NEXT tick.
            ego_speed_this_tick_mps = ego_command_speed_mps
            ego_motion_speed_mps = 0.0

            fixed_delta = (
                world.get_settings().fixed_delta_seconds
                or 0.05
            )

            ego_now = ego_vehicle.get_transform()
            pedestrian_now = pedestrian.get_transform()

            ego_to_pedestrian_m = ego_now.location.distance(
                pedestrian_now.location
            )

            # Do not stop the vehicle here using raw CARLA
            # actor distance. The final failsafe is applied
            # later using the fused pedestrian Risk result.

            if ego_speed_this_tick_mps > 0.0:
                step_m = (
                    ego_speed_this_tick_mps
                    * fixed_delta
                )

                next_ego_location = carla.Location(
                    x=(
                        ego_now.location.x
                        + ego_path_forward.x * step_m
                    ),
                    y=(
                        ego_now.location.y
                        + ego_path_forward.y * step_m
                    ),
                    z=ego_now.location.z,
                )

                next_distance_to_pedestrian = (
                    next_ego_location.distance(
                        pedestrian_now.location
                    )
                )

                # Simulation-only motion guard.
                # A moving Ego must get closer to the test
                # pedestrian. Stop immediately if the geometry
                # becomes inconsistent instead of allowing a
                # runaway vehicle through the CARLA map.
                if (
                    next_distance_to_pedestrian
                    > ego_to_pedestrian_m + 0.001
                ):
                    ego_command_speed_mps = 0.0
                    ego_speed_this_tick_mps = 0.0
                    motion_guard_triggered = True

                    if validation_config.enabled:
                        validation_tracker.mark_motion_guard(
                            distance_m=ego_to_pedestrian_m,
                        )

                    print(
                        "MOTION GUARD STOP | "
                        f"before={ego_to_pedestrian_m:.2f}m | "
                        f"after="
                        f"{next_distance_to_pedestrian:.2f}m"
                    )

                else:
                    next_ego_transform = carla.Transform(
                        next_ego_location,
                        ego_path_rotation,
                    )

                    ego_vehicle.set_transform(
                        next_ego_transform
                    )

                    ego_motion_speed_mps = (
                        ego_speed_this_tick_mps
                    )

                    ego_travelled_m += step_m

            world.tick()

            with frame_lock:
                onboard_frame = (
                    None
                    if latest_frames["onboard"] is None
                    else latest_frames["onboard"].copy()
                )
                rsu_frame = (
                    None
                    if latest_frames["rsu_01"] is None
                    else latest_frames["rsu_01"].copy()
                )

                onboard_frame_id = latest_frame_ids["onboard"]
                rsu_frame_id = latest_frame_ids["rsu_01"]

                onboard_timestamp = latest_timestamps["onboard"]
                rsu_timestamp = latest_timestamps["rsu_01"]

            if (
                onboard_frame is None
                or rsu_frame is None
                or onboard_frame_id is None
                or rsu_frame_id is None
            ):
                continue

            if onboard_frame_id != rsu_frame_id:
                continue

            if onboard_frame_id == last_processed_frame:
                continue

            last_processed_frame = onboard_frame_id
            if first_sim_time_s is None:
                first_sim_time_s = float(onboard_timestamp)
            true_separation = ego_vehicle.get_transform().location.distance(
                pedestrian.get_transform().location
            )
            min_true_separation_m = min(min_true_separation_m, true_separation)

            pipeline_start = time.perf_counter()

            results = model.predict(
                source=[onboard_frame, rsu_frame],
                device=YOLO_DEVICE,
                imgsz=YOLO_IMAGE_SIZE,
                conf=YOLO_CONFIDENCE,
                classes=TARGET_CLASS_IDS,
                verbose=False,
            )

            onboard_detections = parse_result(results[0])
            rsu_detections = parse_result(results[1])

            onboard_packet = {
                "source": "onboard",
                "frame_id": onboard_frame_id,
                "timestamp": onboard_timestamp,
                "detections": onboard_detections,
                "inference_ms": float(
                    (results[0].speed or {}).get("inference", 0.0)
                ),
            }

            rsu_packet = {
                "source": "rsu_01",
                "frame_id": rsu_frame_id,
                "timestamp": rsu_timestamp,
                "detections": rsu_detections,
                "inference_ms": float(
                    (results[1].speed or {}).get("inference", 0.0)
                ),
            }

            onboard_packet = enrich_detection_packet(
                world,
                onboard_camera,
                onboard_packet,
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
                CAMERA_FOV,
            )

            rsu_packet = enrich_detection_packet(
                world,
                rsu_camera,
                rsu_packet,
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
                CAMERA_FOV,
            )

            if (first_rsu_pedestrian_time_s is None
                    and get_pedestrian_detection(rsu_packet) is not None):
                first_rsu_pedestrian_time_s = float(rsu_timestamp)

            # Feed real per-camera detections/inference timing
            # into Metrics v1 before the remote packet crosses V2I.
            metrics_collector.record_perception_packet(
                onboard_packet
            )

            # ------------------------------------------------
            # V2I / Baseline control selection
            #
            # COOPERATIVE:
            #   onboard + post-V2I RSU packets feed FusionEngine.
            #
            # BASELINE:
            #   RSU transmission is OFF and only the onboard
            #   packet is available to the Ego-side fusion/risk
            #   pipeline. The RSU detector may still run as a
            #   shadow evaluator, but its data never influences
            #   control decisions.
            # ------------------------------------------------

            fusion_packets = [onboard_packet]
            delivered_rsu = None

            if validation_config.cooperative_enabled:
                if rain_demo:
                    # The original V2I model records simulated delay but
                    # returns the packet immediately. This scenario queues
                    # messages until the actual CARLA sim-time receive
                    # deadline; dropped packets must NOT skip Onboard risk.
                    degraded_v2i.offer(
                        rsu_packet, sim_time_s=onboard_timestamp
                    )
                    delivered_rsu = degraded_v2i.receive_ready(
                        sim_time_s=onboard_timestamp
                    )
                    if delivered_rsu is not None:
                        fusion_packets.append(delivered_rsu)
                        metrics_collector.record_perception_packet(
                            delivered_rsu
                        )
                else:
                    received_rsu_packet = v2i_channel.transmit(
                        rsu_packet, source_id="RSU-01"
                    )
                    if received_rsu_packet is None:
                        continue
                    rsu_packet = received_rsu_packet
                    metrics_collector.record_perception_packet(rsu_packet)
                    fusion_packets.append(rsu_packet)

            fusion_start = time.perf_counter()

            fused_world = fusion_engine.fuse(
                fusion_packets
            )

            fusion_latency_ms = (
                time.perf_counter() - fusion_start
            ) * 1000.0

            fused_pedestrian = next(
                (
                    obj
                    for obj in fused_world["objects"]
                    if obj["class_name"] == "pedestrian"
                ),
                None,
            )
            if (fused_pedestrian is not None
                    and first_fused_pedestrian_time_s is None):
                first_fused_pedestrian_time_s = float(onboard_timestamp)

            # ------------------------------------------------
            # Risk Assessment v1
            #
            # Read REAL Ego Vehicle state directly from CARLA.
            # ------------------------------------------------

            ego_transform = ego_vehicle.get_transform()
            # Physics is disabled in this deterministic
            # scenario, therefore CARLA get_velocity() remains
            # zero. Use the actual commanded kinematic speed
            # for Risk Assessment v1.
            ego_speed_mps = ego_speed_this_tick_mps

            ego_state = {
                "world_position": {
                    "x": float(ego_transform.location.x),
                    "y": float(ego_transform.location.y),
                    "z": float(ego_transform.location.z),
                },
                "yaw_deg": float(ego_path_yaw_deg),
                "speed_mps": float(ego_speed_mps),
            }

            risk_start = time.perf_counter()

            risk_result = risk_engine.assess_world(
                ego_state,
                fused_world,
            )

            risk_latency_ms = (
                time.perf_counter() - risk_start
            ) * 1000.0

            if validation_config.enabled:
                evaluation_ego = ego_vehicle.get_transform()
                evaluation_pedestrian = pedestrian.get_transform()
                evaluation_truck = occluder_vehicle.get_transform()

                evaluation_distance_m = (
                    evaluation_ego.location.distance(
                        evaluation_pedestrian.location
                    )
                )

                truck_dx_eval = (
                    evaluation_truck.location.x
                    - evaluation_ego.location.x
                )
                truck_dy_eval = (
                    evaluation_truck.location.y
                    - evaluation_ego.location.y
                )

                truck_ahead_m = (
                    ego_path_forward.x * truck_dx_eval
                    + ego_path_forward.y * truck_dy_eval
                )

                validation_tracker.observe_frame(
                    frame_id=onboard_frame_id,
                    sim_time_s=onboard_timestamp,
                    ego_to_pedestrian_m=evaluation_distance_m,
                    truck_ahead_m=truck_ahead_m,
                    onboard_detected=(
                        get_pedestrian_detection(
                            onboard_packet
                        )
                        is not None
                    ),
                    rsu_detected=(
                        get_pedestrian_detection(
                            rsu_packet
                        )
                        is not None
                    ),
                    ego_travelled_m=ego_travelled_m,
                    ego_speed_mps=ego_speed_this_tick_mps,
                )

            pedestrian_risk = next(
                (
                    risk
                    for risk in risk_result["risks"]
                    if risk["class_name"] == "pedestrian"
                ),
                None,
            )

            # ------------------------------------------------
            # Pedestrian-driven automatic braking
            # ------------------------------------------------

            observed_pedestrian_hazard = "NORMAL"

            if pedestrian_risk is not None:
                observed_pedestrian_hazard = (
                    pedestrian_risk["hazard_level"]
                )

            # Log the raw RiskEngine state independently from
            # the braking state used by the controller.
            if (
                observed_pedestrian_hazard
                != last_observed_pedestrian_hazard
            ):
                print(
                    "HAZARD OBSERVED | "
                    f"{last_observed_pedestrian_hazard} -> "
                    f"{observed_pedestrian_hazard}"
                )

                metrics_collector.record_hazard_transition(
                    old_level=last_observed_pedestrian_hazard,
                    new_level=observed_pedestrian_hazard,
                    sim_time_s=onboard_timestamp,
                    frame_id=onboard_frame_id,
                )

                last_observed_pedestrian_hazard = (
                    observed_pedestrian_hazard
                )

            # CAUTION is advisory only.
            #
            # Start safety latching at WARNING so detector
            # dropouts cannot release an active braking event.
            if (
                hazard_priority[observed_pedestrian_hazard]
                >= hazard_priority["WARNING"]
                and
                hazard_priority[observed_pedestrian_hazard]
                > hazard_priority[latched_pedestrian_hazard]
            ):
                print(
                    "HAZARD LATCH | "
                    f"{latched_pedestrian_hazard} -> "
                    f"{observed_pedestrian_hazard}"
                )

                latched_pedestrian_hazard = (
                    observed_pedestrian_hazard
                )

            # Once stopped, keep the safety latch only until
            # RiskEngine has reported NORMAL continuously for a
            # short persistence window. This prevents endless
            # CRITICAL braking commands after the vehicle is already
            # stationary while still rejecting one-frame dropouts.
            if braking_state == "STOPPED_HOLD":
                if observed_pedestrian_hazard == "NORMAL":
                    normal_release_frames += 1
                else:
                    normal_release_frames = 0

                if (
                    normal_release_frames
                    >= HAZARD_RELEASE_NORMAL_FRAMES
                    and latched_pedestrian_hazard != "NORMAL"
                ):
                    print(
                        "HAZARD RELEASE | "
                        f"{latched_pedestrian_hazard} -> NORMAL | "
                        f"normal_frames={normal_release_frames}"
                    )

                    latched_pedestrian_hazard = "NORMAL"
                    braking_state = "STOPPED_RELEASED"

            if (
                hazard_priority[latched_pedestrian_hazard]
                >= hazard_priority["WARNING"]
            ):
                control_pedestrian_hazard = (
                    latched_pedestrian_hazard
                )
            else:
                control_pedestrian_hazard = (
                    observed_pedestrian_hazard
                )

            control_start = time.perf_counter()

            braking_command = braking_controller.update(
                current_speed_mps=ego_speed_this_tick_mps,
                hazard_level=control_pedestrian_hazard,
                dt_s=fixed_delta,
            )

            control_latency_ms = (
                time.perf_counter() - control_start
            ) * 1000.0

            end_to_end_latency_ms = (
                time.perf_counter() - pipeline_start
            ) * 1000.0

            metrics_collector.record_pipeline_timing(
                fusion_ms=fusion_latency_ms,
                risk_ms=risk_latency_ms,
                control_ms=control_latency_ms,
                end_to_end_ms=end_to_end_latency_ms,
            )

            # This becomes the speed for the NEXT CARLA tick.
            ego_command_speed_mps = braking_command[
                "target_speed_mps"
            ]

            # Final simulation failsafe.
            # Use the SAME fused pedestrian representation
            # consumed by RiskEngine rather than a separate
            # raw CARLA actor-distance path.
            if (
                pedestrian_risk is not None
                and pedestrian_risk["distance_m"]
                <= EGO_STOP_DISTANCE_M
            ):
                ego_command_speed_mps = 0.0

                print(
                    "FINAL FAILSAFE STOP | "
                    f'distance='
                    f'{pedestrian_risk["distance_m"]:.2f}m'
                )

            if (
                braking_command["braking_active"]
                and not brake_started
            ):
                brake_started = True
                first_brake_time_s = float(onboard_timestamp)

                brake_start_speed_mps = (
                    braking_command["speed_before_mps"]
                )

                brake_start_travelled_m = ego_travelled_m

                brake_estimated_stop_distance_m = (
                    braking_command[
                        "estimated_stopping_distance_m"
                    ]
                )

                if pedestrian_risk is not None:
                    brake_start_distance_m = (
                        pedestrian_risk["distance_m"]
                    )

                    brake_start_ttc_s = (
                        pedestrian_risk["ttc_s"]
                    )

                print(
                    "BRAKE TRIGGERED | "
                    f"hazard={control_pedestrian_hazard} | "
                    f"speed={brake_start_speed_mps:.2f}m/s | "
                    f"distance="
                    f"{brake_start_distance_m:.2f}m | "
                    f"TTC="
                    f"{brake_start_ttc_s:.2f}s"
                    if brake_start_ttc_s is not None
                    else
                    "BRAKE TRIGGERED | TTC=N/A"
                )

                metrics_collector.record_brake_trigger(
                    sim_time_s=onboard_timestamp,
                    frame_id=onboard_frame_id,
                    distance_m=brake_start_distance_m,
                    ttc_s=brake_start_ttc_s,
                    speed_mps=brake_start_speed_mps,
                    hazard_level=control_pedestrian_hazard,
                )

                if validation_config.enabled:
                    validation_tracker.mark_brake_trigger(
                        distance_m=brake_start_distance_m,
                        ttc_s=brake_start_ttc_s,
                        speed_mps=brake_start_speed_mps,
                        hazard_level=control_pedestrian_hazard,
                    )

            if (
                braking_command["braking_active"]
                and not stop_reported
            ):
                stop_distance = braking_command[
                    "estimated_stopping_distance_m"
                ]

                stop_text = (
                    "N/A"
                    if stop_distance is None
                    else f"{stop_distance:.2f}m"
                )

                print(
                    f'BRAKE | '
                    f'hazard={control_pedestrian_hazard} | '
                    f'speed='
                    f'{braking_command["speed_before_mps"]:.2f}'
                    f'->'
                    f'{braking_command["target_speed_mps"]:.2f}'
                    f'm/s | '
                    f'decel='
                    f'{braking_command["deceleration_mps2"]:.2f}'
                    f'm/s^2 | '
                    f'est_stop={stop_text}'
                )

            # Final stop report.
            if (
                brake_started
                and ego_command_speed_mps <= 0.001
                and not stop_reported
            ):
                actual_braking_distance = (
                    ego_travelled_m
                    - brake_start_travelled_m
                )

                clearance_text = "N/A"

                if pedestrian_risk is not None:
                    clearance_text = (
                        f'{pedestrian_risk["distance_m"]:.2f}m'
                    )

                print(
                    "VEHICLE STOPPED | "
                    f"actual_braking_distance="
                    f"{actual_braking_distance:.2f}m | "
                    f"clearance={clearance_text} | "
                    f"hazard={control_pedestrian_hazard}"
                )

                final_clearance_m = (
                    None
                    if pedestrian_risk is None
                    else pedestrian_risk["distance_m"]
                )

                metrics_collector.record_stop(
                    sim_time_s=onboard_timestamp,
                    frame_id=onboard_frame_id,
                    actual_braking_distance_m=actual_braking_distance,
                    final_clearance_m=final_clearance_m,
                    hazard_level=control_pedestrian_hazard,
                )

                if validation_config.enabled:
                    validation_tracker.mark_system_stop(
                        final_clearance_m=final_clearance_m,
                        actual_braking_distance_m=(
                            actual_braking_distance
                        ),
                    )

                metrics_collector.export_summary_json(
                    metrics_summary_path
                )
                metrics_collector.export_events_csv(
                    metrics_events_path
                )

                print(
                    "METRICS SAVED | "
                    f"summary={metrics_summary_path.name} | "
                    f"events={metrics_events_path.name}"
                )

                stop_reported = True
                braking_state = "STOPPED_HOLD"
                normal_release_frames = 0

                print(
                    "BRAKING STATE | DRIVING -> STOPPED_HOLD"
                )

            # ------------------------------------------------
            # Live Dashboard Data Layer v1
            #
            # Publish one normalized snapshot consumed by both
            # Driver Mode and Pro Mode. Raw metrics remain
            # separate from the filtered dashboard timeline.
            # ------------------------------------------------

            road_context = None
            if road_context_sampler is not None:
                try:
                    road_context = road_context_sampler.snapshot(ego_state)
                except Exception as exc:
                    print(f"Road-context update unavailable: {exc}")
                    road_context_sampler = None

            dashboard_snapshot = (
                dashboard_data_layer.build_snapshot(
                    road_context=road_context,
                    frame_id=onboard_frame_id,
                    sim_time_s=onboard_timestamp,
                    ego_state=ego_state,
                    fused_world=fused_world,
                    risk_result=risk_result,
                    braking_command=braking_command,
                    braking_state=braking_state,
                    control_hazard=control_pedestrian_hazard,
                    metrics_summary=metrics_collector.summary(),
                    metrics_events=metrics_collector.events,
                    live_latencies={
                        "onboard_inference_ms":
                            onboard_packet["inference_ms"],
                        "rsu_inference_ms":
                            rsu_packet["inference_ms"],
                        "v2i_latency_ms":
                            (
                                ((delivered_rsu if rain_demo else rsu_packet)
                                 or {}).get("v2i", {}).get("latency_ms")
                            ),
                        "fusion_latency_ms":
                            fusion_latency_ms,
                        "risk_latency_ms":
                            risk_latency_ms,
                        "control_latency_ms":
                            control_latency_ms,
                        "end_to_end_latency_ms":
                            end_to_end_latency_ms,
                    },
                )
            )

            if teacher_scenario is not None:
                dashboard_snapshot["experiment"] = {
                    "scenario": teacher_scenario.key,
                    "label": teacher_scenario.label,
                    "weather": "HEAVY RAIN" if rain_demo else "CLEAR",
                    "fault_injected": rain_demo,
                    "v2i": degraded_v2i.stats() if rain_demo else {
                        "fault_injected": False,
                        "base_latency_ms": 25.0,
                        "packet_loss_rate": 0.0,
                    },
                }
                if rain_demo:
                    dashboard_snapshot["system_health"]["status"] = (
                        "DATA LIMITED"
                    )
            try:
                dashboard_data_layer.write_snapshot_json(
                    dashboard_snapshot,
                    dashboard_snapshot_path,
                )
                dashboard_publish_failures = 0
            except PermissionError as exc:
                if not teacher_demo:
                    raise
                dashboard_publish_failures += 1
                if dashboard_publish_failures == 1 or dashboard_publish_failures % 10 == 0:
                    print(
                        "TEACHER DEMO | Snapshot publish blocked; "
                        f"dashboard may show DATA STALE (failures={dashboard_publish_failures}): "
                        f"{exc}"
                    )
                # Dashboard transport is presentation only. Never abort
                # CARLA's independent risk/braking controller because a
                # Windows reader temporarily locks a JSON file.

            if pedestrian_risk is not None:
                ttc = pedestrian_risk["ttc_s"]

                ttc_text = (
                    "N/A"
                    if ttc is None
                    else f"{ttc:.2f}s"
                )

                print(
                    f'RISK | '
                    f'ego_speed={ego_speed_mps:.2f}m/s | '
                    f'distance={pedestrian_risk["distance_m"]:.2f}m | '
                    f'long={pedestrian_risk["longitudinal_m"]:.2f}m | '
                    f'lat={pedestrian_risk["lateral_m"]:.2f}m | '
                    f'TTC={ttc_text} | '
                    f'hazard={pedestrian_risk["hazard_level"]} | '
                    f'sources={pedestrian_risk["sources"]}'
                )

            if fused_pedestrian is not None:
                fused_pos = fused_pedestrian["world_position"]

                print(
                    f'FUSION | '
                    f'objects={fused_world["object_count"]} | '
                    f'pedestrian actor={fused_pedestrian["actor_id"]} | '
                    f'sources={fused_pedestrian["sources"]} | '
                    f'world=('
                    f'{fused_pos["x"]:.2f}, '
                    f'{fused_pos["y"]:.2f}, '
                    f'{fused_pos["z"]:.2f})'
                )

            onboard_ped = get_pedestrian_detection(onboard_packet)
            rsu_ped = get_pedestrian_detection(rsu_packet)

            if not validation_config.headless:
                onboard_view = draw_detections(
                    onboard_frame,
                    onboard_packet["detections"],
                )

                rsu_view = draw_detections(
                    rsu_frame,
                    rsu_packet["detections"],
                )

                onboard_view = cv2.resize(
                    onboard_view,
                    (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )

                rsu_view = cv2.resize(
                    rsu_view,
                    (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )

                if rain_demo:
                    add_header(onboard_view, "ONBOARD / HEAVY RAIN")
                    add_header(rsu_view, "RSU-01 / V2I FAULT TEST")
                else:
                    add_header(onboard_view, "ONBOARD PERCEPTION")
                    add_header(rsu_view, "RSU-01 PERCEPTION")

                add_status(
                    onboard_view,
                    "Pedestrian: NOT DETECTED"
                    if onboard_ped is None
                    else world_position_text(onboard_ped),
                )

                add_status(
                    rsu_view,
                    world_position_text(rsu_ped),
                )
                if rain_demo:
                    if (degraded_v2i.outage_started_s is not None
                            and float(onboard_timestamp)
                            - degraded_v2i.outage_started_s
                            < teacher_scenario.blackout_s):
                        channel_overlay = "INJECTED V2I BLACKOUT"
                    elif delivered_rsu is None:
                        channel_overlay = "RSU DATA DELAYED / DROPPED"
                    else:
                        channel_overlay = (
                            "V2I ARRIVED: "
                            f"{delivered_rsu['v2i']['latency_ms']:.0f}ms"
                        )
                    add_status(rsu_view, channel_overlay, y=113)

                combined = np.hstack(
                    (onboard_view, rsu_view)
                )

                cv2.imshow(
                    "Perceptinet - Automatic Braking Demo v1",
                    combined,
                )

            # Save every matched detection to CSV.
            for packet in (onboard_packet, rsu_packet):
                for det in packet["detections"]:
                    match = det.get("world_match")

                    if not match:
                        continue

                    pos = match["world_position"]

                    csv_writer.writerow(
                        [
                            packet["frame_id"],
                            packet["timestamp"],
                            packet["source"],
                            det["class_name"],
                            det["confidence"],
                            match["actor_id"],
                            match["actor_type"],
                            pos["x"],
                            pos["y"],
                            pos["z"],
                            match["camera_depth_m"],
                            packet["inference_ms"],
                        ]
                    )

            csv_file.flush()

            if rsu_ped is not None and rsu_ped.get("world_match") is not None:
                match = rsu_ped["world_match"]
                pos = match["world_position"]

                print(
                    f'frame={rsu_frame_id} | '
                    f'RSU pedestrian conf={rsu_ped["confidence"]:.2f} | '
                    f'actor={match["actor_id"]} | '
                    f'world=({pos["x"]:.2f}, {pos["y"]:.2f}, {pos["z"]:.2f}) | '
                    f'depth={match["camera_depth_m"]:.2f}m | '
                    f'v2i='
                    f'{("OFF" if not (rsu_packet.get("v2i") or {}).get("latency_ms") else format((rsu_packet.get("v2i") or {}).get("latency_ms"), ".2f") + "ms")}'
                )

            elif rsu_ped is not None:
                print(
                    f'frame={rsu_frame_id} | '
                    f'RSU pedestrian detected | '
                    f'WORLD MATCH FAILED'
                )

            key = 255

            if not validation_config.headless:
                key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if teacher_demo:
                if demo_stop_file.is_file():
                    print("TEACHER DEMO | Operator requested clean shutdown")
                    break

                # Baseline retains the original confirmed-stop completion.
                # A degraded-link run may instead reach the kinematic
                # scenario guard with NO braking trigger. The actor physics
                # is disabled, so this is a measured proximity-limit event,
                # NOT a physical CARLA collision.
                if stop_reported:
                    scenario_end_reason = "controller_confirmed_stop"
                elif rain_demo and motion_guard_triggered:
                    scenario_end_reason = "kinematic_proximity_limit"
                elif (rain_demo and first_sim_time_s is not None
                      and float(onboard_timestamp) - first_sim_time_s >= 16.0):
                    scenario_end_reason = "observation_window_complete"

                if scenario_end_reason is not None:
                    if demo_stop_since is None:
                        demo_stop_since = time.monotonic()
                    if time.monotonic() - demo_stop_since >= demo_hold_s:
                        result = {
                            "completed": True,
                            "reason": scenario_end_reason,
                            "scenario": teacher_scenario.key,
                            "frame_id": onboard_frame_id,
                            "sim_time_s": onboard_timestamp,
                            "first_rsu_detection_s": first_rsu_pedestrian_time_s,
                            "first_fused_detection_s": first_fused_pedestrian_time_s,
                            "first_brake_s": first_brake_time_s,
                            "brake_triggered": bool(brake_started),
                            "controller_confirmed_stop": bool(stop_reported),
                            "minimum_center_distance_m": (
                                None if not math.isfinite(min_true_separation_m)
                                else round(min_true_separation_m, 3)
                            ),
                            "kinematic_motion_guard": bool(motion_guard_triggered),
                            "physical_collision_observed": None,
                            "collision_sensor_available": False,
                            "v2i": degraded_v2i.stats() if rain_demo else {
                                "fault_injected": False,
                                "total_sent": v2i_channel.total_sent,
                                "total_received": v2i_channel.total_received,
                                "total_dropped": v2i_channel.total_dropped,
                            },
                        }
                        # One report per cycle; data is measured, not an
                        # invented success/failure or collision result.
                        (results_dir / "scenario_result.json").write_text(
                            json.dumps(result, indent=2), encoding="utf-8"
                        )
                        demo_cycle_file.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        temporary = demo_cycle_file.with_suffix(".tmp")
                        temporary.write_text(
                            json.dumps(result, indent=2), encoding="utf-8"
                        )
                        temporary.replace(demo_cycle_file)
                        print(
                            "TEACHER DEMO | CYCLE COMPLETE | "
                            f"scenario={teacher_scenario.key} | "
                            f"reason={scenario_end_reason} | "
                            f"minimum_center_distance={result['minimum_center_distance_m']}m",
                            flush=True,
                        )
                        break

            if validation_config.enabled:
                if validation_tracker.reached_time_limit(
                    onboard_timestamp
                ):
                    validation_tracker.mark_timeout()
                    print(
                        "VALIDATION TIME LIMIT | "
                        f"{validation_config.max_sim_seconds:.1f}s"
                    )
                    break

                if (
                    validation_config.auto_exit
                    and (
                        stop_reported
                        or motion_guard_triggered
                    )
                ):
                    print(
                        "VALIDATION AUTO EXIT | "
                        f"mode={validation_config.mode.upper()}"
                    )
                    break

            elapsed = time.perf_counter() - loop_start
            sleep_time = FIXED_DELTA - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping test...")

    finally:
        try:
            metrics_collector.export_summary_json(
                metrics_summary_path
            )
            metrics_collector.export_events_csv(
                metrics_events_path
            )

            if validation_config.enabled:
                validation_summary_path = (
                    results_dir / "validation_summary.json"
                )

                validation_tracker.export_summary_json(
                    validation_summary_path,
                    pipeline_metrics=metrics_collector.summary(),
                )

                print(
                    "VALIDATION SUMMARY SAVED | "
                    f"{validation_summary_path}"
                )
        except Exception as exc:
            print(f"Metrics export warning: {exc}")

        try:
            csv_file.close()
        except Exception:
            pass

        try:
            destroy_tracked_actors(
                client,
                world,
                actors,
            )
        except Exception as exc:
            print(
                "Scenario cleanup verification warning | "
                f"{exc}"
            )

        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        try:
            world.set_weather(original_weather)
        except Exception as exc:
            print(f"CARLA weather restoration warning: {exc}")

        cv2.destroyAllWindows()

        print("Test cleaned up.")


if __name__ == "__main__":
    main()
