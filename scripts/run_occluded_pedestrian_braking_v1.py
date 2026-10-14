import sys
import csv
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
from fusion.fusion_engine import FusionEngine
from risk.risk_engine import RiskEngine, RiskConfig
from control.braking_controller import BrakingController, BrakingConfig


# Deterministic kinematic Ego motion for Risk Transition v1.
# Physics remains disabled in this stage.
EGO_TEST_SPEED_MPS = 4.0
EGO_STOP_DISTANCE_M = 0.5



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

OCCLUDER_FORWARD_DISTANCES = [12.0, 15.0, 18.0, 21.0]
PEDESTRIAN_FORWARD_FROM_OCCLUDER = 8.0
PEDESTRIAN_LATERAL_OFFSETS = [0.7, -0.7, 1.2, -1.2]

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


def spawn_first_available(world, blueprints, transforms, label):
    for blueprint in blueprints:
        for transform in transforms:
            actor = world.try_spawn_actor(blueprint, transform)

            if actor is not None:
                print(
                    f"{label} spawned: {actor.id} "
                    f"({blueprint.id}) at "
                    f"x={transform.location.x:.1f}, "
                    f"y={transform.location.y:.1f}, "
                    f"z={transform.location.z:.1f}"
                )
                return actor, transform

    raise RuntimeError(f"Failed to spawn {label} after all fallback attempts")


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

    world = client.get_world()

    print("Connected to CARLA")
    print("Map:", world.get_map().name)

    original_settings = world.get_settings()
    actors = []
    traffic_manager = None

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "occluded_pedestrian_world_v1.csv"

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

    print("V2I channel: 25 ms base latency +/- 5 ms jitter")

    fusion_engine = FusionEngine(
        fusion_distance_m=2.0
    )

    print("Fusion engine enabled")

    risk_engine = RiskEngine(
        RiskConfig(
            lane_half_width_m=2.5,
            caution_ttc_s=5.0,
            warning_ttc_s=3.0,
            critical_ttc_s=1.5,
        )
    )

    print("Risk engine enabled")

    braking_controller = BrakingController(
        BrakingConfig(
            caution_decel_mps2=0.75,
            warning_decel_mps2=2.0,
            critical_decel_mps2=6.0,
        )
    )

    # Persistent kinematic speed command.
    ego_command_speed_mps = EGO_TEST_SPEED_MPS

    # Conservative v1 hazard latch.
    # A one-frame detector dropout must not cancel braking.
    latched_pedestrian_hazard = "NORMAL"

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

    print(
        f"Automatic braking Ego cruise speed: "
        f"{EGO_TEST_SPEED_MPS:.1f} m/s"
    )
    print(
        "Braking controller: "
        "CAUTION=0.75, WARNING=2.0, CRITICAL=6.0 m/s^2"
    )
    print("World-coordinate association enabled\nV2I channel enabled")

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

        world.set_weather(carla.WeatherParameters.ClearNoon)

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

        ego_vehicle = None
        ego_transform = None

        for transform in spawn_points:
            actor = world.try_spawn_actor(ego_candidates[0], transform)

            if actor is not None:
                ego_vehicle = actor
                ego_transform = transform
                break

        if ego_vehicle is None:
            raise RuntimeError("Failed to spawn Ego Vehicle")

        actors.append(ego_vehicle)
        ego_vehicle.set_simulate_physics(False)

        print(f"Ego Vehicle spawned: {ego_vehicle.id}")

        occluder_transforms = []

        for distance in OCCLUDER_FORWARD_DISTANCES:
            loc = relative_location(
                ego_transform,
                forward=distance,
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

            # Final hard failsafe only.
            # Normal stopping should happen through Risk +
            # BrakingController well before this distance.
            if ego_to_pedestrian_m <= EGO_STOP_DISTANCE_M:
                ego_command_speed_mps = 0.0
                ego_speed_this_tick_mps = 0.0

            if ego_speed_this_tick_mps > 0.0:
                step_m = (
                    ego_speed_this_tick_mps
                    * fixed_delta
                )

                forward = ego_now.get_forward_vector()

                next_ego_transform = carla.Transform(
                    carla.Location(
                        x=(
                            ego_now.location.x
                            + forward.x * step_m
                        ),
                        y=(
                            ego_now.location.y
                            + forward.y * step_m
                        ),
                        z=ego_now.location.z,
                    ),
                    ego_now.rotation,
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

            # ------------------------------------------------
            # V2I v1
            # RSU packet must cross the simulated V2I channel
            # before the Ego/Fusion side can consume it.
            # ------------------------------------------------

            received_rsu_packet = v2i_channel.transmit(
                rsu_packet,
                source_id="RSU-01",
            )

            if received_rsu_packet is None:
                # V1 currently uses 0% packet loss, but keep
                # this guard for future network experiments.
                continue

            # Everything downstream now sees the packet
            # AFTER V2I transmission.
            rsu_packet = received_rsu_packet

            # ------------------------------------------------
            # Fusion v1
            #
            # Onboard packet:
            #   local Ego Vehicle perception
            #
            # RSU packet:
            #   remote perception AFTER V2I transmission
            #
            # Output:
            #   unified world model
            # ------------------------------------------------

            fused_world = fusion_engine.fuse(
                [
                    onboard_packet,
                    rsu_packet,
                ]
            )

            fused_pedestrian = next(
                (
                    obj
                    for obj in fused_world["objects"]
                    if obj["class_name"] == "pedestrian"
                ),
                None,
            )

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
                "yaw_deg": float(ego_transform.rotation.yaw),
                "speed_mps": float(ego_speed_mps),
            }

            risk_result = risk_engine.assess_world(
                ego_state,
                fused_world,
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

                # Keep the highest pedestrian hazard observed.
                if (
                    hazard_priority[
                        observed_pedestrian_hazard
                    ]
                    >
                    hazard_priority[
                        latched_pedestrian_hazard
                    ]
                ):
                    print(
                        "HAZARD TRANSITION | "
                        f"{latched_pedestrian_hazard} -> "
                        f"{observed_pedestrian_hazard}"
                    )

                    latched_pedestrian_hazard = (
                        observed_pedestrian_hazard
                    )

            braking_command = braking_controller.update(
                current_speed_mps=ego_speed_this_tick_mps,
                hazard_level=latched_pedestrian_hazard,
                dt_s=fixed_delta,
            )

            # This becomes the speed for the NEXT CARLA tick.
            ego_command_speed_mps = braking_command[
                "target_speed_mps"
            ]

            if (
                braking_command["braking_active"]
                and not brake_started
            ):
                brake_started = True

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
                    f"hazard={latched_pedestrian_hazard} | "
                    f"speed={brake_start_speed_mps:.2f}m/s | "
                    f"distance="
                    f"{brake_start_distance_m:.2f}m | "
                    f"TTC="
                    f"{brake_start_ttc_s:.2f}s"
                    if brake_start_ttc_s is not None
                    else
                    "BRAKE TRIGGERED | TTC=N/A"
                )

            if braking_command["braking_active"]:
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
                    f'hazard={latched_pedestrian_hazard} | '
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
                    f"hazard={latched_pedestrian_hazard}"
                )

                stop_reported = True

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
                    f'v2i={rsu_packet["v2i"]["latency_ms"]:.2f}ms'
                )

            elif rsu_ped is not None:
                print(
                    f'frame={rsu_frame_id} | '
                    f'RSU pedestrian detected | '
                    f'WORLD MATCH FAILED'
                )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elapsed = time.perf_counter() - loop_start
            sleep_time = FIXED_DELTA - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping test...")

    finally:
        try:
            csv_file.close()
        except Exception:
            pass

        for actor in reversed(actors):
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass

        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        cv2.destroyAllWindows()

        print("Test cleaned up.")


if __name__ == "__main__":
    main()
