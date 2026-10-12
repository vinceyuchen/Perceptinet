import carla
import cv2
import numpy as np
import threading
import time
import math

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

# Relative placement
OCCLUDER_FORWARD_DISTANCES = [12.0, 15.0, 18.0, 21.0]
PEDESTRIAN_FORWARD_FROM_OCCLUDER = 8.0

# Start slightly closer to the occluder's centreline than v2.
PEDESTRIAN_LATERAL_OFFSETS = [0.7, -0.7, 1.2, -1.2]

# RSU placement relative to Ego
RSU_FORWARD = 20.0
RSU_LATERAL = -10.0
RSU_HEIGHT = 7.0

latest_frames = {
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


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)

    world = client.get_world()
    print("Connected to CARLA")
    print("Map:", world.get_map().name)

    original_settings = world.get_settings()
    actors = []
    traffic_manager = None

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
        if not ego_candidates:
            raise RuntimeError("No ego vehicle blueprint found")

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
        if not occluder_candidates:
            raise RuntimeError("No occluding vehicle blueprint found")

        walker_candidates = find_blueprints(
            blueprints,
            ["walker.pedestrian.*"],
        )
        if not walker_candidates:
            raise RuntimeError("No pedestrian blueprint found")

        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))
        camera_bp.set_attribute("sensor_tick", str(CAMERA_TICK))

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No map spawn points found")

        # Ego Vehicle: first valid official spawn point
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

        print(
            f"Ego Vehicle spawned: {ego_vehicle.id} at "
            f"x={ego_transform.location.x:.1f}, "
            f"y={ego_transform.location.y:.1f}"
        )

        # Occluding vehicle
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

        # Pedestrian just beyond occluder, close to its centreline
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

        # Onboard Camera
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

        print(f"Onboard Camera spawned: {onboard_camera.id}")

        # RSU camera aimed automatically at pedestrian
        rsu_location = relative_location(
            ego_transform,
            forward=RSU_FORWARD,
            lateral=RSU_LATERAL,
            vertical=RSU_HEIGHT,
        )

        direction = pedestrian_transform.location - rsu_location

        rsu_yaw = math.degrees(
            math.atan2(direction.y, direction.x)
        )

        horizontal = math.sqrt(
            direction.x ** 2 + direction.y ** 2
        )

        rsu_pitch = math.degrees(
            math.atan2(direction.z, horizontal)
        )

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

        print(f"RSU-01 Camera spawned: {rsu_camera.id}")

        # Spectator overview
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
        print("Occluded Pedestrian Scenario v3 running")
        print("Two live previews are enabled:")
        print("  ONBOARD CAMERA")
        print("  RSU-01 CAMERA")
        print()
        print("Goal:")
        print("  Onboard -> pedestrian blocked by occluding vehicle")
        print("  RSU-01  -> pedestrian clearly visible")
        print()
        print("Press Q / Esc in preview window or Ctrl+C to stop.")

        while True:
            loop_start = time.perf_counter()

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

            if onboard_frame is not None and rsu_frame is not None:
                onboard_view = cv2.resize(
                    onboard_frame,
                    (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )

                rsu_view = cv2.resize(
                    rsu_frame,
                    (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )

                add_header(onboard_view, "ONBOARD CAMERA")
                add_header(rsu_view, "RSU-01 CAMERA")

                combined = np.hstack(
                    (onboard_view, rsu_view)
                )

                cv2.imshow(
                    "Perceptinet - Occluded Pedestrian Scenario",
                    combined,
                )

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            elapsed = time.perf_counter() - loop_start
            sleep_time = FIXED_DELTA - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping scenario...")

    finally:
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

        print("Scenario cleaned up.")


if __name__ == "__main__":
    main()
