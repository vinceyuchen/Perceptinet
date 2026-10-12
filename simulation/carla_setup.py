import carla
import time
import threading
import sys

try:
    import numpy as np
    import cv2
except ImportError:
    print("Missing dependency: numpy and/or opencv-python")
    print("Install with:")
    print("  py -3.12 -m pip install numpy opencv-python")
    sys.exit(1)

HOST = "127.0.0.1"
PORT = 2000
TM_PORT = 8000

SIM_FPS = 20
FIXED_DELTA = 1.0 / SIM_FPS

CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FOV = 90
CAM_TICK = FIXED_DELTA

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360

latest_frames = {
    "onboard": None,
    "rsu": None,
}
frame_counts = {
    "onboard": 0,
    "rsu": 0,
}
frame_lock = threading.Lock()


def carla_image_to_bgr(image):
    """Convert CARLA BGRA image buffer to OpenCV BGR."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3].copy()


def make_camera_callback(name):
    def callback(image):
        frame = carla_image_to_bgr(image)
        with frame_lock:
            latest_frames[name] = frame
            frame_counts[name] += 1
    return callback


def draw_label(frame, text):
    cv2.rectangle(frame, (0, 0), (230, 38), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


client = None
world = None
traffic_manager = None

ego_vehicle = None
onboard_camera = None
rsu_camera = None

original_settings = None

try:
    # --------------------------------------------------
    # Connect to CARLA
    # --------------------------------------------------
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)

    world = client.get_world()

    print("Connected to CARLA")
    print("Map:", world.get_map().name)

    # --------------------------------------------------
    # Configure synchronous real-time simulation
    # --------------------------------------------------
    original_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(TM_PORT)
    traffic_manager.set_synchronous_mode(True)

    print(f"Synchronous mode: {SIM_FPS} Hz")
    print(f"Fixed delta: {FIXED_DELTA:.3f} s")

    # --------------------------------------------------
    # Weather
    # --------------------------------------------------
    world.set_weather(carla.WeatherParameters.ClearNoon)
    print("Weather: ClearNoon")

    # --------------------------------------------------
    # Blueprints
    # --------------------------------------------------
    blueprints = world.get_blueprint_library()

    vehicle_candidates = blueprints.filter("vehicle.tesla.model3")
    if not vehicle_candidates:
        vehicle_candidates = blueprints.filter("vehicle.*")

    if not vehicle_candidates:
        raise RuntimeError("No vehicle blueprint found.")

    vehicle_bp = vehicle_candidates[0]

    camera_bp = blueprints.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(CAM_WIDTH))
    camera_bp.set_attribute("image_size_y", str(CAM_HEIGHT))
    camera_bp.set_attribute("fov", str(CAM_FOV))
    camera_bp.set_attribute("sensor_tick", str(CAM_TICK))

    # --------------------------------------------------
    # Spawn ego vehicle
    # --------------------------------------------------
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No vehicle spawn points found.")

    ego_transform = None

    for transform in spawn_points:
        candidate = world.try_spawn_actor(vehicle_bp, transform)
        if candidate is not None:
            ego_vehicle = candidate
            ego_transform = transform
            break

    if ego_vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle.")

    ego_vehicle.set_autopilot(True, TM_PORT)

    print("Ego vehicle spawned:", ego_vehicle.id)

    # --------------------------------------------------
    # Onboard camera
    # --------------------------------------------------
    onboard_camera_transform = carla.Transform(
        carla.Location(x=1.5, z=1.7),
        carla.Rotation(pitch=-5.0)
    )

    onboard_camera = world.spawn_actor(
        camera_bp,
        onboard_camera_transform,
        attach_to=ego_vehicle
    )

    onboard_camera.listen(make_camera_callback("onboard"))
    print("Onboard camera spawned:", onboard_camera.id)

    # --------------------------------------------------
    # RSU camera
    # --------------------------------------------------
    # Initial RSU position for infrastructure-side perception.
    # We can tune this after choosing the final blind-spot scenario.
    rsu_transform = carla.Transform(
        carla.Location(
            x=ego_transform.location.x + 15.0,
            y=ego_transform.location.y + 8.0,
            z=6.0
        ),
        carla.Rotation(
            pitch=-25.0,
            yaw=180.0,
            roll=0.0
        )
    )

    rsu_camera = world.spawn_actor(
        camera_bp,
        rsu_transform
    )

    rsu_camera.listen(make_camera_callback("rsu"))
    print("RSU camera spawned:", rsu_camera.id)

    # --------------------------------------------------
    # Spectator
    # --------------------------------------------------
    spectator = world.get_spectator()

    print()
    print("Perceptinet simulation running")
    print("Main simulation: 20 Hz real-time pacing")
    print("Onboard camera: 1280x720 @ 20 Hz")
    print("RSU camera: 1280x720 @ 20 Hz")
    print("Press Q in the preview window or Ctrl+C to stop.")
    print()

    last_stats_time = time.perf_counter()
    last_onboard_count = 0
    last_rsu_count = 0

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------
    while True:
        tick_start = time.perf_counter()

        # Advance exactly one simulation step.
        world.tick()

        # Spectator follows behind and above the ego vehicle.
        ego_tf = ego_vehicle.get_transform()
        spectator_tf = carla.Transform(
            ego_tf.transform(carla.Location(x=-7.0, z=4.0)),
            carla.Rotation(
                pitch=-15.0,
                yaw=ego_tf.rotation.yaw,
                roll=0.0
            )
        )
        spectator.set_transform(spectator_tf)

        # Build onboard + RSU camera preview.
        with frame_lock:
            onboard_frame = None if latest_frames["onboard"] is None else latest_frames["onboard"].copy()
            rsu_frame = None if latest_frames["rsu"] is None else latest_frames["rsu"].copy()
            onboard_count = frame_counts["onboard"]
            rsu_count = frame_counts["rsu"]

        if onboard_frame is not None and rsu_frame is not None:
            onboard_preview = cv2.resize(
                onboard_frame,
                (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                interpolation=cv2.INTER_AREA
            )
            rsu_preview = cv2.resize(
                rsu_frame,
                (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            onboard_preview = draw_label(onboard_preview, "ONBOARD CAMERA")
            rsu_preview = draw_label(rsu_preview, "RSU CAMERA")

            combined = np.hstack((onboard_preview, rsu_preview))

            cv2.imshow("Perceptinet - Ego + RSU", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("Preview exit requested.")
                break

        # Print stats once per real-world second.
        now = time.perf_counter()
        if now - last_stats_time >= 1.0:
            onboard_fps = (onboard_count - last_onboard_count) / (now - last_stats_time)
            rsu_fps = (rsu_count - last_rsu_count) / (now - last_stats_time)

            print(
                f"Real-time camera rate | "
                f"Onboard: {onboard_fps:5.1f} FPS | "
                f"RSU: {rsu_fps:5.1f} FPS"
            )

            last_onboard_count = onboard_count
            last_rsu_count = rsu_count
            last_stats_time = now

        # Real-time pacing: one 0.05 s simulation step per ~0.05 s wall-clock time.
        tick_elapsed = time.perf_counter() - tick_start
        sleep_time = FIXED_DELTA - tick_elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nCtrl+C received. Stopping simulation...")

except Exception as exc:
    print(f"\nERROR: {exc}")
    raise

finally:
    print("Cleaning up CARLA actors...")

    for sensor in (onboard_camera, rsu_camera):
        if sensor is not None:
            try:
                sensor.stop()
            except Exception:
                pass

    cv2.destroyAllWindows()

    for actor in (onboard_camera, rsu_camera, ego_vehicle):
        if actor is not None:
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass

    if traffic_manager is not None:
        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass

    if world is not None:
        try:
            if original_settings is not None:
                world.apply_settings(original_settings)
            else:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
        except Exception:
            pass

    print("Cleanup complete.")
