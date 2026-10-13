import sys
import time

import carla
import cv2
import numpy as np

from config import (
    CARLA_HOST,
    CARLA_PORT,
    TRAFFIC_MANAGER_PORT,
    SIM_FPS,
    FIXED_DELTA_SECONDS,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FOV,
    CAMERA_TICK,
    PREVIEW_WIDTH,
    PREVIEW_HEIGHT,
)
from nodes import PerceptionNode
from perception import YOLO26Perception


def carla_image_to_bgr(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3].copy()


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


def resize_preview(frame):
    return cv2.resize(
        frame,
        (PREVIEW_WIDTH, PREVIEW_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def main():
    client = None
    world = None
    traffic_manager = None
    original_settings = None

    ego_vehicle = None
    onboard_camera = None
    rsu_camera = None

    onboard_node = PerceptionNode("onboard")
    rsu_node = PerceptionNode("rsu_01")

    try:
        print("Connecting to CARLA...")
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(20.0)
        world = client.get_world()

        print("Connected to CARLA")
        print("Map:", world.get_map().name)

        original_settings = world.get_settings()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
        traffic_manager.set_synchronous_mode(True)

        world.set_weather(carla.WeatherParameters.ClearNoon)

        blueprints = world.get_blueprint_library()

        vehicle_candidates = blueprints.filter("vehicle.tesla.model3")
        if not vehicle_candidates:
            vehicle_candidates = blueprints.filter("vehicle.*")

        if not vehicle_candidates:
            raise RuntimeError("No vehicle blueprint found.")

        vehicle_bp = vehicle_candidates[0]

        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))
        camera_bp.set_attribute("sensor_tick", str(CAMERA_TICK))

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

        ego_vehicle.set_autopilot(True, TRAFFIC_MANAGER_PORT)
        print("Ego Vehicle:", ego_vehicle.id)

        onboard_transform = carla.Transform(
            carla.Location(x=1.5, z=1.7),
            carla.Rotation(pitch=-5.0),
        )

        onboard_camera = world.spawn_actor(
            camera_bp,
            onboard_transform,
            attach_to=ego_vehicle,
        )
        onboard_camera.listen(
            lambda image: onboard_node.ingest_carla_image(
                image, carla_image_to_bgr
            )
        )
        print("Onboard Camera:", onboard_camera.id)

        rsu_transform = carla.Transform(
            carla.Location(
                x=ego_transform.location.x + 15.0,
                y=ego_transform.location.y + 8.0,
                z=6.0,
            ),
            carla.Rotation(
                pitch=-25.0,
                yaw=180.0,
                roll=0.0,
            ),
        )

        rsu_camera = world.spawn_actor(camera_bp, rsu_transform)
        rsu_camera.listen(
            lambda image: rsu_node.ingest_carla_image(
                image, carla_image_to_bgr
            )
        )
        print("RSU Camera (rsu_01):", rsu_camera.id)

        detector = YOLO26Perception()

        spectator = world.get_spectator()

        print()
        print("Perceptinet YOLO v1 running")
        print("Logical perception nodes:")
        print("  onboard  -> YOLO26n")
        print("  rsu_01   -> YOLO26n")
        print("Press Q / Esc in preview window, or Ctrl+C, to stop.")
        print()

        last_stats = time.perf_counter()
        processed_batches = 0
        last_batch_wall_ms = 0.0

        latest_annotated = {
            "onboard": None,
            "rsu_01": None,
        }

        while True:
            loop_start = time.perf_counter()

            world.tick()

            ego_tf = ego_vehicle.get_transform()
            spectator.set_transform(
                carla.Transform(
                    ego_tf.transform(carla.Location(x=-7.0, z=4.0)),
                    carla.Rotation(
                        pitch=-15.0,
                        yaw=ego_tf.rotation.yaw,
                        roll=0.0,
                    ),
                )
            )

            samples = []

            onboard_sample = onboard_node.get_new_frame()
            if onboard_sample is not None:
                samples.append(onboard_sample)

            rsu_sample = rsu_node.get_new_frame()
            if rsu_sample is not None:
                samples.append(rsu_sample)

            if samples:
                packets, annotated, batch_wall_ms = detector.detect_batch(samples)
                processed_batches += 1
                last_batch_wall_ms = batch_wall_ms
                latest_annotated.update(annotated)

                for packet in packets:
                    if packet["detections"]:
                        short_detections = ", ".join(
                            f'{d["class_name"]} {d["confidence"]:.2f}'
                            for d in packet["detections"][:4]
                        )
                    else:
                        short_detections = "none"

                    print(
                        f'[{packet["source"]}] '
                        f'frame={packet["frame_id"]} | '
                        f'objects={len(packet["detections"])} | '
                        f'infer={packet["inference_ms"]:.1f} ms | '
                        f'{short_detections}'
                    )

            if (
                latest_annotated["onboard"] is not None
                and latest_annotated["rsu_01"] is not None
            ):
                onboard_view = resize_preview(
                    latest_annotated["onboard"].copy()
                )
                rsu_view = resize_preview(
                    latest_annotated["rsu_01"].copy()
                )

                add_header(onboard_view, "ONBOARD PERCEPTION")
                add_header(rsu_view, "RSU-01 PERCEPTION")

                combined = np.hstack((onboard_view, rsu_view))
                cv2.imshow("Perceptinet - YOLO26n Perception", combined)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            now = time.perf_counter()
            if now - last_stats >= 1.0:
                elapsed = now - last_stats
                batch_rate = processed_batches / elapsed

                print(
                    f"STATUS | target={SIM_FPS} Hz | "
                    f"YOLO batch rate={batch_rate:.1f}/s | "
                    f"last batch wall={last_batch_wall_ms:.1f} ms"
                )

                processed_batches = 0
                last_stats = now

            # Keep wall-clock pacing near the requested simulation frequency
            # when inference/rendering is faster than the 20 Hz budget.
            loop_elapsed = time.perf_counter() - loop_start
            sleep_time = FIXED_DELTA_SECONDS - loop_elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nCtrl+C received.")

    finally:
        print("Cleaning up...")

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

        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass

        print("Cleanup complete.")


if __name__ == "__main__":
    main()
