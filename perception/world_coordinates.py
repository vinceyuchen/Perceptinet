import math
from typing import Dict, List, Optional, Tuple

import carla
import numpy as np


# Perceptinet naming -> CARLA actor families
CARLA_CLASS_FILTERS = {
    "pedestrian": ["walker.pedestrian.*"],
    "bicycle": ["vehicle.*"],
    "motorcycle": ["vehicle.*"],
    "car": ["vehicle.*"],
    "bus": ["vehicle.*"],
    "truck": ["vehicle.*"],
}


def build_camera_intrinsic(width: int, height: int, fov_degrees: float) -> np.ndarray:
    """
    Build the 3x3 pinhole-camera intrinsic matrix used by CARLA RGB cameras.
    """
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))

    k = np.identity(3, dtype=np.float64)
    k[0, 0] = focal
    k[1, 1] = focal
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0

    return k


def world_to_camera_matrix(camera: carla.Actor) -> np.ndarray:
    """
    Return the 4x4 world-to-camera transform matrix.
    """
    return np.array(camera.get_transform().get_inverse_matrix(), dtype=np.float64)


def project_world_location(
    location: carla.Location,
    camera: carla.Actor,
    intrinsic: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """
    Project a CARLA world location into camera image coordinates.

    Returns:
        (u, v, depth) when the point is in front of the camera.
        None when it is behind the camera.
    """
    point = np.array(
        [location.x, location.y, location.z, 1.0],
        dtype=np.float64,
    )

    world_to_camera = world_to_camera_matrix(camera)
    camera_point = world_to_camera @ point

    # CARLA/UE coordinate convention -> standard camera convention
    # UE:      x forward, y right, z up
    # Camera:  x right, y down, z forward
    x_cam = camera_point[1]
    y_cam = -camera_point[2]
    z_cam = camera_point[0]

    if z_cam <= 0.0:
        return None

    image_point = intrinsic @ np.array(
        [x_cam, y_cam, z_cam],
        dtype=np.float64,
    )

    u = image_point[0] / image_point[2]
    v = image_point[1] / image_point[2]

    return float(u), float(v), float(z_cam)


def actor_reference_location(actor: carla.Actor) -> carla.Location:
    """
    Use a point near the visual centre of the actor rather than its origin.
    """
    transform = actor.get_transform()
    location = transform.location

    try:
        extent = actor.bounding_box.extent
        z_offset = max(0.5, float(extent.z))
    except Exception:
        z_offset = 0.8

    return carla.Location(
        x=location.x,
        y=location.y,
        z=location.z + z_offset,
    )


def bbox_contains_pixel(
    bbox: List[float],
    u: float,
    v: float,
    margin: float = 10.0,
) -> bool:
    x1, y1, x2, y2 = bbox

    return (
        x1 - margin <= u <= x2 + margin
        and y1 - margin <= v <= y2 + margin
    )


def bbox_center_distance(
    bbox: List[float],
    u: float,
    v: float,
) -> float:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    return math.hypot(u - cx, v - cy)


def actor_matches_perceptinet_class(actor: carla.Actor, class_name: str) -> bool:
    """
    Coarse actor-type mapping for V1.

    Pedestrian mapping is exact.
    Vehicle subclass mapping is intentionally conservative and based on
    CARLA blueprint/type strings. It can be refined later.
    """
    type_id = actor.type_id.lower()

    if class_name == "pedestrian":
        return type_id.startswith("walker.pedestrian.")

    if not type_id.startswith("vehicle."):
        return False

    if class_name == "bicycle":
        return any(
            token in type_id
            for token in ("bike", "bicycle", "diamondback", "gazelle")
        )

    if class_name == "motorcycle":
        return any(
            token in type_id
            for token in ("motorcycle", "yamaha", "kawasaki", "harley", "vespa")
        )

    if class_name == "bus":
        return "bus" in type_id

    if class_name == "truck":
        return any(
            token in type_id
            for token in ("truck", "carlacola", "hgv", "firetruck", "ambulance")
        )

    if class_name == "car":
        # Exclude obvious non-car vehicle families where possible.
        excluded = (
            "bus",
            "truck",
            "carlacola",
            "hgv",
            "motorcycle",
            "yamaha",
            "kawasaki",
            "harley",
            "vespa",
            "bike",
            "bicycle",
            "diamondback",
            "gazelle",
        )
        return not any(token in type_id for token in excluded)

    return False


def candidate_actors_for_class(
    world: carla.World,
    class_name: str,
) -> List[carla.Actor]:
    actors = world.get_actors()

    if class_name == "pedestrian":
        return list(actors.filter("walker.pedestrian.*"))

    if class_name in {"bicycle", "motorcycle", "car", "bus", "truck"}:
        return [
            actor
            for actor in actors.filter("vehicle.*")
            if actor_matches_perceptinet_class(actor, class_name)
        ]

    return []


def match_detection_to_actor(
    world: carla.World,
    camera: carla.Actor,
    intrinsic: np.ndarray,
    detection: Dict,
) -> Optional[Dict]:
    """
    Associate one YOLO detection with a CARLA ground-truth actor.

    V1 strategy:
      1. find CARLA actors of the same Perceptinet class
      2. project each actor reference point into the image
      3. keep candidates whose projected point lies inside the YOLO bbox
      4. choose the candidate closest to the YOLO bbox centre

    This intentionally uses CARLA ground truth only for the first fusion
    prototype. Later versions can replace this with camera/depth geometry.
    """
    class_name = detection["class_name"]
    bbox = detection["bbox"]

    best = None

    for actor in candidate_actors_for_class(world, class_name):
        reference = actor_reference_location(actor)
        projected = project_world_location(reference, camera, intrinsic)

        if projected is None:
            continue

        u, v, depth = projected

        if not bbox_contains_pixel(bbox, u, v):
            continue

        score = bbox_center_distance(bbox, u, v)

        if best is None or score < best["match_score"]:
            transform = actor.get_transform()
            velocity = actor.get_velocity()

            best = {
                "actor_id": int(actor.id),
                "actor_type": actor.type_id,
                "match_score": float(score),
                "camera_depth_m": float(depth),
                "projected_pixel": {
                    "u": float(u),
                    "v": float(v),
                },
                "world_position": {
                    "x": float(transform.location.x),
                    "y": float(transform.location.y),
                    "z": float(transform.location.z),
                },
                "world_velocity": {
                    "x": float(velocity.x),
                    "y": float(velocity.y),
                    "z": float(velocity.z),
                },
            }

    return best


def enrich_detection_packet(
    world: carla.World,
    camera: carla.Actor,
    packet: Dict,
    image_width: int,
    image_height: int,
    camera_fov: float,
) -> Dict:
    """
    Add world-coordinate information to every detection in a packet.

    Expected packet input:
    {
        "source": "onboard" | "rsu_01",
        "frame_id": ...,
        "timestamp": ...,
        "detections": [...]
    }
    """
    intrinsic = build_camera_intrinsic(
        image_width,
        image_height,
        camera_fov,
    )

    enriched_detections = []

    for detection in packet.get("detections", []):
        enriched = dict(detection)

        actor_match = match_detection_to_actor(
            world,
            camera,
            intrinsic,
            detection,
        )

        enriched["world_match"] = actor_match
        enriched_detections.append(enriched)

    output = dict(packet)
    output["detections"] = enriched_detections

    return output
