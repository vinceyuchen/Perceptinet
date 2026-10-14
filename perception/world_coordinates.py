import math
from typing import Dict, List, Optional, Tuple

import carla
import numpy as np


def build_camera_intrinsic(width: int, height: int, fov_degrees: float) -> np.ndarray:
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))

    k = np.identity(3, dtype=np.float64)
    k[0, 0] = focal
    k[1, 1] = focal
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0

    return k


def world_to_camera_matrix(camera: carla.Actor) -> np.ndarray:
    return np.array(
        camera.get_transform().get_inverse_matrix(),
        dtype=np.float64,
    )


def project_world_location(
    location: carla.Location,
    camera: carla.Actor,
    intrinsic: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    point = np.array(
        [location.x, location.y, location.z, 1.0],
        dtype=np.float64,
    )

    camera_point = world_to_camera_matrix(camera) @ point

    # CARLA/UE -> conventional camera coordinates
    x_cam = camera_point[1]
    y_cam = -camera_point[2]
    z_cam = camera_point[0]

    if z_cam <= 0.05:
        return None

    image_point = intrinsic @ np.array(
        [x_cam, y_cam, z_cam],
        dtype=np.float64,
    )

    u = image_point[0] / image_point[2]
    v = image_point[1] / image_point[2]

    return float(u), float(v), float(z_cam)


def actor_matches_perceptinet_class(actor: carla.Actor, class_name: str) -> bool:
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
            for token in ("truck", "carlacola", "hgv", "firetruck")
        )

    if class_name == "car":
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


def bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def bbox_center_distance(a: List[float], b: List[float]) -> float:
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0

    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0

    return math.hypot(acx - bcx, acy - bcy)


def project_actor_bbox(
    actor: carla.Actor,
    camera: carla.Actor,
    intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
) -> Optional[Dict]:
    """
    Project the full CARLA 3D actor bounding box into the camera image.

    This is substantially more robust than projecting only one actor-centre
    point, especially for pedestrians and high-angle RSU cameras.
    """
    try:
        vertices = actor.bounding_box.get_world_vertices(
            actor.get_transform()
        )
    except Exception:
        return None

    projected = []

    for vertex in vertices:
        p = project_world_location(vertex, camera, intrinsic)

        if p is None:
            continue

        u, v, depth = p
        projected.append((u, v, depth))

    # Need enough visible vertices to form a meaningful image-space box.
    if len(projected) < 4:
        return None

    us = [p[0] for p in projected]
    vs = [p[1] for p in projected]
    depths = [p[2] for p in projected]

    x1 = min(us)
    y1 = min(vs)
    x2 = max(us)
    y2 = max(vs)

    # Reject actors whose projected box is entirely outside the image.
    if x2 < 0 or y2 < 0 or x1 >= image_width or y1 >= image_height:
        return None

    # Clamp only for matching/UI purposes.
    clipped_bbox = [
        max(0.0, x1),
        max(0.0, y1),
        min(float(image_width - 1), x2),
        min(float(image_height - 1), y2),
    ]

    return {
        "bbox": clipped_bbox,
        "depth_m": float(sum(depths) / len(depths)),
    }


def match_detection_to_actor(
    world: carla.World,
    camera: carla.Actor,
    intrinsic: np.ndarray,
    detection: Dict,
    image_width: int,
    image_height: int,
) -> Optional[Dict]:
    """
    Match YOLO 2D bbox to a CARLA actor using the projected 3D actor bbox.

    Primary score:
        highest IoU between YOLO bbox and projected CARLA bbox

    Fallback:
        nearest bbox centre when IoU is small but the projected box is
        still spatially close.

    This is a V1 ground-truth association method for getting the fusion
    pipeline working. It is not the final real-world localization method.
    """
    class_name = detection["class_name"]
    yolo_bbox = detection["bbox"]

    candidates = []

    for actor in candidate_actors_for_class(world, class_name):
        projection = project_actor_bbox(
            actor,
            camera,
            intrinsic,
            image_width,
            image_height,
        )

        if projection is None:
            continue

        projected_bbox = projection["bbox"]

        iou = bbox_iou(yolo_bbox, projected_bbox)
        center_distance = bbox_center_distance(
            yolo_bbox,
            projected_bbox,
        )

        candidates.append(
            {
                "actor": actor,
                "projection": projection,
                "iou": iou,
                "center_distance": center_distance,
            }
        )

    if not candidates:
        return None

    # Prefer actual bbox overlap.
    candidates.sort(
        key=lambda item: (
            item["iou"],
            -item["center_distance"],
        ),
        reverse=True,
    )

    best = candidates[0]


    # Practical V1 gate:
    # accept reasonable overlap, otherwise accept a close projected bbox.
    if best["iou"] < 0.02 and best["center_distance"] > 120.0:
        return None

    actor = best["actor"]
    transform = actor.get_transform()
    velocity = actor.get_velocity()

    return {
        "actor_id": int(actor.id),
        "actor_type": actor.type_id,
        "match_iou": float(best["iou"]),
        "match_center_distance_px": float(best["center_distance"]),
        "projected_actor_bbox": [
            float(v) for v in best["projection"]["bbox"]
        ],
        "camera_depth_m": float(best["projection"]["depth_m"]),
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


def enrich_detection_packet(
    world: carla.World,
    camera: carla.Actor,
    packet: Dict,
    image_width: int,
    image_height: int,
    camera_fov: float,
) -> Dict:
    intrinsic = build_camera_intrinsic(
        image_width,
        image_height,
        camera_fov,
    )

    enriched_detections = []

    for detection in packet.get("detections", []):
        enriched = dict(detection)

        enriched["world_match"] = match_detection_to_actor(
            world,
            camera,
            intrinsic,
            detection,
            image_width,
            image_height,
        )

        enriched_detections.append(enriched)

    output = dict(packet)
    output["detections"] = enriched_detections

    return output
