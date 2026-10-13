import math
import cv2
import numpy as np


def build_camera_matrix(width, height, fov):
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))

    k = np.identity(3)
    k[0, 0] = focal
    k[1, 1] = focal
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0

    return k


def get_world_to_camera(camera):
    transform = camera.get_transform()
    return np.array(transform.get_inverse_matrix())


def project_world_point(location, world_to_camera, k):
    point = np.array([
        location.x,
        location.y,
        location.z,
        1.0
    ])

    point_camera = world_to_camera @ point

    # CARLA coordinates:
    # x = forward
    # y = right
    # z = up
    #
    # Camera projection:
    # x_img <- CARLA y
    # y_img <- -CARLA z
    # depth <- CARLA x
    camera_coords = np.array([
        point_camera[1],
        -point_camera[2],
        point_camera[0]
    ])

    depth = camera_coords[2]

    if depth <= 0.05:
        return None

    image_point = k @ camera_coords

    x = image_point[0] / image_point[2]
    y = image_point[1] / image_point[2]

    return float(x), float(y), float(depth)


def project_actor_bbox(actor, camera, width, height, fov):
    k = build_camera_matrix(width, height, fov)
    world_to_camera = get_world_to_camera(camera)

    vertices = actor.bounding_box.get_world_vertices(
        actor.get_transform()
    )

    projected = []

    for vertex in vertices:
        point = project_world_point(
            vertex,
            world_to_camera,
            k
        )

        if point is not None:
            projected.append(point)

    if len(projected) < 2:
        return None

    xs = [p[0] for p in projected]
    ys = [p[1] for p in projected]
    depths = [p[2] for p in projected]

    x1 = max(0, min(width - 1, min(xs)))
    y1 = max(0, min(height - 1, min(ys)))
    x2 = max(0, min(width - 1, max(xs)))
    y2 = max(0, min(height - 1, max(ys)))

    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "bbox": (
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        ),
        "depth": float(np.mean(depths)),
        "actor_id": actor.id,
        "location": actor.get_location()
    }


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def bbox_center_distance(a, b):
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0

    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0

    return math.sqrt(
        (ax - bx) ** 2 +
        (ay - by) ** 2
    )


def debug_walkers(frame, world, camera):
    """
    Draw every CARLA walker projected into the camera.

    Green rectangle = CARLA ground-truth projected bounding box.
    """

    height, width = frame.shape[:2]

    try:
        fov = float(camera.attributes["fov"])
    except Exception:
        fov = 90.0

    walkers = world.get_actors().filter("walker.pedestrian.*")

    results = []

    for actor in walkers:
        projected = project_actor_bbox(
            actor,
            camera,
            width,
            height,
            fov
        )

        if projected is None:
            continue

        x1, y1, x2, y2 = projected["bbox"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        location = projected["location"]

        text = (
            f"CARLA WALKER ID={actor.id} "
            f"({location.x:.1f}, "
            f"{location.y:.1f}, "
            f"{location.z:.1f})"
        )

        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        results.append(projected)

    return results


def compare_yolo_to_walkers(yolo_bbox, projected_walkers):
    """
    Compare one YOLO pedestrian bbox against every
    projected CARLA pedestrian bbox.
    """

    comparisons = []

    for walker in projected_walkers:
        carla_bbox = walker["bbox"]

        iou = bbox_iou(
            yolo_bbox,
            carla_bbox
        )

        distance = bbox_center_distance(
            yolo_bbox,
            carla_bbox
        )

        comparisons.append({
            "actor_id": walker["actor_id"],
            "carla_bbox": carla_bbox,
            "iou": iou,
            "center_distance": distance,
            "location": walker["location"]
        })

    comparisons.sort(
        key=lambda x: (
            -x["iou"],
            x["center_distance"]
        )
    )

    return comparisons