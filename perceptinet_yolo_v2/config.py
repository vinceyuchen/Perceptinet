# Perceptinet YOLO v2 configuration
# Road-user-only perception pipeline

CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
TRAFFIC_MANAGER_PORT = 8000

SIM_FPS = 20
FIXED_DELTA_SECONDS = 1.0 / SIM_FPS

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOV = 90
CAMERA_TICK = FIXED_DELTA_SECONDS

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360

YOLO_MODEL_PATH = r"C:\CARLA\yolo26n.pt"
YOLO_DEVICE = 0
YOLO_IMAGE_SIZE = 640
YOLO_CONFIDENCE = 0.25

# COCO class IDs used by Perceptinet:
# 0 person      -> renamed to pedestrian in Perceptinet
# 1 bicycle
# 2 car
# 3 motorcycle
# 5 bus
# 7 truck
TARGET_CLASS_IDS = [0, 1, 2, 3, 5, 7]

CLASS_NAME_OVERRIDES = {
    "person": "pedestrian",
}
