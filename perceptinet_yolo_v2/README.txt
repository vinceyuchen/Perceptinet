PERCEPTINET YOLO V2
====================

Purpose
-------
Road-user-only YOLO26n perception for Perceptinet.

Detected classes
----------------
Perceptinet keeps only:

- pedestrian
- bicycle
- motorcycle
- car
- bus
- truck

Ultralytics/COCO internally calls class 0 "person".
Perceptinet renames that class to "pedestrian" everywhere in its own output.

Ignored examples
----------------
Traffic lights, stop signs, benches, chairs and other unrelated COCO
objects are not returned by this perception pipeline.

Logical architecture
--------------------
Onboard Camera -> onboard node -> YOLO26n
RSU Camera     -> rsu_01 node  -> YOLO26n

Both logical nodes currently share one YOLO26n model on the NVIDIA L4.
This is intentional for the prototype and avoids loading duplicate
weights into GPU memory.

The output packet format remains ready for later V2I/fusion work:

{
    "source": "onboard" or "rsu_01",
    "frame_id": ...,
    "timestamp": ...,
    "detections": [
        {
            "class_id": ...,
            "class_name": "pedestrian" | "car" | ...,
            "confidence": ...,
            "bbox": [x1, y1, x2, y2]
        }
    ],
    "preprocess_ms": ...,
    "inference_ms": ...,
    "postprocess_ms": ...,
    "batch_wall_ms": ...
}

Run
---
1. Start CARLA:

   C:\CARLA\CarlaUE4.exe -quality-level=Epic -ResX=1920 -ResY=1080 -windowed -carla-rpc-port=2000

2. Extract this folder to:

   C:\CARLA\perceptinet_yolo_v2

3. Run:

   cd C:\CARLA\perceptinet_yolo_v2
   py -3.12 main.py

Exit with Q, Esc or Ctrl+C.

Next milestone
--------------
Build the first standard Occluded Pedestrian Scenario:

- pedestrian visible to RSU
- pedestrian occluded from Onboard Camera
- RSU detection succeeds
- Onboard detection does not

After that:
coordinate transformation -> V2I -> fusion -> unified world model.
