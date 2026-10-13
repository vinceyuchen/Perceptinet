import time
from pathlib import Path

from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH,
    YOLO_DEVICE,
    YOLO_IMAGE_SIZE,
    YOLO_CONFIDENCE,
    YOLO_HALF,
)


class YOLO26Perception:
    """
    Shared YOLO26n perception engine.

    It accepts frames from logical perception nodes such as:
      - onboard
      - rsu_01
      - rsu_02
      - ...

    It returns a standardized detection packet for every source.
    """

    def __init__(self):
        model_path = Path(YOLO_MODEL_PATH)

        if model_path.exists():
            load_path = str(model_path)
        else:
            # Ultralytics will resolve/download the official pretrained model
            # if it is not already present at C:\CARLA\yolo26n.pt.
            load_path = "yolo26n.pt"

        print(f"Loading YOLO26n: {load_path}")
        self.model = YOLO(load_path)
        print("YOLO26n ready.")

    def detect_batch(self, samples):
        """
        samples: list of dicts with:
            source, frame, frame_id, timestamp

        Returns:
            packets: standardized detection packets
            annotated: {source: annotated_bgr_frame}
            batch_wall_ms: wall-clock latency for the complete batch
        """
        if not samples:
            return [], {}, 0.0

        images = [sample["frame"] for sample in samples]

        start = time.perf_counter()

        results = self.model.predict(
            source=images,
            device=YOLO_DEVICE,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            half=YOLO_HALF,
            verbose=False,
        )

        batch_wall_ms = (time.perf_counter() - start) * 1000.0

        packets = []
        annotated = {}

        for sample, result in zip(samples, results):
            detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    x1, y1, x2, y2 = [
                        float(v) for v in box.xyxy[0].tolist()
                    ]

                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": result.names[class_id],
                            "confidence": confidence,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )

            speed = result.speed or {}

            packet = {
                "source": sample["source"],
                "frame_id": sample["frame_id"],
                "timestamp": sample["timestamp"],
                "detections": detections,
                "preprocess_ms": float(speed.get("preprocess", 0.0)),
                "inference_ms": float(speed.get("inference", 0.0)),
                "postprocess_ms": float(speed.get("postprocess", 0.0)),
                "batch_wall_ms": batch_wall_ms,
            }

            packets.append(packet)
            annotated[sample["source"]] = result.plot()

        return packets, annotated, batch_wall_ms
