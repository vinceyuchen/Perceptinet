import time
from pathlib import Path

from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH,
    YOLO_DEVICE,
    YOLO_IMAGE_SIZE,
    YOLO_CONFIDENCE,
    TARGET_CLASS_IDS,
    CLASS_NAME_OVERRIDES,
)


class YOLO26Perception:
    """
    Shared YOLO26n road-user perception engine.

    Only Perceptinet-relevant road-user classes are returned:
      pedestrian
      bicycle
      car
      motorcycle
      bus
      truck
    """

    def __init__(self):
        model_path = Path(YOLO_MODEL_PATH)
        load_path = str(model_path) if model_path.exists() else "yolo26n.pt"

        print(f"Loading YOLO26n: {load_path}")
        self.model = YOLO(load_path)
        print("YOLO26n ready.")
        print("Target classes: pedestrian, bicycle, car, motorcycle, bus, truck")

    @staticmethod
    def _normalize_class_name(name):
        return CLASS_NAME_OVERRIDES.get(name, name)

    def detect_batch(self, samples):
        if not samples:
            return [], {}, 0.0

        images = [sample["frame"] for sample in samples]

        start = time.perf_counter()

        # No `half=` argument here. This avoids the current Ultralytics
        # deprecation warning while keeping the pipeline stable.
        results = self.model.predict(
            source=images,
            device=YOLO_DEVICE,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            classes=TARGET_CLASS_IDS,
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

                    raw_name = result.names[class_id]
                    class_name = self._normalize_class_name(raw_name)

                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
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

            # result.plot() would use Ultralytics' original label "person".
            # Draw our own boxes so the UI consistently displays "pedestrian".
            frame = sample["frame"].copy()

            import cv2

            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                label = f'{det["class_name"]} {det["confidence"]:.2f}'

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                (tw, th), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    2,
                )

                label_y1 = max(0, y1 - th - baseline - 6)
                cv2.rectangle(
                    frame,
                    (x1, label_y1),
                    (x1 + tw + 8, y1),
                    (0, 0, 0),
                    -1,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            annotated[sample["source"]] = frame

        return packets, annotated, batch_wall_ms
