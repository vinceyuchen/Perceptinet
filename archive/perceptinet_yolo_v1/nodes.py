import threading


class PerceptionNode:
    """
    Lightweight logical perception node.

    The Onboard Camera and each RSU Camera use separate node objects.
    YOLO itself is shared by the perception module so the prototype does
    not load multiple copies of the same model into GPU memory.
    """

    def __init__(self, source_id):
        self.source_id = source_id
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = None
        self._timestamp = None
        self._last_processed_frame_id = None

    def ingest_carla_image(self, image, converter):
        frame = converter(image)
        with self._lock:
            self._frame = frame
            self._frame_id = int(image.frame)
            self._timestamp = float(image.timestamp)

    def get_new_frame(self):
        with self._lock:
            if self._frame is None:
                return None

            if self._frame_id == self._last_processed_frame_id:
                return None

            self._last_processed_frame_id = self._frame_id

            return {
                "source": self.source_id,
                "frame": self._frame.copy(),
                "frame_id": self._frame_id,
                "timestamp": self._timestamp,
            }
