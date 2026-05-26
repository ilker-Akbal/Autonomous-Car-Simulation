from pathlib import Path
from typing import List, Dict, Any

from ultralytics import YOLO


class Predictor:
    def __init__(self, model_path: str, conf: float = 0.25):
        self.model_path = str(model_path)
        self.conf = conf
        self.model = YOLO(self.model_path)

    def predict_frame(self, frame) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            verbose=False,
        )

        detections = []
        result = results[0]

        names = result.names
        boxes = result.boxes

        if boxes is None:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id]

            detections.append(
                {
                    "box": tuple(map(int, xyxy)),
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                }
            )

        return detections