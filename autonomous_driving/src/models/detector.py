from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: Union[str, Path], conf: float = 0.35, iou: float = 0.45):
        self.model_path = str(model_path)
        self.conf = conf
        self.iou = iou
        self.model = YOLO(self.model_path)

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        result = results[0]
        detections: List[Dict[str, Any]] = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            xyxy = tuple(map(int, box.xyxy[0].tolist()))
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id]

            detections.append(
                {
                    "box": xyxy,
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                }
            )

        return detections