from typing import Tuple, Optional
import cv2
import numpy as np


Box = Tuple[int, int, int, int]


def clamp_box(box: Box, frame_shape) -> Box:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)

    return x1, y1, x2, y2


def crop_from_box(frame: np.ndarray, box: Box) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def box_area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_center(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def draw_box(
    frame: np.ndarray,
    box: Box,
    label: str,
    color=(0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    ((text_w, text_h), _) = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    y_text = max(0, y1 - 8)
    cv2.rectangle(
        frame,
        (x1, max(0, y_text - text_h - 6)),
        (x1 + text_w + 8, y_text + 4),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 4, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return frame