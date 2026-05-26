from typing import Dict
import cv2
import numpy as np


def _count_mask_pixels(hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> int:
    mask = cv2.inRange(hsv, lower, upper)
    return int(np.count_nonzero(mask))


def detect_traffic_light_color(crop) -> Dict:
    if crop is None or crop.size == 0:
        return {"state": "unknown", "scores": {"red": 0, "yellow": 0, "green": 0}}

    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return {"state": "unknown", "scores": {"red": 0, "yellow": 0, "green": 0}}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    top = hsv[0:h // 3, :]
    mid = hsv[h // 3: 2 * h // 3, :]
    bot = hsv[2 * h // 3:, :]

    red_1_low = np.array([0, 90, 90], dtype=np.uint8)
    red_1_up = np.array([10, 255, 255], dtype=np.uint8)
    red_2_low = np.array([160, 90, 90], dtype=np.uint8)
    red_2_up = np.array([180, 255, 255], dtype=np.uint8)

    yellow_low = np.array([15, 90, 90], dtype=np.uint8)
    yellow_up = np.array([40, 255, 255], dtype=np.uint8)

    green_low = np.array([40, 70, 70], dtype=np.uint8)
    green_up = np.array([95, 255, 255], dtype=np.uint8)

    red_score = (
        _count_mask_pixels(top, red_1_low, red_1_up)
        + _count_mask_pixels(top, red_2_low, red_2_up)
    )

    yellow_score = _count_mask_pixels(mid, yellow_low, yellow_up)
    green_score = _count_mask_pixels(bot, green_low, green_up)

    scores = {
        "red": red_score,
        "yellow": yellow_score,
        "green": green_score,
    }

    best_state = max(scores, key=scores.get)
    best_score = scores[best_state]

    min_pixels = max(10, int((h * w) * 0.01))
    if best_score < min_pixels:
        best_state = "unknown"

    return {"state": best_state, "scores": scores}