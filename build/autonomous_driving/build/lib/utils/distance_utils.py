from typing import Tuple


def estimate_distance_level(box: Tuple[int, int, int, int], frame_shape) -> str:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box

    box_h = max(1, y2 - y1)
    ratio = box_h / float(h)

    if ratio > 0.45:
        return "very_near"
    if ratio > 0.28:
        return "near"
    if ratio > 0.14:
        return "medium"
    return "far"