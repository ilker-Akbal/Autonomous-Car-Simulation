import cv2


def clamp_box(box, shape):
    h, w = shape[:2]
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


def draw_box(frame, box, label, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    ((text_w, text_h), _) = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    )
    y_text = max(20, y1 - 8)

    cv2.rectangle(
        frame,
        (x1, max(0, y_text - text_h - 8)),
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


def draw_overlay(frame, summary):
    lines = [
        f"ACTION: {summary['action']}",
        f"TL_STATE: {summary['traffic_light_state']}",
        f"DIST: {summary['critical_car_distance']:.2f} m" if summary["critical_car_distance"] is not None else "DIST: -",
        f"REASON: {summary['reason']}",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30