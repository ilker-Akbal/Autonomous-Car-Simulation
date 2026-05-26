def compute_steering(frame_shape, left_lane, right_lane):
    h, w = frame_shape[:2]
    car_center = w // 2

    if left_lane and right_lane:
        _, _, lx2, _ = left_lane
        _, _, rx2, _ = right_lane

        lane_center = (lx2 + rx2) // 2

    elif left_lane:
        _, _, lx2, _ = left_lane
        lane_center = lx2 + 200

    elif right_lane:
        _, _, rx2, _ = right_lane
        lane_center = rx2 - 200

    else:
        return "UNKNOWN", 0

    offset = lane_center - car_center

    if abs(offset) < 30:
        return "STRAIGHT", offset
    elif offset > 0:
        return "RIGHT", offset
    else:
        return "LEFT", offset