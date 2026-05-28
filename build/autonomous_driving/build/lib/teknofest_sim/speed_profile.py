def braking_distance_m(speed_mps, decel_mps2=3.0, reaction_time_s=0.75, safety_margin_m=2.0, pitch_deg=0.0):
    v = max(0.0, float(speed_mps or 0.0))
    decel = max(0.1, float(decel_mps2 or 3.0))
    margin = float(safety_margin_m or 0.0)
    try:
        pitch = float(pitch_deg or 0.0)
    except Exception:
        pitch = 0.0
    if pitch < -2.0:
        margin += min(4.0, abs(pitch) * 0.35)
    return (v * v) / (2.0 * decel) + float(reaction_time_s or 0.0) * v + margin


def stopline_speed_profile(
    *,
    current_speed_mps,
    stop_target_dist_m,
    default_speed_mps,
    approach_speed_mps=4.0,
    crawl_speed_mps=1.1,
    stop_zone_m=0.5,
    crawl_zone_m=4.0,
    decel_mps2=3.0,
    reaction_time_s=0.75,
    safety_margin_m=2.0,
    pitch_deg=0.0,
):
    target_dist = None if stop_target_dist_m is None else float(stop_target_dist_m)
    v = max(0.0, float(current_speed_mps or 0.0))
    brake_dist = braking_distance_m(v, decel_mps2, reaction_time_s, safety_margin_m, pitch_deg)

    if target_dist is None:
        return {
            "phase": "NO_STOPLINE",
            "target_speed_mps": max(0.0, float(default_speed_mps or 0.0)),
            "should_stop": False,
            "emergency_brake": False,
            "brake_distance_m": round(brake_dist, 3),
            "marker": "TL_APPROACH_PROFILE:no_stopline|TL_BRAKE_PROFILE:no_stopline",
        }

    if target_dist <= stop_zone_m:
        return {
            "phase": "STOP_ZONE",
            "target_speed_mps": 0.0,
            "should_stop": True,
            "emergency_brake": True,
            "brake_distance_m": round(brake_dist, 3),
            "marker": "TL_APPROACH_PROFILE:stop_zone|TL_BRAKE_PROFILE:stop_zone|TL_BRAKE_TO_STOPLINE",
        }

    if target_dist <= brake_dist:
        return {
            "phase": "BRAKING_ZONE",
            "target_speed_mps": 0.0,
            "should_stop": True,
            "emergency_brake": bool(target_dist <= max(stop_zone_m, brake_dist * 0.80) and v > 0.35),
            "brake_distance_m": round(brake_dist, 3),
            "marker": "TL_APPROACH_PROFILE:braking_zone|TL_BRAKE_PROFILE:braking_zone|TL_BRAKE_TO_STOPLINE",
        }

    if target_dist <= crawl_zone_m:
        target_speed = min(float(crawl_speed_mps or 1.1), max(0.0, (2.0 * max(0.1, decel_mps2) * target_dist) ** 0.5 * 0.45))
        return {
            "phase": "CRAWL_ZONE",
            "target_speed_mps": max(0.0, target_speed),
            "should_stop": False,
            "emergency_brake": False,
            "brake_distance_m": round(brake_dist, 3),
            "marker": "TL_APPROACH_PROFILE:crawl_zone|TL_BRAKE_PROFILE:crawl_zone",
        }

    target_speed = min(float(default_speed_mps or 0.0), float(approach_speed_mps or default_speed_mps or 0.0))
    return {
        "phase": "APPROACH_ZONE",
        "target_speed_mps": max(0.0, target_speed),
        "should_stop": False,
        "emergency_brake": False,
        "brake_distance_m": round(brake_dist, 3),
        "marker": "TL_APPROACH_PROFILE:approach_zone|TL_BRAKE_PROFILE:approach_zone",
    }
