import math


def angle_norm_deg(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


class StoplineManager:
    def __init__(self, stoplines, default_stop_before_m=1.0):
        self.stoplines = list(stoplines or [])
        self.default_stop_before_m = float(default_stop_before_m or 1.0)

    def empty_context(self, reason):
        return {
            "distance_to_stopline_m": None,
            "stopline_id": None,
            "stopline_valid": False,
            "stopline_pose": None,
            "stopline_lateral_m": None,
            "stop_before_m": round(self.default_stop_before_m, 3),
            "stop_target_dist_m": None,
            "release_after_pass_m": 2.0,
            "road_id": None,
            "lane_id": None,
            "road_match": False,
            "lane_match": False,
            "route_connected_match": False,
            "route_valid": False,
            "route_valid_reason": None,
            "yaw_diff_deg": None,
            "selection_reject_reason": reason,
            "debug": reason,
        }

    def select(self, ego_ctx, map_name="", route_lane_id=None, route_road_id=None):
        if not self.stoplines:
            return self.empty_context("no_manual_stoplines")
        if not isinstance(ego_ctx, dict):
            return self.empty_context("ego_missing")

        candidates = []
        rejected = []
        ego_lane = ego_ctx.get("lane_id")
        ego_road = ego_ctx.get("road_id")
        for line in self.stoplines:
            line_map = str(line.get("map", "") or "").split("/")[-1]
            if line_map and map_name and line_map != map_name:
                continue

            try:
                dx = float(line["x"]) - float(ego_ctx["x"])
                dy = float(line["y"]) - float(ego_ctx["y"])
                longitudinal_m = dx * float(ego_ctx["fwd"].x) + dy * float(ego_ctx["fwd"].y)
                lateral_m = dx * float(ego_ctx["right"].x) + dy * float(ego_ctx["right"].y)
            except Exception:
                continue

            approach_m = float(line.get("approach_m", 120.0) or 120.0)
            release_after = max(1.0, float(line.get("release_after_pass_m", 2.0) or 2.0))
            if longitudinal_m < -release_after or longitudinal_m > approach_m:
                continue

            lat_limit = float(line.get("lateral_half_width_m", 6.0) or 6.0)
            # A broad gate keeps debug visibility for nearby stopline candidates,
            # but route_valid below uses a much tighter lane-level gate.
            debug_lat_limit = max(lat_limit, 14.0 if 0.0 <= longitudinal_m <= 20.0 else lat_limit)
            if abs(lateral_m) > debug_lat_limit:
                rejected.append({
                    "id": line.get("id"),
                    "dist": longitudinal_m,
                    "lat": lateral_m,
                    "road_id": line.get("road_id"),
                    "lane_id": line.get("lane_id"),
                    "road_match": False,
                    "lane_match": False,
                    "route_connected_match": False,
                    "yaw_diff": None,
                    "selection_reject_reason": "outside_debug_lateral_gate",
                    "route_valid_reason": "outside_debug_lateral_gate",
                })
                continue

            line_lane = line.get("lane_id")
            line_road = line.get("road_id")
            lane_match = _same_int(line_lane, route_lane_id) or _same_int(line_lane, ego_lane)
            road_match = _same_int(line_road, route_road_id) or _same_int(line_road, ego_road)
            route_connected_match = False
            yaw_diff = abs(angle_norm_deg(float(line.get("yaw_deg", 0.0) or 0.0) - float(ego_ctx.get("yaw_deg", 0.0) or 0.0)))

            # Route-valid means this stopline belongs to the ego approach,
            # not merely that it is near the vehicle. Logs showed lateral=4.312
            # being accepted as same-lane, which can select the adjacent approach.
            route_lat_limit = min(max(3.0, lat_limit), 4.0)
            connected_lat_limit = min(max(3.2, lat_limit), 4.0)
            lateral_ok = abs(lateral_m) <= route_lat_limit
            connected_lateral_ok = abs(lateral_m) <= connected_lat_limit
            heading_ok = yaw_diff <= 85.0
            lane_known = line_lane is not None and (route_lane_id is not None or ego_lane is not None)
            lane_verified = (not lane_known) or lane_match
            same_route_segment = road_match and lane_verified
            connected_route_segment = route_connected_match and connected_lateral_ok
            route_valid = bool((same_route_segment and lateral_ok and heading_ok) or (connected_route_segment and heading_ok))

            reject_reason = None
            route_valid_reason = None
            if route_valid:
                route_valid_reason = "route_connected_lateral_heading" if connected_route_segment and not same_route_segment else "same_road_lane_lateral_heading"
            elif not road_match and not route_connected_match:
                reject_reason = "road_mismatch"
                route_valid_reason = "no_road_or_route_connection"
            elif not lane_verified:
                reject_reason = "lane_mismatch"
                route_valid_reason = "lane_not_on_route"
            elif not lateral_ok and not connected_route_segment:
                reject_reason = "lateral_mismatch"
                route_valid_reason = f"abs_lateral>{route_lat_limit:.1f}"
            elif not heading_ok:
                reject_reason = "heading_mismatch"
                route_valid_reason = "heading_not_aligned"
            else:
                reject_reason = "route_not_connected"
                route_valid_reason = "route_geometry_not_verified"

            if not route_valid:
                rejected.append({
                    "id": line.get("id"),
                    "dist": longitudinal_m,
                    "lat": lateral_m,
                    "road_id": line_road,
                    "lane_id": line_lane,
                    "road_match": road_match,
                    "lane_match": lane_match,
                    "route_connected_match": route_connected_match,
                    "yaw_diff": yaw_diff,
                    "selection_reject_reason": reject_reason,
                    "route_valid_reason": route_valid_reason,
                })
                continue

            score = abs(lateral_m) + 0.025 * max(0.0, longitudinal_m) + 0.012 * min(180.0, yaw_diff)
            if road_match:
                score -= 2.5
            if lane_match:
                score -= 4.0
            if longitudinal_m < 0.0:
                score += 20.0
            if yaw_diff > 120.0 and not lane_match:
                score += 8.0

            candidates.append({
                "score": score,
                "line": line,
                "dist": longitudinal_m,
                "lat": lateral_m,
                "yaw_diff": yaw_diff,
                "lane_match": lane_match,
                "road_match": road_match,
                "route_connected_match": route_connected_match,
                "route_valid": route_valid,
                "route_valid_reason": route_valid_reason,
            })

        if not candidates:
            if rejected:
                best_reject = min(rejected, key=lambda item: abs(float(item.get("lat", 999.0))) + 0.025 * max(0.0, float(item.get("dist", 999.0))))
                reject_reason = str(best_reject.get("selection_reject_reason") or "route_invalid")
                ctx = self.empty_context(f"TL_STOPLINE_REJECTED:{reject_reason}")
                ctx.update({
                    "candidate_stopline_id": best_reject.get("id"),
                    "distance_to_stopline_m": round(float(best_reject.get("dist", 0.0)), 3),
                    "stopline_lateral_m": round(float(best_reject.get("lat", 0.0)), 3),
                    "road_id": best_reject.get("road_id"),
                    "lane_id": best_reject.get("lane_id"),
                    "road_match": bool(best_reject.get("road_match")),
                    "lane_match": bool(best_reject.get("lane_match")),
                    "route_connected_match": bool(best_reject.get("route_connected_match")),
                    "route_valid": False,
                    "yaw_diff_deg": round(float(best_reject.get("yaw_diff", 0.0) or 0.0), 2)
                    if best_reject.get("yaw_diff") is not None else None,
                    "route_valid_reason": best_reject.get("route_valid_reason"),
                    "selection_reject_reason": reject_reason,
                    "debug": (
                        "TL_STOPLINE_REJECTED|TL_ROUTE_VALIDATION:"
                        f"id={best_reject.get('id')},dist={float(best_reject.get('dist', 0.0)):.2f},"
                        f"lat={float(best_reject.get('lat', 0.0)):.2f},road_id={best_reject.get('road_id')},"
                        f"lane_id={best_reject.get('lane_id')},road_match={int(bool(best_reject.get('road_match')))},"
                        f"lane_match={int(bool(best_reject.get('lane_match')))},"
                        f"route_connected_match={int(bool(best_reject.get('route_connected_match')))},"
                        f"yaw_diff={best_reject.get('yaw_diff')},route_valid=0,"
                        f"route_valid_reason={best_reject.get('route_valid_reason')},"
                        f"selection_reject_reason={reject_reason}"
                    ),
                })
                return ctx
            return self.empty_context("none_near_route")

        best = min(candidates, key=lambda item: item["score"])
        line = best["line"]
        dist_m = float(best["dist"])
        stop_before = float(line.get("stop_before_m", self.default_stop_before_m) or self.default_stop_before_m)
        release_after = max(1.0, float(line.get("release_after_pass_m", 2.0) or 2.0))
        return {
            "distance_to_stopline_m": round(dist_m, 3),
            "stopline_id": str(line.get("id", "manual_stopline")),
            "stopline_valid": True,
            "stopline_pose": {
                "x": float(line["x"]),
                "y": float(line["y"]),
                "z": float(line.get("z", 0.0) or 0.0),
                "yaw_deg": float(line.get("yaw_deg", 0.0) or 0.0),
            },
            "stopline_lateral_m": round(float(best["lat"]), 3),
            "stop_before_m": round(stop_before, 3),
            "stop_target_dist_m": round(dist_m - stop_before, 3),
            "release_after_pass_m": round(release_after, 3),
            "road_id": line.get("road_id"),
            "lane_id": line.get("lane_id"),
            "traffic_light_id": line.get("traffic_light_id"),
            "road_match": bool(best["road_match"]),
            "lane_match": bool(best["lane_match"]),
            "route_connected_match": bool(best["route_connected_match"]),
            "route_valid": bool(best["route_valid"]),
            "route_valid_reason": best.get("route_valid_reason"),
            "yaw_diff_deg": round(float(best.get("yaw_diff", 0.0) or 0.0), 2),
            "selection_reject_reason": None,
                "debug": (
                    "TL_STOPLINE_SELECTED|TL_ROUTE_VALIDATION:"
                f"id={line.get('id')},dist={dist_m:.2f},target={dist_m - stop_before:.2f},"
                f"lat={best['lat']:.2f},road_id={line.get('road_id')},lane_id={line.get('lane_id')},"
                f"lane_match={int(best['lane_match'])},road_match={int(best['road_match'])},"
                f"route_connected_match={int(best['route_connected_match'])},"
                f"yaw_diff={float(best.get('yaw_diff', 0.0)):.1f},"
                f"route_valid={int(bool(best['route_valid']))},"
                f"route_valid_reason={best.get('route_valid_reason')},selection_reject_reason=None"
            ),
        }


def _same_int(a, b):
    try:
        return a is not None and b is not None and int(a) == int(b)
    except Exception:
        return False
