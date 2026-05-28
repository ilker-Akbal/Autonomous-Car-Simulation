import time

from teknofest_sim.speed_profile import stopline_speed_profile


class TrafficLightStoplineStateMachine:
    STATES = {
        "CLEAR",
        "APPROACHING_RED_OR_YELLOW",
        "BRAKING_TO_STOPLINE",
        "STOPPED_AT_STOPLINE",
        "WAITING_FOR_GREEN",
        "GREEN_RELEASED",
        "PASSED_INTERSECTION",
        "RED_LIGHT_VIOLATION",
    }

    def __init__(
        self,
        *,
        default_go_speed=8.333,
        approach_speed=4.0,
        crawl_speed=1.1,
        stop_speed=0.0,
        green_confirm_count=2,
        red_confirm_count=1,
        yellow_confirm_count=1,
        decel_mps2=3.0,
        reaction_time_s=0.75,
        safety_margin_m=2.0,
    ):
        self.state = "CLEAR"
        self.stopline_id = None
        self.hold_active = False
        self.hold_stopline_id = None
        self.hold_since = 0.0
        self.hold_reason = None
        self.green_counts = {}
        self.red_counts = {}
        self.yellow_counts = {}
        self.last_green_time = 0.0
        self.last_non_green_light_state = "unknown"
        self.default_go_speed = float(default_go_speed or 0.0)
        self.approach_speed = float(approach_speed or self.default_go_speed)
        self.crawl_speed = float(crawl_speed or 1.1)
        self.stop_speed = float(stop_speed or 0.0)
        self.green_confirm_count = max(1, int(green_confirm_count or 2))
        self.red_confirm_count = max(1, int(red_confirm_count or 1))
        self.yellow_confirm_count = max(1, int(yellow_confirm_count or 1))
        self.decel_mps2 = float(decel_mps2 or 3.0)
        self.reaction_time_s = float(reaction_time_s or 0.75)
        self.safety_margin_m = float(safety_margin_m or 2.0)
        self.stop_zone_m = 0.35

    def transition(self, new_state, marker):
        old = self.state
        if new_state not in self.STATES:
            new_state = "CLEAR"
        self.state = new_state
        if old != new_state:
            return f"TL_SM_TRANSITION:{old}->{new_state}|{marker}"
        return marker

    def _same_hold_stopline(self, stopline_id):
        if not self.hold_active:
            return False
        if self.hold_stopline_id is None or stopline_id is None:
            return False
        return str(self.hold_stopline_id) == str(stopline_id)

    def _reset_counts_except(self, state, stopline_id):
        key = str(stopline_id) if stopline_id is not None else None
        if state != "green" and key is not None:
            self.green_counts[key] = 0
        if state != "red" and key is not None:
            self.red_counts[key] = 0
        if state != "yellow" and key is not None:
            self.yellow_counts[key] = 0

    def _count(self, table, stopline_id):
        key = str(stopline_id)
        table[key] = table.get(key, 0) + 1
        return table[key]

    def _set_hold(self, stopline_id, reason, light_state):
        if self.hold_active:
            marker = "TL_HOLD_KEEP_UNKNOWN"
        elif light_state == "unknown":
            marker = "TL_HOLD_UNKNOWN_NEAR_STOPLINE"
        elif light_state == "yellow":
            marker = "TL_HOLD_YELLOW"
        else:
            marker = "TL_HOLD_RED"
        self.hold_active = True
        self.hold_stopline_id = stopline_id
        self.hold_reason = reason
        if self.hold_since <= 0.0:
            self.hold_since = time.time()
        if light_state in {"red", "yellow", "unknown"}:
            self.last_non_green_light_state = light_state
        return marker

    def _release_hold(self):
        self.hold_active = False
        self.hold_stopline_id = None
        self.hold_since = 0.0
        self.hold_reason = None
        self.last_green_time = time.time()
        return "TL_RELEASE_GREEN_CONFIRMED|TL_RELEASE_CONFIRMED_GREEN"

    def _confirmed_green(self, state, stopline_id, traffic_light_det):
        if state != "green" or not stopline_id:
            if stopline_id is not None:
                self.green_counts[str(stopline_id)] = 0
            return False, "TL_RELEASE_REJECTED_REASON:not_green"

        det = traffic_light_det or {}
        try:
            state_conf = float(det.get("traffic_light_state_confidence", det.get("state_confidence", det.get("confidence", 0.0))) or 0.0)
        except Exception:
            state_conf = 0.0
        try:
            det_conf = float(det.get("det_confidence", det.get("detection_confidence", det.get("confidence", 0.0))) or 0.0)
        except Exception:
            det_conf = 0.0
        try:
            area_ratio = float(det.get("tl_area_ratio", det.get("area_ratio", 0.0)) or 0.0)
            height_ratio = float(det.get("tl_height_ratio", det.get("height_ratio", 0.0)) or 0.0)
        except Exception:
            area_ratio, height_ratio = 0.0, 0.0

        if state_conf < 0.55:
            return False, f"TL_RELEASE_REJECTED_REASON:low_state_conf:{state_conf:.2f}"
        if det_conf < 0.08:
            return False, f"TL_RELEASE_REJECTED_REASON:low_det_conf:{det_conf:.2f}"
        if area_ratio < 0.00008 and height_ratio < 0.015:
            return False, f"TL_RELEASE_REJECTED_REASON:tiny_green:a={area_ratio:.6f},h={height_ratio:.3f}"

        count = self._count(self.green_counts, stopline_id)
        if count < self.green_confirm_count:
            return False, f"TL_RELEASE_REJECTED_REASON:green_wait_stable:{count}/{self.green_confirm_count}"
        return True, "TL_RELEASE_GREEN_CONFIRMED|TL_RELEASE_CONFIRMED_GREEN"

    def _rule(self, decision, risk, target_speed, reason, **kwargs):
        out = {
            "decision": decision,
            "risk": risk,
            "target_speed": float(target_speed),
            "target_speed_mps": float(target_speed),
            "reason": reason,
            "should_stop": decision == "STOP",
            "stop_reason": kwargs.pop("stop_reason", None),
            "stop_distance_m": kwargs.pop("stop_distance_m", None),
            "distance_to_stopline_m": kwargs.pop("distance_to_stopline_m", None),
            "stop_target_distance_m": kwargs.pop("stop_target_distance_m", None),
            "active_stopline_id": kwargs.pop("active_stopline_id", None),
            "emergency_brake": bool(kwargs.pop("emergency_brake", False)),
            "release_condition": kwargs.pop("release_condition", None),
            "green_release_allowed": kwargs.pop("green_release_allowed", False),
            "tl_applicable": bool(kwargs.pop("tl_applicable", True)),
            "active_light_state": kwargs.pop("active_light_state", None),
            "stop_before_m": kwargs.pop("stop_before_m", None),
            "stop_required": bool(kwargs.pop("stop_required", decision == "STOP")),
            "release_allowed": bool(kwargs.pop("release_allowed", False)),
            "violation": bool(kwargs.pop("violation", False)),
            "priority_winner": kwargs.pop("priority_winner", "traffic_light_stopline"),
            "tl_debug": kwargs.pop("tl_debug", {}),
        }
        out.update(kwargs)
        if out["should_stop"]:
            out["target_speed"] = 0.0
            out["target_speed_mps"] = 0.0
            out["stop_required"] = True
        return out

    def _target_reached(self, target_dist_m):
        try:
            return float(target_dist_m) <= self.stop_zone_m
        except Exception:
            return False

    def _controlled_approach_speed(self, target_dist_m, profile):
        try:
            target_f = float(target_dist_m)
        except Exception:
            return min(self.approach_speed, self.default_go_speed)
        if target_f <= self.stop_zone_m:
            return self.stop_speed
        if target_f <= 1.5:
            return min(self.crawl_speed, max(0.35, target_f * 0.45))
        if target_f <= 8.0:
            return min(self.crawl_speed, max(0.65, target_f * 0.20))
        profile_speed = float((profile or {}).get("target_speed_mps", self.approach_speed) or self.approach_speed)
        return min(self.approach_speed, max(1.2, min(profile_speed, target_f * 0.22)))

    def _approach_rule(
        self,
        state,
        stopline,
        stopline_id,
        dist_f,
        target_f,
        marker,
        selected_marker,
        brake_marker,
        profile,
        debug,
        hold=False,
    ):
        speed = self._controlled_approach_speed(target_f, profile)
        reason_marker = "TL_HOLD_APPROACH_TO_STOPLINE" if hold else "TL_APPROACH_TO_STOPLINE"
        return self._rule(
            "SLOW",
            "MEDIUM",
            speed,
            f"{marker}|{reason_marker}|{selected_marker}|{brake_marker}",
            should_stop=False,
            stop_reason=f"traffic_light_{state}_approach",
            stop_distance_m=target_f,
            distance_to_stopline_m=dist_f,
            stop_target_distance_m=target_f,
            active_stopline_id=stopline_id,
            emergency_brake=False,
            release_condition="approach_stopline_hold" if hold else "approach_stopline",
            active_light_state=state,
            stop_before_m=stopline.get("stop_before_m"),
            stop_required=False,
            tl_debug=debug,
        )

    def _stop_at_target_rule(
        self,
        state,
        stopline,
        stopline_id,
        dist_f,
        target_f,
        marker,
        selected_marker,
        brake_marker,
        debug,
        emergency=False,
    ):
        return self._rule(
            "STOP",
            "HIGH",
            self.stop_speed,
            f"{marker}|{selected_marker}|{brake_marker}",
            stop_reason=f"traffic_light_{state}_hold",
            stop_distance_m=target_f,
            distance_to_stopline_m=dist_f,
            stop_target_distance_m=target_f,
            active_stopline_id=stopline_id,
            emergency_brake=bool(emergency),
            release_condition="wait_confirmed_green",
            active_light_state=state,
            stop_before_m=stopline.get("stop_before_m"),
            tl_debug=debug,
        )

    def _selected_marker(self, stopline, stopline_id, dist, target_dist):
        return (
            f"TL_STOPLINE_SELECTED id={stopline_id} dist={dist} target_dist={target_dist} "
            f"lateral={stopline.get('lateral_m', stopline.get('stopline_lateral_m'))} "
            f"road_id={stopline.get('road_id')} lane_id={stopline.get('lane_id')} "
            f"route_valid={int(bool(stopline.get('route_valid', stopline.get('valid', False))))} "
            f"road_match={int(bool(stopline.get('road_match', False)))} "
            f"lane_match={int(bool(stopline.get('lane_match', False)))} "
            f"route_connected_match={int(bool(stopline.get('route_connected_match', False)))} "
            f"yaw_diff={stopline.get('yaw_diff_deg')} "
            f"route_valid_reason={stopline.get('route_valid_reason')} "
            f"selection_reject_reason={stopline.get('selection_reject_reason')}"
            f"|TL_ROUTE_VALIDATION"
        )

    def evaluate(self, traffic_light_state, stopline, current_speed_mps, traffic_light_det=None, pitch_deg=0.0):
        now = time.time()
        state = str(traffic_light_state or "unknown").lower().strip()
        if state not in {"red", "yellow", "green"}:
            state = "unknown"

        stopline = stopline or {}
        stopline_valid = bool(stopline.get("valid", stopline.get("stopline_valid", False)))
        stopline_id = stopline.get("id", stopline.get("stopline_id"))
        dist = stopline.get("distance_m", stopline.get("distance_to_stopline_m"))
        target_dist = stopline.get("target_dist_m", stopline.get("stop_target_dist_m"))
        release_after = max(1.0, float(stopline.get("release_after_pass_m", 2.0) or 2.0))

        if stopline_id != self.stopline_id:
            self.stopline_id = stopline_id
            if not self._same_hold_stopline(stopline_id) and self.state in {"GREEN_RELEASED", "PASSED_INTERSECTION", "CLEAR"}:
                self.state = "CLEAR"

        profile = stopline_speed_profile(
            current_speed_mps=current_speed_mps,
            stop_target_dist_m=target_dist,
            default_speed_mps=self.default_go_speed,
            approach_speed_mps=self.approach_speed,
            crawl_speed_mps=self.crawl_speed,
            stop_zone_m=self.stop_zone_m,
            decel_mps2=self.decel_mps2,
            reaction_time_s=self.reaction_time_s,
            safety_margin_m=self.safety_margin_m,
            pitch_deg=pitch_deg,
        )

        debug = {
            "traffic_light_state": state,
            "tl_state_machine_state": self.state,
            "stopline_id": stopline_id,
            "stopline_dist_m": dist,
            "stop_target_dist_m": target_dist,
            "release_after_pass_m": release_after,
            "current_speed_mps": round(float(current_speed_mps or 0.0), 3),
            "brake_distance_m": profile.get("brake_distance_m"),
            "profile_phase": profile.get("phase"),
            "hold_active": self.hold_active,
            "hold_stopline_id": self.hold_stopline_id,
            "green_release_age": round(now - self.last_green_time, 3) if self.last_green_time else None,
            "route_valid": bool(stopline.get("route_valid", stopline_valid)),
            "road_match": bool(stopline.get("road_match", False)),
            "lane_match": bool(stopline.get("lane_match", False)),
            "route_connected_match": bool(stopline.get("route_connected_match", False)),
            "route_valid_reason": stopline.get("route_valid_reason"),
            "selection_reject_reason": stopline.get("selection_reject_reason"),
            "markers": [profile.get("marker")],
        }

        if not stopline_valid or stopline_id is None or dist is None or target_dist is None:
            if self.hold_active and (stopline_id is None or self._same_hold_stopline(stopline_id)):
                marker = self.transition(
                    "WAITING_FOR_GREEN",
                    "TL_HOLD_KEEP_UNKNOWN:no_valid_stopline_context|TL_RELEASE_REJECTED_REASON:no_route_stopline_match",
                )
                return self._rule(
                    "STOP",
                    "HIGH",
                    self.stop_speed,
                    marker,
                    stop_reason="traffic_light_hold_unknown",
                    release_condition="wait_confirmed_green",
                    active_light_state=state,
                    tl_debug=debug,
                )
            return None

        dist_f = float(dist)
        target_f = float(target_dist)
        selected_marker = self._selected_marker(stopline, stopline_id, dist, target_dist)
        brake_marker = (
            f"TL_BRAKE_PROFILE speed={float(current_speed_mps or 0.0):.2f} "
            f"brake_distance={profile.get('brake_distance_m')} "
            f"stop_target_dist={target_dist} target_speed={profile.get('target_speed_mps')}"
        )

        if state in {"red", "yellow", "unknown"}:
            self._reset_counts_except(state, stopline_id)
            self.last_non_green_light_state = state

        if self.hold_active and self._same_hold_stopline(stopline_id) and dist_f < -0.05 and state in {"red", "yellow", "unknown"}:
            marker = self.transition("RED_LIGHT_VIOLATION", f"TL_RED_LIGHT_VIOLATION|TL_RED_PASSED_VIOLATION:{state}")
            return self._rule(
                "STOP",
                "HIGH",
                self.stop_speed,
                f"{marker}|{selected_marker}",
                stop_reason=f"traffic_light_{state}_passed_violation",
                stop_distance_m=target_f,
                distance_to_stopline_m=dist_f,
                stop_target_distance_m=target_f,
                active_stopline_id=stopline_id,
                emergency_brake=True,
                release_condition="wait_confirmed_green",
                active_light_state=state,
                stop_before_m=stopline.get("stop_before_m"),
                violation=True,
                tl_debug=debug,
            )

        if self.state == "PASSED_INTERSECTION" and dist_f > 0.0:
            marker = self.transition("APPROACHING_RED_OR_YELLOW", "TL_PASSED_INTERSECTION_REJECTED_POSITIVE_DIST")
            debug["markers"].append(marker)

        if self.state == "GREEN_RELEASED":
            if state in {"red", "yellow"} and dist_f > 0.0:
                hold_marker = self._set_hold(stopline_id, "non_green_after_release_before_pass", state)
                if self._target_reached(target_f):
                    marker = self.transition("WAITING_FOR_GREEN", f"{hold_marker}|TL_STOPPED_AT_TARGET")
                    return self._stop_at_target_rule(
                        state, stopline, stopline_id, dist_f, target_f, marker, selected_marker, brake_marker, debug
                    )
                marker = self.transition("BRAKING_TO_STOPLINE", f"{hold_marker}|TL_BRAKE_TO_STOPLINE")
                return self._approach_rule(
                    state, stopline, stopline_id, dist_f, target_f, marker, selected_marker, brake_marker, profile, debug, hold=True
                )
            if dist_f <= -release_after:
                marker = self.transition("PASSED_INTERSECTION", "TL_PASSED_INTERSECTION_VALIDATED")
                return self._rule(
                    "GO",
                    "LOW",
                    self.default_go_speed,
                    f"{marker}|{selected_marker}",
                    distance_to_stopline_m=dist_f,
                    stop_target_distance_m=target_f,
                    active_stopline_id=stopline_id,
                    release_condition="passed_after_confirmed_green",
                    active_light_state=state,
                    stop_before_m=stopline.get("stop_before_m"),
                    release_allowed=True,
                    tl_debug=debug,
                )
            return self._rule(
                "GO",
                "LOW",
                self.default_go_speed,
                f"TL_RELEASE_GREEN_CONFIRMED|TL_RELEASE_CONFIRMED_GREEN:GREEN_RELEASED|{selected_marker}",
                distance_to_stopline_m=dist_f,
                stop_target_distance_m=target_f,
                active_stopline_id=stopline_id,
                release_condition="confirmed_green",
                green_release_allowed=True,
                active_light_state=state,
                stop_before_m=stopline.get("stop_before_m"),
                release_allowed=True,
                tl_debug=debug,
            )

        green_ok, green_reason = self._confirmed_green(state, stopline_id, traffic_light_det)
        if green_ok:
            release_marker = self._release_hold()
            marker = self.transition("GREEN_RELEASED", release_marker)
            return self._rule(
                "GO",
                "LOW",
                self.default_go_speed,
                f"{marker}:green_light_confirmed_stable|{selected_marker}",
                stop_distance_m=target_f,
                distance_to_stopline_m=dist_f,
                stop_target_distance_m=target_f,
                active_stopline_id=stopline_id,
                release_condition="confirmed_green",
                green_release_allowed=True,
                active_light_state=state,
                stop_before_m=stopline.get("stop_before_m"),
                release_allowed=True,
                tl_debug=debug,
            )

        if self.hold_active and self._same_hold_stopline(stopline_id):
            if self._target_reached(target_f):
                marker = self.transition("WAITING_FOR_GREEN", f"TL_HOLD_KEEP_UNKNOWN:{green_reason}|TL_STOPPED_AT_TARGET")
                return self._stop_at_target_rule(
                    state, stopline, stopline_id, dist_f, target_f, marker, selected_marker, brake_marker, debug
                )
            marker = self.transition("BRAKING_TO_STOPLINE", f"TL_HOLD_KEEP_UNKNOWN:{green_reason}|TL_BRAKE_TO_STOPLINE")
            return self._approach_rule(
                state, stopline, stopline_id, dist_f, target_f, marker, selected_marker, brake_marker, profile, debug, hold=True
            )

        if state == "green":
            return None

        red_confirmed = state == "red" and self._count(self.red_counts, stopline_id) >= self.red_confirm_count
        yellow_confirmed = state == "yellow" and self._count(self.yellow_counts, stopline_id) >= self.yellow_confirm_count
        try:
            brake_distance_m = float(profile.get("brake_distance_m") or 0.0)
        except Exception:
            brake_distance_m = 0.0
        unknown_risk = state == "unknown" and (
            dist_f <= 8.0
            or target_f <= 6.0
            or target_f <= max(6.0, brake_distance_m)
        )
        if not (red_confirmed or yellow_confirmed or unknown_risk):
            return None

        if unknown_risk:
            hold_marker = self._set_hold(stopline_id, "unknown_near_stopline", state)
            if self._target_reached(target_f):
                marker = self.transition("WAITING_FOR_GREEN", f"{hold_marker}:TL_UNKNOWN_NEAR_STOPLINE_HOLD|TL_STOPPED_AT_TARGET")
                return self._stop_at_target_rule(
                    state, stopline, stopline_id, dist_f, target_f, marker, selected_marker, brake_marker, debug
                )
            marker = self.transition("BRAKING_TO_STOPLINE", f"{hold_marker}:TL_UNKNOWN_NEAR_STOPLINE_APPROACH|TL_BRAKE_TO_STOPLINE")
            return self._approach_rule(
                state, stopline, stopline_id, dist_f, target_f, marker, selected_marker, brake_marker, profile, debug, hold=True
            )

        if profile["should_stop"]:
            hold_marker = self._set_hold(stopline_id, f"{state}_braking_to_stopline", state)
            if not self._target_reached(target_f):
                marker = self.transition("BRAKING_TO_STOPLINE", f"{profile['marker']}|TL_STOPLINE_CONTROL|TL_BRAKE_TO_STOPLINE|{hold_marker}")
                return self._approach_rule(
                    state, stopline, stopline_id, dist_f, target_f, marker, selected_marker, brake_marker, profile, debug, hold=True
                )
            target_marker = "TL_STOPPED_AT_TARGET"
            if float(current_speed_mps or 0.0) <= 0.15 and abs(target_f) > 0.75:
                target_marker = f"{target_marker}|TL_STOP_TARGET_ERROR:target_dist={target_f:.2f}"
            marker = self.transition("STOPPED_AT_STOPLINE", f"{profile['marker']}|TL_STOPLINE_CONTROL|{target_marker}|{hold_marker}")
            return self._stop_at_target_rule(
                state,
                stopline,
                stopline_id,
                dist_f,
                target_f,
                marker,
                selected_marker,
                brake_marker,
                debug,
                emergency=profile["emergency_brake"],
            )

        marker = self.transition("APPROACHING_RED_OR_YELLOW", profile["marker"])
        return self._rule(
            "SLOW",
            "MEDIUM",
            profile["target_speed_mps"],
            f"{marker}|{selected_marker}|{brake_marker}",
            should_stop=False,
            stop_reason=f"traffic_light_{state}",
            stop_distance_m=target_f,
            distance_to_stopline_m=dist_f,
            stop_target_distance_m=target_f,
            active_stopline_id=stopline_id,
            emergency_brake=False,
            release_condition="approach_stopline",
            active_light_state=state,
            stop_before_m=stopline.get("stop_before_m"),
            stop_required=False,
            tl_debug=debug,
        )
