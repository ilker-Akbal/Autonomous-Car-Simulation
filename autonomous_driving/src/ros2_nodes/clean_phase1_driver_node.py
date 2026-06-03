#!/usr/bin/env python3
import json
import math
import time
from dataclasses import dataclass
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


def clamp(value, low, high):
    return max(low, min(high, value))


def rate_limit(prev_value, desired_value, max_delta):
    if desired_value > prev_value + max_delta:
        return prev_value + max_delta
    if desired_value < prev_value - max_delta:
        return prev_value - max_delta
    return desired_value


def norm_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def dot_location(a, b):
    return float(a.x * b.x + a.y * b.y + a.z * b.z)


def distance_2d(a, b):
    return math.hypot(float(a.x - b.x), float(a.y - b.y))


@dataclass
class SignTrack:
    rule: str
    confidence: float
    first_seen_s: float
    last_seen_s: float
    hits: int
    source: str
    actor_id: Optional[int] = None
    distance_m: Optional[float] = None


@dataclass
class LocalPath:
    mode: str
    branch: str
    waypoints: List[object]
    target_wp: object
    lookahead_m: float
    candidates: str
    reason: str
    max_lateral_jump_m: float = 0.0
    locked: bool = False
    target_index: int = -1
    progress_index: int = -1
    locked_path_len: int = 0
    stale: bool = False
    release_reason: Optional[str] = None


class CleanPhase1DriverNode(Node):
    def __init__(self):
        super().__init__("clean_phase1_driver_node")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")

        self.declare_parameter("command_topic", "/adas/phase1/command")
        self.declare_parameter("mission_topic", "/adas/teknofest/mission")
        self.declare_parameter("sign_facts_topic", "/adas/phase1/sign_facts_json")
        self.declare_parameter("lane_vision_topic", "/adas/phase1/lane_vision_json")

        self.declare_parameter("tick_hz", 10.0)
        self.declare_parameter("lookahead_m", 8.0)
        self.declare_parameter("path_length_m", 36.0)
        self.declare_parameter("path_step_m", 2.0)
        self.declare_parameter("junction_probe_m", 18.0)
        self.declare_parameter("junction_lock_s", 2.0)
        self.declare_parameter("junction_exit_hold_s", 1.2)
        self.declare_parameter("cruise_speed_mps", 5.8)
        self.declare_parameter("turn_speed_mps", 3.2)
        self.declare_parameter("caution_speed_mps", 2.8)
        self.declare_parameter("max_steer_command", 0.55)
        self.declare_parameter("lane_heading_gain", 0.78)
        self.declare_parameter("lane_offset_gain", 0.045)

        self.declare_parameter("traffic_light_detect_m", 65.0)
        self.declare_parameter("red_slowdown_m", 35.0)
        self.declare_parameter("red_hard_brake_m", 6.0)
        self.declare_parameter("stop_before_line_m", 1.0)
        self.declare_parameter("green_release_s", 5.0)

        self.declare_parameter("sign_detect_m", 45.0)
        self.declare_parameter("sign_hold_s", 12.0)
        self.declare_parameter("sign_min_hits", 2)
        self.declare_parameter("sign_min_confidence", 0.55)

        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)

        self.command_topic = str(self.get_parameter("command_topic").value)
        self.mission_topic = str(self.get_parameter("mission_topic").value)
        self.sign_facts_topic = str(self.get_parameter("sign_facts_topic").value)
        self.lane_vision_topic = str(self.get_parameter("lane_vision_topic").value)

        self.tick_hz = float(self.get_parameter("tick_hz").value)
        self.lookahead_m = float(self.get_parameter("lookahead_m").value)
        self.path_length_m = float(self.get_parameter("path_length_m").value)
        self.path_step_m = float(self.get_parameter("path_step_m").value)
        self.junction_probe_m = float(self.get_parameter("junction_probe_m").value)
        self.junction_lock_s = float(self.get_parameter("junction_lock_s").value)
        self.junction_exit_hold_s = float(self.get_parameter("junction_exit_hold_s").value)
        self.cruise_speed_mps = float(self.get_parameter("cruise_speed_mps").value)
        self.turn_speed_mps = float(self.get_parameter("turn_speed_mps").value)
        self.caution_speed_mps = float(self.get_parameter("caution_speed_mps").value)
        self.max_steer_command = float(self.get_parameter("max_steer_command").value)
        self.lane_heading_gain = float(self.get_parameter("lane_heading_gain").value)
        self.lane_offset_gain = float(self.get_parameter("lane_offset_gain").value)

        self.traffic_light_detect_m = float(self.get_parameter("traffic_light_detect_m").value)
        self.red_slowdown_m = float(self.get_parameter("red_slowdown_m").value)
        self.red_hard_brake_m = float(self.get_parameter("red_hard_brake_m").value)
        self.stop_before_line_m = float(self.get_parameter("stop_before_line_m").value)
        self.green_release_s = float(self.get_parameter("green_release_s").value)

        self.sign_detect_m = float(self.get_parameter("sign_detect_m").value)
        self.sign_hold_s = float(self.get_parameter("sign_hold_s").value)
        self.sign_min_hits = int(self.get_parameter("sign_min_hits").value)
        self.sign_min_confidence = float(self.get_parameter("sign_min_confidence").value)

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.ego = None
        self.last_ego_lookup_s = 0.0

        self.latest_mission = {}
        self.latest_lane_vision = None
        self.latest_lane_vision_s = 0.0
        self.last_reliable_lane_vision = None
        self.last_reliable_lane_vision_s = 0.0
        self.sign_tracks = {}
        self.last_driver_steer = 0.0
        self.last_raw_steer = 0.0
        self.last_limited_steer = 0.0
        self.last_steer_rate_limited = False
        self.last_accepted_target_wp = None
        self.last_accepted_path = None
        self.active_junction_branch = None
        self.last_junction_release_reason = None
        self.last_target_diag = {}
        self.stable_lane_path = None
        self.red_hold_light_id = None
        self.green_release_until_s = 0.0
        self.last_log_s = {
            "lane": 0.0,
            "tl": 0.0,
            "sign": 0.0,
            "cmd": 0.0,
            "path": 0.0,
            "junction": 0.0,
            "target": 0.0,
            "fusion": 0.0,
        }
        self.last_lane_source = "map"
        self.last_lane_source_change_s = 0.0
        self.smoothed_vision_weight = 0.0
        self.vision_offset_bias_norm = None
        self.vision_bias_samples = 0
        self.junction_exit_ramp_until_s = 0.0
        self.junction_exit_ramp_start_s = 0.0

        self.command_pub = self.create_publisher(String, self.command_topic, 10)
        self.create_subscription(String, self.mission_topic, self.mission_cb, 10)
        self.create_subscription(String, self.sign_facts_topic, self.sign_facts_cb, 10)
        self.create_subscription(String, self.lane_vision_topic, self.lane_vision_cb, 10)

        self.timer = self.create_timer(1.0 / max(1.0, self.tick_hz), self.tick)
        self.get_logger().info(
            f"clean_phase1_driver_node ready: {self.host}:{self.port} command={self.command_topic}"
        )

    def mission_cb(self, msg):
        try:
            self.latest_mission = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Mission JSON parse failed: {exc}")

    def sign_facts_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Sign facts JSON parse failed: {exc}")
            return

        facts = data.get("facts", data if isinstance(data, list) else [data])
        now = time.time()

        for fact in facts:
            if not isinstance(fact, dict):
                continue
            rule = str(fact.get("rule", "")).strip()
            confidence = float(fact.get("confidence", 0.0) or 0.0)
            active = bool(fact.get("active", True))
            if active:
                self.observe_sign_rule(
                    rule=rule,
                    confidence=confidence,
                    source="facts_topic",
                    now=now,
                    actor_id=fact.get("actor_id"),
                    distance_m=fact.get("distance_m"),
                )

    def lane_vision_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Lane vision JSON parse failed: {exc}")
            return

        now = time.time()
        self.latest_lane_vision = data
        self.latest_lane_vision_s = now
        if (
            bool(data.get("valid", False))
            and float(data.get("confidence", 0.0) or 0.0) >= 0.75
            and str(data.get("source", "")) == "two_lines_stable"
        ):
            self.last_reliable_lane_vision = data
            self.last_reliable_lane_vision_s = now

    def find_ego(self):
        now = time.time()

        if self.ego is not None:
            try:
                if self.ego.is_alive:
                    return self.ego
            except Exception:
                self.ego = None

        if now - self.last_ego_lookup_s < 1.0:
            return self.ego

        self.last_ego_lookup_s = now
        for vehicle in self.world.get_actors().filter("vehicle.*"):
            if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                self.ego = vehicle
                self.get_logger().info(f"Clean driver ego found: id={vehicle.id}")
                return self.ego

        return None

    def speed_mps(self, ego):
        velocity = ego.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def target_from_mission(self):
        target = self.latest_mission.get("objective_target") or self.latest_mission.get("target")
        if not isinstance(target, dict):
            return None

        x = target.get("carla_x")
        y = target.get("carla_y")
        if x is None or y is None:
            return None

        return float(x), float(y), str(target.get("name", "") or "")

    def waypoint_options(self, ego_wp):
        options = ego_wp.next(self.lookahead_m)
        if options:
            return list(options)
        return [ego_wp]

    def turn_label(self, ego_wp, option_wp):
        base_loc = ego_wp.transform.location
        option_loc = option_wp.transform.location
        vec = self.carla.Location(
            x=option_loc.x - base_loc.x,
            y=option_loc.y - base_loc.y,
            z=0.0,
        )
        forward = ego_wp.transform.get_forward_vector()
        right = ego_wp.transform.get_right_vector()
        forward_m = dot_location(vec, forward)
        right_m = dot_location(vec, right)

        if abs(right_m) < 2.0 or abs(right_m) < abs(forward_m) * 0.30:
            return "straight"
        if right_m > 0.0:
            return "right"
        return "left"

    def sign_rule_allows(self, rule, label):
        if not rule:
            return True
        if rule == "no_right_turn" and label == "right":
            return False
        if rule == "no_left_turn" and label == "left":
            return False
        if rule == "go_straight" and label != "straight":
            return False
        if rule == "mandatory_right" and label != "right":
            return False
        if rule == "mandatory_left" and label != "left":
            return False
        return True

    def waypoint_key(self, wp):
        return (
            int(getattr(wp, "road_id", 0)),
            int(getattr(wp, "lane_id", 0)),
            int(getattr(wp, "section_id", 0)),
        )

    def mission_direction_score(self, ego, option_wp):
        target = self.target_from_mission()
        if target is None:
            return 0.0

        ego_loc = ego.get_location()
        goal_x, goal_y, _ = target
        option_loc = option_wp.transform.location

        to_goal = math.atan2(goal_y - ego_loc.y, goal_x - ego_loc.x)
        to_option = math.atan2(option_loc.y - ego_loc.y, option_loc.x - ego_loc.x)
        return abs(norm_deg(math.degrees(to_option - to_goal)))

    def candidate_lateral_jump(self, prev_wp, candidate_wp):
        prev_loc = prev_wp.transform.location
        cand_loc = candidate_wp.transform.location
        vec = self.carla.Location(
            x=cand_loc.x - prev_loc.x,
            y=cand_loc.y - prev_loc.y,
            z=0.0,
        )
        return dot_location(vec, prev_wp.transform.get_right_vector())

    def score_candidate(self, ego, prev_wp, candidate_wp, active_sign_rule, locked_branch):
        label = self.turn_label(prev_wp, candidate_wp)
        if not self.sign_rule_allows(active_sign_rule, label):
            return None

        heading_diff = abs(norm_deg(
            candidate_wp.transform.rotation.yaw - prev_wp.transform.rotation.yaw
        ))
        mission_score = self.mission_direction_score(ego, candidate_wp)
        lateral_jump = abs(self.candidate_lateral_jump(prev_wp, candidate_wp))
        same_direction_penalty = 0.0 if heading_diff <= 95.0 else 80.0
        lane_penalty = 0.0

        try:
            if candidate_wp.lane_type != self.carla.LaneType.Driving:
                lane_penalty = 100.0
        except Exception:
            lane_penalty = 0.0

        branch_bonus = 0.0
        if locked_branch is not None:
            if label == locked_branch.get("label"):
                branch_bonus -= 25.0
            if self.waypoint_key(candidate_wp) == locked_branch.get("key"):
                branch_bonus -= 25.0

        score = (
            heading_diff * 0.55
            + mission_score * 0.85
            + lateral_jump * 18.0
            + same_direction_penalty
            + lane_penalty
            + branch_bonus
        )

        if label != "straight":
            score += 3.0

        return {
            "score": score,
            "label": label,
            "heading_diff": heading_diff,
            "mission_score": mission_score,
            "lateral_jump": lateral_jump,
            "key": self.waypoint_key(candidate_wp),
            "wp": candidate_wp,
        }

    def summarize_candidates(self, scored):
        parts = []
        for item in scored[:4]:
            parts.append(
                f"{item['label']}:{item['score']:.1f}/h{item['heading_diff']:.0f}"
                f"/m{item['mission_score']:.0f}/lat{item['lateral_jump']:.2f}"
            )
        return "|".join(parts) if parts else "none"

    def select_next_waypoint(self, ego, prev_wp, active_sign_rule, locked_branch):
        options = list(prev_wp.next(self.path_step_m) or [])
        if not options:
            return prev_wp, "straight", "no_next_waypoint", "none", 0.0

        scored = []
        for option in options:
            item = self.score_candidate(ego, prev_wp, option, active_sign_rule, locked_branch)
            if item is not None:
                scored.append(item)

        if not scored:
            for option in options:
                item = self.score_candidate(ego, prev_wp, option, None, locked_branch)
                if item is not None:
                    scored.append(item)

        if not scored:
            return options[0], self.turn_label(prev_wp, options[0]), "unscored_fallback", "none", 0.0

        scored.sort(key=lambda item: item["score"])
        best = scored[0]
        return (
            best["wp"],
            best["label"],
            "mission_branch_score",
            self.summarize_candidates(scored),
            float(best["lateral_jump"]),
        )

    def probe_junction_ahead(self, ego_wp):
        current = ego_wp
        travelled = 0.0

        if ego_wp.is_junction:
            return True

        while travelled < self.junction_probe_m:
            options = list(current.next(self.path_step_m) or [])
            if not options:
                return False
            if len(options) > 1:
                return True
            nxt = options[0]
            if nxt.is_junction:
                return True
            current = nxt
            travelled += self.path_step_m

        return False

    def adaptive_lookahead_m(self, speed_mps, mode):
        if speed_mps < 1.5:
            lookahead = 7.5
        elif speed_mps < 3.0:
            lookahead = 10.0
        elif speed_mps < 5.0:
            lookahead = 13.5
        else:
            lookahead = 18.0

        if mode == "junction":
            return clamp(lookahead, 10.0, 14.0)
        return lookahead

    def path_target_from_waypoints_with_progress(
        self,
        ego,
        waypoints,
        lookahead_m,
        min_forward_m=1.0,
    ):
        if not waypoints:
            return None, -1, -1, 0.0

        ego_loc = ego.get_location()
        closest_i = 0
        closest_d = float("inf")
        ego_forward = ego.get_transform().get_forward_vector()
        any_near_forward = False

        for i, wp in enumerate(waypoints):
            loc = wp.transform.location
            rel = self.carla.Location(x=loc.x - ego_loc.x, y=loc.y - ego_loc.y, z=0.0)
            ahead = dot_location(rel, ego_forward)
            dist = distance_2d(loc, ego_loc)
            if ahead > -2.0 and dist < closest_d:
                any_near_forward = True
                closest_i = i
                closest_d = dist

        if not any_near_forward:
            for i, wp in enumerate(waypoints):
                dist = distance_2d(wp.transform.location, ego_loc)
                if dist < closest_d:
                    closest_i = i
                    closest_d = dist

        total = 0.0
        last_loc = ego_loc
        best_forward_i = -1
        best_forward_m = 0.0
        for i in range(closest_i, len(waypoints)):
            wp = waypoints[i]
            loc = wp.transform.location
            total += distance_2d(last_loc, loc)
            rel = self.carla.Location(x=loc.x - ego_loc.x, y=loc.y - ego_loc.y, z=0.0)
            forward_m = dot_location(rel, ego_forward)
            if forward_m > min_forward_m:
                best_forward_i = i
                best_forward_m = forward_m
            if total >= lookahead_m and forward_m > min_forward_m:
                return wp, i, closest_i, forward_m
            last_loc = loc

        if best_forward_i >= 0:
            return waypoints[best_forward_i], best_forward_i, closest_i, best_forward_m

        return None, -1, closest_i, 0.0

    def path_target_from_waypoints(self, ego, waypoints, lookahead_m):
        target_wp, _, _, _ = self.path_target_from_waypoints_with_progress(
            ego,
            waypoints,
            lookahead_m,
        )
        if target_wp is not None:
            return target_wp
        return waypoints[-1] if waypoints else None

    def path_distance_to_index(self, waypoints, index):
        if not waypoints or index <= 0:
            return 0.0

        distance = 0.0
        last_loc = waypoints[0].transform.location
        for wp in waypoints[1:min(index + 1, len(waypoints))]:
            loc = wp.transform.location
            distance += distance_2d(last_loc, loc)
            last_loc = loc
        return distance

    def build_path_from(self, ego, start_wp, mode, active_sign_rule, locked_branch, lookahead_m):
        waypoints = [start_wp]
        current = start_wp
        branch = locked_branch.get("label") if locked_branch else "straight"
        reason = "lane_center_follow" if mode == "lane" else "junction_branch_select"
        candidates_summary = "none"
        max_lateral_jump = 0.0

        steps = max(2, int(self.path_length_m / max(0.5, self.path_step_m)))
        for _ in range(steps):
            nxt, label, step_reason, candidates, lateral_jump = self.select_next_waypoint(
                ego,
                current,
                active_sign_rule,
                locked_branch,
            )
            if candidates != "none":
                candidates_summary = candidates
            if branch == "straight" and label != "straight":
                branch = label
            max_lateral_jump = max(max_lateral_jump, abs(lateral_jump))
            waypoints.append(nxt)
            current = nxt
            if step_reason != "mission_branch_score" and reason == "junction_branch_select":
                reason = step_reason

        target_wp, target_index, progress_index, _ = self.path_target_from_waypoints_with_progress(
            ego,
            waypoints,
            lookahead_m,
        )
        if target_wp is None:
            target_wp = waypoints[-1]
            target_index = len(waypoints) - 1
        return LocalPath(
            mode=mode,
            branch=branch,
            waypoints=waypoints,
            target_wp=target_wp,
            lookahead_m=lookahead_m,
            candidates=candidates_summary,
            reason=reason,
            max_lateral_jump_m=max_lateral_jump,
            locked=locked_branch is not None,
            target_index=target_index,
            progress_index=progress_index,
            locked_path_len=len(waypoints) if locked_branch is not None else 0,
        )

    def activate_junction_branch(self, local_path, ego_wp, now):
        self.active_junction_branch = {
            "label": local_path.branch,
            "key": self.waypoint_key(local_path.waypoints[1])
            if len(local_path.waypoints) > 1 else self.waypoint_key(ego_wp),
            "path": local_path.waypoints,
            "created_s": now,
            "started_s": now,
            "locked_until_s": now + self.junction_lock_s,
            "last_junction_s": now,
            "last_progress_s": now,
            "last_target_update_s": now,
            "last_target_index": local_path.target_index,
            "last_progress_index": local_path.progress_index,
            "last_forward_progress_m": 0.0,
            "behind_count": 0,
            "stale": False,
            "release_reason": None,
        }

    def release_junction_branch(self, reason):
        if self.active_junction_branch is not None:
            self.active_junction_branch["release_reason"] = reason
        self.last_junction_release_reason = reason
        if reason == "junction_exit_hold_elapsed":
            now = time.time()
            self.junction_exit_ramp_start_s = now
            self.junction_exit_ramp_until_s = now + 1.8
        self.active_junction_branch = None

    def update_junction_lock(self, ego_wp, now, in_junction_context):
        if self.active_junction_branch is None:
            return None

        # Straight branch'i kilitli tutma. Düz kavşakta her tick güncel lane/path hedefi seçilmeli.
        if self.active_junction_branch.get("label") == "straight":
            self.release_junction_branch("straight_branch_no_lock_release")
            return None

        if in_junction_context:
            self.active_junction_branch["last_junction_s"] = now
            return self.active_junction_branch

        last_junction_s = self.active_junction_branch.get("last_junction_s", now)
        locked_until_s = self.active_junction_branch.get("locked_until_s", 0.0)
        if now < locked_until_s or now - last_junction_s < self.junction_exit_hold_s:
            return self.active_junction_branch

        self.release_junction_branch("junction_exit_hold_elapsed")
        return None

    def select_local_path(self, ego, ego_wp, active_sign_rule, speed_mps):
        now = time.time()
        in_junction_context = self.probe_junction_ahead(ego_wp)
        locked_branch = self.update_junction_lock(ego_wp, now, in_junction_context)
        mode = "junction" if in_junction_context or locked_branch is not None else "lane"
        lookahead_m = self.adaptive_lookahead_m(speed_mps, mode)

        if locked_branch is not None:
            locked_path = locked_branch.get("path", [])
            target_wp, target_index, progress_index, target_forward_m = (
                self.path_target_from_waypoints_with_progress(
                    ego,
                    locked_path,
                    lookahead_m,
                    min_forward_m=1.0,
                )
            )
            if target_wp is None:
                locked_branch["behind_count"] = int(locked_branch.get("behind_count", 0)) + 1
                self.release_junction_branch("locked_branch_no_forward_target")
                locked_branch = None
            else:
                last_target_index = int(locked_branch.get("last_target_index", -1))
                last_progress_index = int(locked_branch.get("last_progress_index", -1))
                progress_m = self.path_distance_to_index(locked_path, progress_index)
                last_progress_m = float(locked_branch.get("last_forward_progress_m", 0.0) or 0.0)
                made_progress = (
                    target_index > last_target_index
                    or progress_index > last_progress_index
                    or progress_m > last_progress_m + 0.5
                )

                if made_progress:
                    locked_branch["last_progress_s"] = now
                    locked_branch["last_progress_index"] = progress_index
                    locked_branch["last_forward_progress_m"] = progress_m
                    locked_branch["behind_count"] = 0
                if target_index != last_target_index:
                    locked_branch["last_target_update_s"] = now
                    locked_branch["last_target_index"] = target_index

                near_path_end = (
                    target_index >= max(0, len(locked_path) - 2)
                    and target_forward_m < max(5.5, 0.65 * lookahead_m)
                )
                target_age_s = now - float(locked_branch.get("last_target_update_s", now))
                stale = now - float(locked_branch.get("last_progress_s", now)) > 1.5

                if stale:
                    locked_branch["stale"] = True
                    self.release_junction_branch("locked_branch_stale_target_behind")
                    locked_branch = None
                elif near_path_end and target_age_s > 0.8:
                    self.release_junction_branch("locked_branch_near_path_end")
                    locked_branch = None
                else:
                    return LocalPath(
                        mode="junction",
                        branch=locked_branch.get("label", "straight"),
                        waypoints=locked_path,
                        target_wp=target_wp,
                        lookahead_m=lookahead_m,
                        candidates="locked",
                        reason="active_junction_branch_locked",
                        max_lateral_jump_m=0.0,
                        locked=True,
                        target_index=target_index,
                        progress_index=progress_index,
                        locked_path_len=len(locked_path),
                        stale=False,
                        release_reason=locked_branch.get("release_reason"),
                    )

        if locked_branch is None:
            mode = "junction" if in_junction_context else "lane"
            lookahead_m = self.adaptive_lookahead_m(speed_mps, mode)

        local_path = self.build_path_from(
            ego,
            ego_wp,
            mode,
            active_sign_rule,
            locked_branch,
            lookahead_m,
        )

        if mode == "junction" and self.active_junction_branch is None:
            # Straight junction için branch lock gereksiz ve zararlı.
            # Logda straight branch uzun süre locked kalıp target sonuna dayanıyor,
            # sonra araç geç kalmış düzeltmeyle şeritten çıkıyor.
            # Sadece gerçek left/right dönüşlerde path lock kullan.
            if local_path.branch != "straight":
                self.activate_junction_branch(local_path, ego_wp, now)
                local_path.locked = True
                local_path.locked_path_len = len(local_path.waypoints)
            else:
                local_path.locked = False
                local_path.locked_path_len = 0
                local_path.reason = "junction_straight_no_lock"
        elif mode == "lane":
            self.stable_lane_path = local_path

        return local_path

    def target_errors(self, ego, ego_wp, target_wp):
        ego_transform = ego.get_transform()
        ego_loc = ego_transform.location
        target_loc = target_wp.transform.location

        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()
        to_target = self.carla.Location(
            x=target_loc.x - ego_loc.x,
            y=target_loc.y - ego_loc.y,
            z=0.0,
        )

        forward_m = dot_location(to_target, forward)
        right_m = dot_location(to_target, right)
        heading_error_rad = math.atan2(right_m, max(0.1, forward_m))
        heading_error_deg = math.degrees(heading_error_rad)

        lane_right = ego_wp.transform.get_right_vector()
        lane_center = ego_wp.transform.location
        center_vec = self.carla.Location(
            x=ego_loc.x - lane_center.x,
            y=ego_loc.y - lane_center.y,
            z=0.0,
        )
        offset_m = dot_location(center_vec, lane_right)
        return forward_m, right_m, offset_m, heading_error_rad, heading_error_deg

    def reject_target_reason(self, ego, ego_wp, local_path, forward_m, right_m, heading_error_deg):
        if local_path.locked and local_path.stale:
            return "locked_branch_stale_target_behind"

        if forward_m <= 1.0:
            return "target_behind"

        if abs(heading_error_deg) > 60.0:
            return "heading_error_gt_60"

        if abs(heading_error_deg) > 45.0:
            return "heading_error_gt_45"

        # Straight branch içinde 20-30 derece hedef sapması normal değildir.
        # Önceki logda branch straight iken heading_error=28.9 kabul edildi ve araç şerit dışına çıktı.
        if local_path.mode == "junction" and local_path.branch == "straight" and abs(heading_error_deg) > 16.0:
            return "straight_junction_heading_gt_16"

        if local_path.mode == "junction" and abs(heading_error_deg) > 28.0:
            return "junction_heading_gt_28"

        if local_path.max_lateral_jump_m > 1.0:
            return "path_lateral_jump"

        if (
            local_path.mode == "lane"
            and self.waypoint_key(local_path.target_wp) != self.waypoint_key(ego_wp)
            and (
                abs(heading_error_deg) > 8.0
                or local_path.max_lateral_jump_m > 0.50
            )
        ):
            return "lane_target_road_jump"

        raw_steer = self.lane_heading_gain * math.radians(heading_error_deg)
        if (
            abs(raw_steer) > 0.28
            and abs(self.last_raw_steer) > 0.28
            and raw_steer * self.last_raw_steer < 0.0
            and abs(raw_steer - self.last_raw_steer) > 0.55
        ):
            return "steering_sign_flip"

        return None

    def stable_fallback_target(self, ego):
        for target in [self.last_accepted_target_wp]:
            if target is None:
                continue
            ego_loc = ego.get_location()
            rel = self.carla.Location(
                x=target.transform.location.x - ego_loc.x,
                y=target.transform.location.y - ego_loc.y,
                z=0.0,
            )
            ahead = dot_location(rel, ego.get_transform().get_forward_vector())
            if 1.0 < ahead <= max(2.0, self.path_length_m):
                return target
        return None

    def map_lane_fallback_target(self, ego, ego_wp, lookahead_m):
        waypoints = [ego_wp]
        current = ego_wp
        steps = max(2, int(max(6.0, min(lookahead_m, 10.0)) / max(0.5, self.path_step_m)))

        for _ in range(steps):
            options = list(current.next(self.path_step_m) or [])
            if not options:
                break

            current_yaw = current.transform.rotation.yaw

            def score(option):
                yaw_diff = abs(norm_deg(option.transform.rotation.yaw - current_yaw))
                lane_change = 0.0 if self.waypoint_key(option) == self.waypoint_key(current) else 20.0
                return yaw_diff + lane_change

            current = min(options, key=score)
            waypoints.append(current)

        target_wp, _, _, _ = self.path_target_from_waypoints_with_progress(
            ego,
            waypoints,
            max(4.0, min(lookahead_m, 8.0)),
            min_forward_m=1.0,
        )
        return target_wp or (waypoints[-1] if waypoints else ego_wp)

    def resolve_target(self, ego, ego_wp, local_path):
        target_wp = local_path.target_wp
        forward_m, right_m, lane_offset, heading_rad, heading_deg = self.target_errors(
            ego,
            ego_wp,
            target_wp,
        )
        reject_reason = self.reject_target_reason(
            ego,
            ego_wp,
            local_path,
            forward_m,
            right_m,
            heading_deg,
        )

        primary_forward_m = forward_m
        primary_heading_deg = heading_deg
        fallback_used = False
        accepted = reject_reason is None
        if not accepted:
            original_reject_reason = reject_reason
            if local_path.locked and original_reject_reason in {
                "target_behind",
                "heading_error_gt_45",
                "heading_error_gt_60",
                "locked_branch_stale_target_behind",
                "straight_junction_heading_gt_16",
                "junction_heading_gt_28",
            }:
                release_reason = "emergency_path_recovery_heading" if abs(primary_heading_deg) > 75.0 else original_reject_reason
                local_path.release_reason = release_reason
                local_path.stale = original_reject_reason == "locked_branch_stale_target_behind"
                self.release_junction_branch(release_reason)

            force_map_fallback = original_reject_reason in {
                "target_behind",
                "heading_error_gt_45",
                "heading_error_gt_60",
                "locked_branch_stale_target_behind",
                "straight_junction_heading_gt_16",
                "junction_heading_gt_28",
            }
            fallback = None if force_map_fallback else self.stable_fallback_target(ego)
            if fallback is None:
                fallback = self.map_lane_fallback_target(ego, ego_wp, local_path.lookahead_m)

            if fallback is not None:
                target_wp = fallback
                forward_m, right_m, lane_offset, heading_rad, heading_deg = self.target_errors(
                    ego,
                    ego_wp,
                    target_wp,
                )
                fallback_reject = self.reject_target_reason(
                    ego,
                    ego_wp,
                    LocalPath(
                        mode="lane",
                        branch="straight",
                        waypoints=[ego_wp, fallback],
                        target_wp=fallback,
                        lookahead_m=local_path.lookahead_m,
                        candidates="fallback",
                        reason="map_lane_centerline_fallback",
                    ),
                    forward_m,
                    right_m,
                    heading_deg,
                )
                if fallback_reject is None:
                    fallback_used = True
                    reject_reason = original_reject_reason
                else:
                    target_wp = ego_wp
                    forward_m, right_m, lane_offset, heading_rad, heading_deg = self.target_errors(
                        ego,
                        ego_wp,
                        target_wp,
                    )
                    fallback_used = True
                    reject_reason = f"{original_reject_reason}:fallback_invalid_{fallback_reject}"
            else:
                target_wp = ego_wp
                forward_m, right_m, lane_offset, heading_rad, heading_deg = self.target_errors(
                    ego,
                    ego_wp,
                    target_wp,
                )
                reject_reason = f"{original_reject_reason}:no_fallback"

        if accepted:
            self.last_accepted_target_wp = target_wp
            self.last_accepted_path = local_path

        target_age_s = 0.0
        if self.active_junction_branch is not None:
            target_age_s = time.time() - float(
                self.active_junction_branch.get("last_target_update_s", time.time())
            )
        self.last_target_diag = {
            "forward_projection_m": primary_forward_m,
            "primary_heading_error_deg": primary_heading_deg,
            "target_age_s": target_age_s,
            "locked_target_index": local_path.target_index,
            "locked_path_len": local_path.locked_path_len,
            "fallback_used": fallback_used,
        }

        return target_wp, accepted, reject_reason, forward_m, right_m, lane_offset, heading_rad, heading_deg

    def lane_control(self, lane_offset, heading_error_rad, heading_error_deg, mode):
        if abs(lane_offset) < 0.05 and abs(heading_error_deg) < 2.0:
            raw_steer = 0.0
        else:
            pure_pursuit = 0.72 * heading_error_rad
            heading_trim = 0.06 * heading_error_rad
            offset_trim = -self.lane_offset_gain * lane_offset
            raw_steer = (
                pure_pursuit
                + heading_trim
                + offset_trim
            )

        abs_offset = abs(lane_offset)
        abs_heading = abs(heading_error_deg)
        if mode == "junction":
            limit = 0.32 if abs_heading > 10.0 or abs_offset > 0.35 else 0.22
        else:
            limit = 0.18
        if abs_offset > 1.20 or abs_heading > 25.0:
            limit = min(0.42, self.max_steer_command)

        limited_steer = clamp(raw_steer, -limit, limit)

        if (
            self.last_driver_steer * limited_steer < 0.0
            and abs_offset < 0.12
            and abs_heading < 2.5
        ):
            limited_steer = 0.0

        alpha = 0.10 if mode == "junction" else 0.075
        rate_limit_per_tick = 0.028 if mode == "junction" else 0.018
        filtered = (1.0 - alpha) * self.last_driver_steer + alpha * limited_steer
        steer = rate_limit(
            self.last_driver_steer,
            filtered,
            rate_limit_per_tick,
        )

        self.last_raw_steer = raw_steer
        self.last_limited_steer = limited_steer
        self.last_steer_rate_limited = abs(steer - filtered) > 1e-6
        self.last_driver_steer = steer
        return steer, raw_steer, limited_steer, self.last_steer_rate_limited

    def current_lane_vision(self):
        now = time.time()

        def held_reliable():
            if self.last_reliable_lane_vision is None:
                return None
            age = now - self.last_reliable_lane_vision_s
            if age > 1.05:
                return None
            held = dict(self.last_reliable_lane_vision)
            base_conf = float(held.get("confidence", 0.0) or 0.0)
            held["confidence"] = round(max(0.45, min(base_conf, base_conf * (1.0 - 0.45 * age / 1.05))), 3)
            held["held_age_s"] = round(age, 3)
            held["source"] = "two_lines_stable"
            held["reason"] = f"driver_held_reliable_{age:.2f}s"
            return held

        if self.latest_lane_vision is not None and now - self.latest_lane_vision_s <= 0.55:
            if bool(self.latest_lane_vision.get("valid", False)):
                return self.latest_lane_vision, "fresh"
            held = held_reliable()
            if held is not None:
                return held, "held_reliable"
            return self.latest_lane_vision, "fresh"

        held = held_reliable()
        if held is not None:
            return held, "held_reliable"
        return None, "missing"

    def vision_steering(self, vision):
        if vision is None or not bool(vision.get("valid", False)):
            return 0.0, 0.0, 0.0, "vision_invalid"

        offset_norm = float(vision.get("vision_offset_norm", 0.0) or 0.0)
        heading_deg = float(vision.get("vision_heading_error_deg", 0.0) or 0.0)
        conf = float(vision.get("confidence", 0.0) or 0.0)
        source = str(vision.get("source", ""))

        # Kamera optik merkezi araç merkeziyle aynı olmak zorunda değil.
        # Logda başlangıçta lane_center_px≈332 iken sistem bunu sağa hata sanıp aracı sağa çekti.
        # İlk güvenilir düz iki-çizgi gözlemlerinden küçük bias öğren.
        if (
            source == "two_lines_stable"
            and conf >= 0.75
            and abs(heading_deg) < 6.0
            and abs(offset_norm) < 0.08
            and self.vision_bias_samples < 30
        ):
            if self.vision_offset_bias_norm is None:
                self.vision_offset_bias_norm = offset_norm
            else:
                self.vision_offset_bias_norm = (
                    0.92 * self.vision_offset_bias_norm + 0.08 * offset_norm
                )
            self.vision_bias_samples += 1

        corrected_offset = offset_norm - float(self.vision_offset_bias_norm or 0.0)

        if abs(corrected_offset) < 0.008 and abs(heading_deg) < 2.0:
            raw = 0.0
        else:
            # Heading etkisini düşük tut; ana karar çizgi merkezidir.
            raw = 1.05 * corrected_offset + 0.0032 * heading_deg

        limited = clamp(raw, -0.16, 0.16)
        return limited, corrected_offset, heading_deg, "vision_lane_center_bias_corrected"

    def fuse_lane_steering(self, map_steer, vision, mode, lane_offset=0.0, target_heading_error_deg=0.0):
        vision_steer, offset_norm, heading_deg, vision_reason = self.vision_steering(vision)
        conf = float(vision.get("confidence", 0.0) or 0.0) if vision is not None else 0.0
        valid = bool(vision.get("valid", False)) if vision is not None else False
        vision_source = str(vision.get("source", "missing")) if vision is not None else "missing"
        rejected_reason = vision.get("rejected_reason") if vision is not None else None

        two_lines_stable = vision_source == "two_lines_stable"
        held_or_last = vision_source == "last_valid" or bool(vision.get("held_reliable", False)) if vision is not None else False

        # Şüpheli vision durumları
        heading_rejected = abs(heading_deg) > 10.0
        recovery_map_priority = (
            abs(lane_offset) > 0.45
            or abs(target_heading_error_deg) > 7.0
            or mode == "junction"
        )

        conflict = (
            valid
            and conf >= 0.45
            and abs(vision_steer - map_steer) > 0.12
            and abs(vision_steer) > 0.025
            and abs(map_steer) > 0.025
            and vision_steer * map_steer < 0.0
        )

        target_vision_weight = 0.0
        reason = "vision_low_conf_map_fallback"

        # Sadece gerçekten sağlam iki çizgi normal yolda vision otoritesi alabilir.
        if (
            valid
            and two_lines_stable
            and conf >= 0.82
            and not heading_rejected
            and not recovery_map_priority
            and not conflict
        ):
            target_vision_weight = 0.70
            reason = vision_reason
        elif (
            valid
            and two_lines_stable
            and conf >= 0.65
            and not heading_rejected
            and not conflict
        ):
            # Junction/recovery içinde bile vision sadece düşük destek olabilir.
            target_vision_weight = 0.20 if recovery_map_priority else 0.45
            reason = "safe_vision_map_blend"
        elif held_or_last:
            target_vision_weight = 0.0
            reason = "held_last_valid_diagnostic_only"
        elif heading_rejected:
            target_vision_weight = 0.0
            reason = "vision_rejected_heading"
        elif valid and not two_lines_stable:
            target_vision_weight = 0.0
            reason = "vision_source_not_stable_map_fallback"

        if conflict:
            target_vision_weight = 0.0
            reason = "lane_vision_map_conflict"

        # Recovery veya şüpheli durumda vision ağırlığını yumuşatmadan hızlı boşalt.
        if target_vision_weight <= 0.0 or recovery_map_priority or heading_rejected or conflict:
            self.smoothed_vision_weight = 0.0
        else:
            max_weight_delta = 0.15 if mode == "junction" else 0.20
            self.smoothed_vision_weight = rate_limit(
                self.smoothed_vision_weight,
                target_vision_weight,
                max_weight_delta,
            )

        vision_weight = clamp(self.smoothed_vision_weight, 0.0, 0.70)
        map_weight = 1.0 - vision_weight
        desired = vision_weight * vision_steer + map_weight * map_steer

        if vision_weight >= 0.55:
            source = "vision"
        elif vision_weight >= 0.08:
            source = "blend"
        else:
            source = "map"

        limit = 0.30 if mode == "junction" else 0.18
        desired = clamp(desired, -limit, limit)

        # Büyük offsette düzeltmeyi fazla boğma; aksi halde araç şeritten çıkıp hız düşürüyor.
        if abs(lane_offset) > 0.55 or abs(target_heading_error_deg) > 8.0:
            desired = map_steer
            source = "map"
            vision_weight = 0.0
            map_weight = 1.0
            reason = "map_priority_lane_recovery"

        if (
            self.last_driver_steer * desired < 0.0
            and abs(lane_offset) < 0.12
            and abs(target_heading_error_deg) < 2.5
        ):
            desired = 0.0

        max_delta = 0.045 if (mode == "junction" or abs(lane_offset) > 0.45) else 0.025
        final_steer = rate_limit(self.last_driver_steer, desired, max_delta)
        rate_limited = abs(final_steer - desired) > 1e-6
        self.last_driver_steer = final_steer

        prev_lane_source = self.last_lane_source
        lane_source_changed = source != prev_lane_source
        if lane_source_changed:
            self.last_lane_source = source
            self.last_lane_source_change_s = time.time()

        return {
            "source": source,
            "lane_source_prev": prev_lane_source,
            "lane_source_changed": lane_source_changed,
            "vision_conf": conf,
            "vision_valid": valid,
            "vision_steer": vision_steer,
            "map_steer": map_steer,
            "final_steer": final_steer,
            "offset_norm": offset_norm,
            "heading_deg": heading_deg,
            "conflict": conflict,
            "vision_source": vision_source,
            "vision_rejected_reason": rejected_reason,
            "vision_weight": vision_weight,
            "map_weight": map_weight,
            "vision_weight_target": target_vision_weight,
            "reason": reason,
            "rate_limited": rate_limited,
        }

    def vision_speed_cap(self, fusion):
        if (
            fusion["source"] == "map"
            and not fusion["vision_valid"]
            and str(fusion.get("vision_source", "")).startswith(("one_line_rejected", "two_lines_rejected"))
        ):
            return 2.5, "vision_rejected_map_slow"
        if fusion["source"] == "map" and not fusion["vision_valid"]:
            return self.cruise_speed_mps, "map_fallback_no_vision"

        abs_offset = abs(float(fusion["offset_norm"]))
        abs_heading = abs(float(fusion["heading_deg"]))
        cap = self.cruise_speed_mps
        reason = "vision_ok"

        if fusion["conflict"]:
            return 1.5, "lane_vision_map_conflict"
        if fusion["vision_valid"] and fusion["vision_conf"] < 0.35:
            return 3.0, "vision_low_conf_map_fallback"
        if abs_offset > 0.10 or abs_heading > 8.0:
            cap = min(cap, 2.5)
            reason = "vision_error_high"
        elif abs_offset > 0.06 or abs_heading > 5.0:
            cap = min(cap, 4.0)
            reason = "vision_error_medium"

        if abs_offset > 0.16 or abs_heading > 14.0:
            cap = min(cap, 1.2)
            reason = "vision_recovery"

        return cap, reason

    def apply_turn_speed_caps(self, base_speed, heading_error_deg, lane_offset, mode, target_accepted):
        target_speed = base_speed
        abs_heading = abs(heading_error_deg)
        abs_offset = abs(lane_offset)

        if mode == "junction":
            target_speed = min(target_speed, self.turn_speed_mps)

        if abs_heading > 15.0:
            target_speed = min(target_speed, 2.5)
        if abs_heading > 30.0:
            target_speed = min(target_speed, 1.8)
        if abs_heading > 45.0:
            target_speed = min(target_speed, 1.2)
        if abs_offset > 0.8:
            target_speed = min(target_speed, 1.5)
        if abs_offset > 1.5:
            target_speed = min(target_speed, 0.8)
        if not target_accepted:
            target_speed = min(target_speed, 1.2)

        return target_speed

    def lane_safety_speed_cap(self, mode, lane_offset, heading_error_deg, target_accepted):
        abs_offset = abs(lane_offset)
        abs_heading = abs(heading_error_deg)
        cap = self.cruise_speed_mps
        level = "ok"

        if mode == "junction":
            cap = min(cap, 2.3)
            level = "mild"
            if abs_offset > 0.35 or abs_heading > 6.0:
                cap = min(cap, 2.0)
                level = "medium"
            if abs_offset > 0.60 or abs_heading > 10.0:
                cap = min(cap, 1.5)
                level = "high"

        if abs_offset > 0.35 or abs_heading > 6.0:
            cap = min(cap, 4.0)
            level = "mild"
        if abs_offset > 0.55 or abs_heading > 10.0:
            cap = min(cap, 2.8)
            level = "medium"
        if abs_offset > 0.80 or abs_heading > 15.0:
            cap = min(cap, 1.8)
            level = "high"
        if abs_offset > 1.20 or abs_heading > 25.0:
            cap = min(cap, 1.0)
            level = "recovery"
        if abs_offset > 1.80:
            cap = min(cap, 0.5)
            level = "recovery"
        if not target_accepted:
            cap = min(cap, 1.2)
            level = "recovery"

        return cap, level

    def rate_limit_speed_return(self, target_speed, mode, lane_offset, heading_error_deg):
        stable_lane = (
            mode == "lane"
            and abs(lane_offset) < 0.18
            and abs(heading_error_deg) < 5.0
        )
        if stable_lane:
            return min(self.cruise_speed_mps, max(target_speed, self.cruise_speed_mps))
        return target_speed

    def observe_sign_rule(self, rule, confidence, source, now, actor_id=None, distance_m=None):
        supported = {
            "no_right_turn",
            "no_left_turn",
            "go_straight",
            "mandatory_right",
            "mandatory_left",
        }
        if rule not in supported or confidence < self.sign_min_confidence:
            return

        if rule not in self.sign_tracks:
            self.sign_tracks[rule] = SignTrack(
                rule=rule,
                confidence=confidence,
                first_seen_s=now,
                last_seen_s=now,
                hits=1,
                source=source,
                actor_id=actor_id,
                distance_m=distance_m,
            )
            return

        track = self.sign_tracks[rule]
        track.confidence = max(track.confidence * 0.65, confidence)
        track.last_seen_s = now
        track.hits += 1
        track.source = source
        track.actor_id = actor_id
        track.distance_m = distance_m

    def scan_carla_sign_actors(self, ego):
        now = time.time()
        ego_loc = ego.get_location()
        ego_forward = ego.get_transform().get_forward_vector()

        mapping = {
            "saga_donulmez": "no_right_turn",
            "sola_donulmez": "no_left_turn",
            "ileri_mecburi_yon": "go_straight",
            "saga_mecburi_yon": "mandatory_right",
            "sola_mecburi_yon": "mandatory_left",
        }

        try:
            actors = self.world.get_actors()
        except Exception:
            return

        for actor in actors:
            type_id = str(getattr(actor, "type_id", "")).lower()
            if "sign" not in type_id:
                continue

            rule = None
            for key, candidate in mapping.items():
                if key in type_id:
                    rule = candidate
                    break
            if rule is None:
                continue

            loc = actor.get_location()
            rel = self.carla.Location(x=loc.x - ego_loc.x, y=loc.y - ego_loc.y, z=0.0)
            ahead_m = dot_location(rel, ego_forward)
            if ahead_m < 0.0 or ahead_m > self.sign_detect_m:
                continue

            lateral = math.sqrt(max(0.0, distance_2d(loc, ego_loc) ** 2 - ahead_m ** 2))
            if lateral > 8.0:
                continue

            self.observe_sign_rule(rule, 0.80, "carla_actor", now, actor.id, ahead_m)

    def active_sign_rule(self):
        now = time.time()
        best = None
        for rule, track in list(self.sign_tracks.items()):
            age = now - track.last_seen_s
            if age > self.sign_hold_s:
                continue
            if track.hits < self.sign_min_hits:
                continue
            score = track.confidence - 0.02 * age
            if best is None or score > best[0]:
                best = (score, track)

        if best is None:
            return None, 0.0, None, "no_stable_sign_rule"
        track = best[1]
        return track.rule, track.confidence, track, f"{track.source}_stable"

    def state_name(self, state):
        if state is None:
            return "none"
        text = str(state).split(".")[-1].lower()
        if "red" in text:
            return "red"
        if "yellow" in text:
            return "yellow"
        if "green" in text:
            return "green"
        return "unknown"

    def light_stop_locations(self, light):
        locations = []

        getter = getattr(light, "get_stop_waypoints", None)
        if callable(getter):
            try:
                for wp in getter() or []:
                    locations.append(wp.transform.location)
            except Exception:
                pass

        if locations:
            return locations

        try:
            transform = light.get_transform()
            trigger = getattr(light, "trigger_volume", None)
            if trigger is not None:
                loc = transform.transform(trigger.location)
                locations.append(loc)
            else:
                locations.append(transform.location)
        except Exception:
            pass

        return locations

    def traffic_light_ahead(self, ego, ego_wp):
        ego_loc = ego.get_location()
        ego_forward = ego.get_transform().get_forward_vector()
        best = None

        try:
            lights = self.world.get_actors().filter("traffic.traffic_light*")
        except Exception:
            lights = []

        for light in lights:
            for loc in self.light_stop_locations(light):
                rel = self.carla.Location(x=loc.x - ego_loc.x, y=loc.y - ego_loc.y, z=0.0)
                ahead_m = dot_location(rel, ego_forward)
                if ahead_m < -1.5 or ahead_m > self.traffic_light_detect_m:
                    continue

                try:
                    stop_wp = self.map.get_waypoint(
                        loc,
                        project_to_road=True,
                        lane_type=self.carla.LaneType.Driving,
                    )
                except Exception:
                    stop_wp = None

                lane_match = False
                if stop_wp is not None:
                    lane_match = (
                        int(stop_wp.road_id) == int(ego_wp.road_id)
                        and int(stop_wp.lane_id) == int(ego_wp.lane_id)
                    )

                    if not lane_match:
                        yaw_diff = abs(norm_deg(stop_wp.transform.rotation.yaw - ego_wp.transform.rotation.yaw))
                        lateral = distance_2d(loc, ego_wp.transform.location)
                        lane_match = yaw_diff < 25.0 and lateral < max(7.0, ego_wp.lane_width * 1.7)

                if not lane_match:
                    continue

                if best is None or ahead_m < best["distance_m"]:
                    get_state = getattr(light, "get_state", None)
                    state = get_state() if callable(get_state) else getattr(light, "state", None)
                    best = {
                        "actor_id": int(light.id),
                        "state": self.state_name(state),
                        "distance_m": float(ahead_m),
                    }

        return best

    def apply_traffic_light_policy(self, base_speed, current_speed, light):
        now = time.time()
        decision = "GO"
        target_speed = base_speed
        brake_required = False
        emergency = False
        reason = "lane_follow"
        red_hold_active = False
        green_release_active = now < self.green_release_until_s

        if light is None:
            if self.red_hold_light_id is not None:
                self.red_hold_light_id = None
            return decision, target_speed, brake_required, emergency, reason, False, green_release_active

        state = light["state"]
        distance_m = light["distance_m"]
        stop_distance_m = distance_m - self.stop_before_line_m

        if state == "green":
            if self.red_hold_light_id == light["actor_id"]:
                self.red_hold_light_id = None
                self.green_release_until_s = now + self.green_release_s
                green_release_active = True
            if green_release_active:
                target_speed = max(target_speed, min(self.cruise_speed_mps, 4.8))
                reason = "green_release"
            return decision, target_speed, brake_required, emergency, reason, False, green_release_active

        if state == "red":
            if stop_distance_m < -1.0:
                reason = "red_stopline_passed_no_stop"
                return decision, target_speed, brake_required, emergency, reason, False, green_release_active

            if self.red_hold_light_id == light["actor_id"]:
                red_hold_active = True

            if stop_distance_m <= self.red_hard_brake_m or red_hold_active:
                decision = "STOP"
                target_speed = 0.0
                brake_required = True
                emergency = stop_distance_m <= 1.5 and current_speed > 1.0
                self.red_hold_light_id = light["actor_id"]
                reason = "red_stopline_hold"
                return decision, target_speed, brake_required, emergency, reason, True, green_release_active

            if stop_distance_m <= self.red_slowdown_m:
                decision = "SLOW"
                safe_speed = math.sqrt(max(0.2, 2.0 * 1.3 * stop_distance_m)) * 0.75
                target_speed = min(target_speed, clamp(safe_speed, 1.2, 4.2))
                reason = "red_smooth_approach"
                return decision, target_speed, brake_required, emergency, reason, False, green_release_active

            decision = "SLOW"
            target_speed = min(target_speed, 4.5)
            reason = "red_seen_far"
            return decision, target_speed, brake_required, emergency, reason, False, green_release_active

        if state == "yellow":
            if stop_distance_m < -1.0:
                reason = "yellow_stopline_passed_no_stop"
                return decision, target_speed, brake_required, emergency, reason, False, green_release_active

            comfortable_stop = (current_speed ** 2) / (2.0 * 2.2) + 1.5
            if stop_distance_m > comfortable_stop and stop_distance_m < self.red_slowdown_m:
                decision = "STOP"
                target_speed = 0.0
                brake_required = True
                reason = "yellow_can_stop"
                return decision, target_speed, brake_required, emergency, reason, False, green_release_active

            decision = "SLOW"
            target_speed = min(target_speed, self.caution_speed_mps)
            reason = "yellow_too_close_caution"
            return decision, target_speed, brake_required, emergency, reason, False, green_release_active

        if state == "unknown" and 0.0 <= stop_distance_m <= 10.0:
            decision = "SLOW"
            target_speed = min(target_speed, self.caution_speed_mps)
            reason = "unknown_light_close_caution"

        return decision, target_speed, brake_required, emergency, reason, False, green_release_active

    def maybe_log(self, key, text, interval_s=1.0):
        now = time.time()
        if now - self.last_log_s.get(key, 0.0) >= interval_s:
            self.last_log_s[key] = now
            self.get_logger().info(text)

    def tick(self):
        ego = self.find_ego()
        now = time.time()

        if ego is None:
            return

        ego_loc = ego.get_location()
        try:
            ego_wp = self.map.get_waypoint(
                ego_loc,
                project_to_road=True,
                lane_type=self.carla.LaneType.Driving,
            )
        except Exception as exc:
            self.get_logger().warn(f"CARLA waypoint lookup failed: {exc}")
            return

        if ego_wp is None:
            self.publish_command(
                decision="STOP",
                target_speed=0.0,
                steering_target=0.0,
                brake_required=True,
                emergency=False,
                reason="no_driving_lane_waypoint",
            )
            return

        self.scan_carla_sign_actors(ego)
        sign_rule, sign_conf, sign_track, sign_reason = self.active_sign_rule()
        current_speed = self.speed_mps(ego)
        local_path = self.select_local_path(ego, ego_wp, sign_rule, current_speed)
        (
            target_wp,
            target_accepted,
            reject_reason,
            forward_m,
            right_m,
            lane_offset,
            heading_error_rad,
            heading_error,
        ) = self.resolve_target(ego, ego_wp, local_path)
        map_steer, raw_steer, limited_steer, steer_rate_limited = self.lane_control(
            lane_offset,
            heading_error_rad,
            heading_error,
            local_path.mode,
        )
        lane_vision, lane_vision_status = self.current_lane_vision()
        fusion = self.fuse_lane_steering(map_steer, lane_vision, local_path.mode, lane_offset, heading_error)
        steer = fusion["final_steer"]

        target_speed = self.cruise_speed_mps
        lane_reason = local_path.reason
        yaw_delta = abs(norm_deg(target_wp.transform.rotation.yaw - ego_wp.transform.rotation.yaw))
        turn_label = local_path.branch

        if local_path.mode == "junction" or turn_label != "straight" or yaw_delta > 18.0:
            lane_reason = f"junction_or_turn_{turn_label}:{local_path.reason}"

        lane_speed_cap, lane_error_level = self.lane_safety_speed_cap(
            local_path.mode,
            lane_offset,
            heading_error,
            target_accepted,
        )
        primary_heading_error = abs(float(
            self.last_target_diag.get("primary_heading_error_deg", heading_error) or 0.0
        ))
        if not target_accepted and primary_heading_error > 45.0:
            lane_speed_cap = min(lane_speed_cap, 1.0)
            lane_error_level = "recovery"
        vision_cap, vision_cap_reason = self.vision_speed_cap(fusion)
        lane_speed_cap = min(lane_speed_cap, vision_cap)
        if vision_cap < self.cruise_speed_mps:
            if vision_cap <= 1.5:
                lane_error_level = "recovery"
            elif lane_error_level in {"ok", "mild"}:
                lane_error_level = "medium"
        target_speed = min(target_speed, lane_speed_cap)
        target_speed = self.rate_limit_speed_return(
            target_speed,
            local_path.mode,
            lane_offset,
            heading_error,
        )

        junction_exit_ramp_active = time.time() < self.junction_exit_ramp_until_s
        if junction_exit_ramp_active:
            ramp_elapsed = time.time() - self.junction_exit_ramp_start_s
            stable_after_junction = (
                local_path.mode == "lane"
                and target_accepted
                and abs(lane_offset) < 0.25
                and abs(heading_error) < 4.0
            )
            if stable_after_junction:
                if ramp_elapsed < 0.60:
                    target_speed = min(target_speed, 3.0)
                elif ramp_elapsed < 1.20:
                    target_speed = min(target_speed, 4.0)
            else:
                target_speed = min(target_speed, 3.0)

        mission_target_name = None
        mission_distance = self.latest_mission.get("distance_to_objective_m")
        mission_target = self.target_from_mission()
        if mission_target is not None:
            mission_target_name = mission_target[2]

        light = self.traffic_light_ahead(ego, ego_wp)
        decision, target_speed, brake_required, emergency, reason, red_hold, green_release = (
            self.apply_traffic_light_policy(
                base_speed=target_speed,
                current_speed=current_speed,
                light=light,
            )
        )

        if reason == "lane_follow":
            reason = lane_reason

        if decision != "STOP":
            target_speed = min(target_speed, lane_speed_cap)

        if bool(self.latest_mission.get("completed", False)):
            decision = "STOP"
            target_speed = 0.0
            brake_required = True
            reason = "mission_completed"
        elif bool(self.latest_mission.get("must_stop", False)):
            decision = "STOP"
            target_speed = 0.0
            brake_required = True
            reason = "mission_stop_stage"

        light_state = light["state"] if light is not None else "none"
        light_actor = light["actor_id"] if light is not None else None
        stopline_distance = round(light["distance_m"], 3) if light is not None else None
        target_loc = target_wp.transform.location

        command = {
            "stamp": now,
            "decision": decision,
            "target_speed": round(float(target_speed), 3),
            "steering_target": round(float(steer), 4),
            "brake_required": bool(brake_required),
            "emergency": bool(emergency),
            "reason": reason,
            "current_lane_offset_m": round(float(lane_offset), 3),
            "heading_error_deg": round(float(heading_error), 3),
            "traffic_light_state": light_state,
            "traffic_light_source": "carla_actor" if light is not None else "unknown",
            "traffic_light_actor_id": light_actor,
            "stopline_distance_m": stopline_distance,
            "red_hold_active": bool(red_hold),
            "green_release_active": bool(green_release),
            "active_sign_rule": sign_rule,
            "active_sign_confidence": round(float(sign_conf), 3),
            "mission_target": mission_target_name,
            "target_distance_m": mission_distance,
            "speed_mps": round(float(current_speed), 3),
            "selected_turn": turn_label,
            "path_mode": local_path.mode,
            "active_junction_branch": local_path.branch if local_path.mode == "junction" else None,
            "target_accepted": bool(target_accepted),
            "target_reject_reason": reject_reason,
            "target_forward_projection_m": round(float(
                self.last_target_diag.get("forward_projection_m", forward_m) or 0.0
            ), 3),
            "target_age_s": round(float(self.last_target_diag.get("target_age_s", 0.0) or 0.0), 3),
            "locked_target_index": int(self.last_target_diag.get("locked_target_index", local_path.target_index) or -1),
            "locked_path_len": int(self.last_target_diag.get("locked_path_len", local_path.locked_path_len) or 0),
            "target_fallback_used": bool(self.last_target_diag.get("fallback_used", False)),
            "lookahead_m": round(float(local_path.lookahead_m), 3),
            "raw_steer": round(float(raw_steer), 5),
            "limited_steer": round(float(limited_steer), 5),
            "steer_rate_limited": bool(steer_rate_limited or fusion["rate_limited"]),
            "lane_speed_cap": round(float(lane_speed_cap), 3),
            "final_target_speed": round(float(target_speed), 3),
            "lane_error_level": lane_error_level,
            "lane_source": fusion["source"],
            "lane_vision_status": lane_vision_status,
            "vision_confidence": round(float(fusion["vision_conf"]), 3),
            "vision_steer": round(float(fusion["vision_steer"]), 5),
            "map_steer": round(float(fusion["map_steer"]), 5),
            "vision_offset_norm": round(float(fusion["offset_norm"]), 6),
            "vision_heading_error_deg": round(float(fusion["heading_deg"]), 3),
            "vision_map_conflict": bool(fusion["conflict"]),
            "vision_source": fusion["vision_source"],
            "vision_rejected_reason": fusion["vision_rejected_reason"],
            "vision_weight": round(float(fusion["vision_weight"]), 3),
            "map_weight": round(float(fusion["map_weight"]), 3),
            "vision_speed_cap_reason": vision_cap_reason,
            "lane_source_prev": fusion.get("lane_source_prev"),
            "lane_source_changed": bool(fusion.get("lane_source_changed", False)),
            "vision_weight_target": round(float(fusion.get("vision_weight_target", 0.0)), 3),
            "junction_exit_ramp_active": bool(junction_exit_ramp_active),
        }

        msg = String()
        msg.data = json.dumps(command)
        self.command_pub.publish(msg)

        self.maybe_log(
            "path",
            "CLEAN_PATH "
            f"mode={local_path.mode} branch={local_path.branch} "
            f"target_x={target_loc.x:.3f} target_y={target_loc.y:.3f} "
            f"lookahead={local_path.lookahead_m:.1f} reason={local_path.reason}",
        )
        self.maybe_log(
            "junction",
            "CLEAN_JUNCTION "
            f"active={local_path.mode == 'junction'} "
            f"selected_branch={local_path.branch} "
            f"locked={self.active_junction_branch is not None} "
            f"target_index={local_path.target_index} "
            f"path_len={local_path.locked_path_len if local_path.locked else len(local_path.waypoints)} "
            f"progress_index={local_path.progress_index} "
            f"stale={local_path.stale} "
            f"release_reason={local_path.release_reason or self.last_junction_release_reason} "
            f"exit_ramp={time.time() < self.junction_exit_ramp_until_s} "
            f"candidates={local_path.candidates} reason={local_path.reason}",
        )
        self.maybe_log(
            "target",
            "CLEAN_TARGET "
            f"accepted={target_accepted} reject_reason={reject_reason} "
            f"forward_projection_m={command['target_forward_projection_m']} "
            f"target_age_s={command['target_age_s']} "
            f"locked_target_index={command['locked_target_index']} "
            f"locked_path_len={command['locked_path_len']} "
            f"fallback_used={command['target_fallback_used']} "
            f"heading_error={command['heading_error_deg']} "
            f"lateral_jump={local_path.max_lateral_jump_m:.3f}",
        )
        self.maybe_log(
            "fusion",
            "CLEAN_LANE_FUSION "
            f"source={command['lane_source']} "
            f"vision_conf={command['vision_confidence']} "
            f"vision_steer={command['vision_steer']} "
            f"map_steer={command['map_steer']} "
            f"final_steer={command['steering_target']} "
            f"vision_source={command['vision_source']} "
            f"vision_rejected_reason={command['vision_rejected_reason']} "
            f"vision_weight={command['vision_weight']} "
            f"map_weight={command['map_weight']} "
            f"weight_target={command['vision_weight_target']} "
            f"prev={command['lane_source_prev']} "
            f"changed={command['lane_source_changed']} "
            f"exit_ramp={command['junction_exit_ramp_active']} "
            f"conflict={command['vision_map_conflict']} "
            f"reason={fusion['reason']}",
        )
        self.maybe_log(
            "lane",
            "CLEAN_LANE "
            f"offset={command['current_lane_offset_m']} "
            f"heading={command['heading_error_deg']} "
            f"lookahead={command['lookahead_m']} "
            f"raw_steer={command['raw_steer']} "
            f"limited_steer={command['limited_steer']} "
            f"steer_rate_limited={command['steer_rate_limited']} "
            f"steer={command['steering_target']} "
            f"speed={command['speed_mps']} "
            f"lane_source={command['lane_source']} "
            f"lane_cap={command['lane_speed_cap']} "
            f"final_target={command['final_target_speed']} "
            f"error_level={command['lane_error_level']} "
            f"reason={lane_reason}",
        )
        self.maybe_log(
            "tl",
            "CLEAN_TL "
            f"state={light_state} actor={light_actor} "
            f"stopline={stopline_distance} hold={red_hold} reason={reason}",
        )
        self.maybe_log(
            "sign",
            "CLEAN_SIGN "
            f"rule={sign_rule} confidence={round(float(sign_conf), 3)} "
            f"active={sign_rule is not None} reason={sign_reason}",
        )
        self.maybe_log(
            "cmd",
            "CLEAN_CMD "
            f"decision={decision} target={command['target_speed']} "
            f"steer={command['steering_target']} "
            f"lane_source={command['lane_source']} "
            f"lane_cap={command['lane_speed_cap']} "
            f"final_target={command['final_target_speed']} "
            f"reason={reason}",
        )

    def publish_command(
        self,
        decision,
        target_speed,
        steering_target,
        brake_required,
        emergency,
        reason,
    ):
        msg = String()
        msg.data = json.dumps({
            "stamp": time.time(),
            "decision": decision,
            "target_speed": float(target_speed),
            "steering_target": float(steering_target),
            "brake_required": bool(brake_required),
            "emergency": bool(emergency),
            "reason": reason,
            "current_lane_offset_m": 0.0,
            "heading_error_deg": 0.0,
            "traffic_light_state": "none",
            "traffic_light_source": "unknown",
            "stopline_distance_m": None,
            "red_hold_active": False,
            "green_release_active": False,
            "active_sign_rule": None,
            "mission_target": None,
            "target_distance_m": None,
        })
        self.command_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CleanPhase1DriverNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
