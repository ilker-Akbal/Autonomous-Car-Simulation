#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_common.runtime_logging import RuntimeJsonlLogger
from teknofest_sim.carla_loader import load_carla


class RoutePlannerNode(Node):
    def __init__(self):
        super().__init__("route_planner_node")

        # -------------------------
        # Config / parameter block
        # -------------------------
        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 120.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("sampling_resolution_m", 1.2)
        self.declare_parameter("target_replan_threshold_m", 1.0)
        self.declare_parameter("off_route_replan_distance_m", 4.0)
        self.declare_parameter("route_end_replan_distance_m", 6.0)
        self.declare_parameter("initial_lane_prefix_m", 12.0)
        self.declare_parameter("log_root", "autonomous_driving/outputs/teknofest_sim_logs")
        self.declare_parameter("log_session_id", "")
        self.declare_parameter("jsonl_logging_enabled", True)
        self.declare_parameter("ros_log_period_s", 1.0)

        self.declare_parameter("mission_topic", "/adas/teknofest/mission")
        self.declare_parameter("route_topic", "/adas/planning/route")

        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)
        self.sampling_resolution_m = float(self.get_parameter("sampling_resolution_m").value)
        self.target_replan_threshold_m = float(
            self.get_parameter("target_replan_threshold_m").value
        )
        self.off_route_replan_distance_m = float(
            self.get_parameter("off_route_replan_distance_m").value
        )
        self.route_end_replan_distance_m = float(
            self.get_parameter("route_end_replan_distance_m").value
        )
        self.initial_lane_prefix_m = float(self.get_parameter("initial_lane_prefix_m").value)
        self.ros_log_period_s = float(self.get_parameter("ros_log_period_s").value)

        # -------------------------
        # Runtime state block
        # -------------------------
        self.carla = None
        self.client = None
        self.world = None
        self.map = None
        self.global_route_planner = None
        self.ego_vehicle = None
        self.last_ego_lookup_s = 0.0
        self.last_plan_s = 0.0
        self.last_target_xy: Optional[tuple[float, float]] = None
        self.last_objective_index: Optional[int] = None
        self.last_route_payload: Optional[dict] = None
        self.mission_payload: Optional[dict] = None
        self.last_ros_log_s = 0.0
        self.last_replan_reason = "startup"
        self.last_route_nearest_index = 0
        self.last_route_nearest_distance_m = None

        self.runtime_logger = RuntimeJsonlLogger(
            node_name="route_planner_node",
            file_name="planning.jsonl",
            log_root=str(self.get_parameter("log_root").value),
            session_id=str(self.get_parameter("log_session_id").value) or None,
            enabled=bool(self.get_parameter("jsonl_logging_enabled").value),
        )

        # -------------------------
        # Publisher block
        # -------------------------
        self.route_pub = self.create_publisher(
            String,
            str(self.get_parameter("route_topic").value),
            10,
        )

        # -------------------------
        # Subscriber block
        # -------------------------
        self.create_subscription(
            String,
            str(self.get_parameter("mission_topic").value),
            self.mission_cb,
            10,
        )

        # -------------------------
        # Timer / startup block
        # -------------------------
        self.connect_to_carla()
        self.create_timer(0.2, self.tick)
        self.get_logger().info("Route planner node ready.")

    # -------------------------
    # CARLA helper functions
    # -------------------------
    def connect_to_carla(self):
        self.carla = load_carla(self.carla_root)
        carla_python_api = os.path.join(self.carla_root, "PythonAPI", "carla")
        if os.path.isdir(carla_python_api) and carla_python_api not in sys.path:
            sys.path.append(carla_python_api)

        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner

            self.global_route_planner = GlobalRoutePlanner(
                self.map,
                self.sampling_resolution_m,
            )
            self.get_logger().info("CARLA GlobalRoutePlanner active.")
        except Exception as exc:
            self.global_route_planner = None
            self.get_logger().warning(
                f"GlobalRoutePlanner unavailable, direct waypoint fallback will be used: {exc}"
            )

    def find_ego_vehicle(self):
        now = time.time()

        if self.ego_vehicle is not None:
            try:
                if self.ego_vehicle.is_alive:
                    return self.ego_vehicle
            except Exception:
                self.ego_vehicle = None

        if now - self.last_ego_lookup_s < 1.0:
            return self.ego_vehicle

        self.last_ego_lookup_s = now
        vehicles = self.world.get_actors().filter("vehicle.*")

        for vehicle in vehicles:
            if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                self.ego_vehicle = vehicle
                self.get_logger().info(f"Ego vehicle found for route planner: id={vehicle.id}")
                return vehicle

        return None

    # -------------------------
    # Subscriber callbacks
    # -------------------------
    def mission_cb(self, msg: String):
        try:
            self.mission_payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid mission JSON ignored: {exc}")

    # -------------------------
    # Planning functions
    # -------------------------
    def current_target(self) -> Optional[dict]:
        if not self.mission_payload:
            return None

        return (
            self.mission_payload.get("objective_target")
            or self.mission_payload.get("target")
            or {}
        )

    def target_location(self):
        target = self.current_target()
        if not target:
            return None

        x = target.get("carla_x")
        y = target.get("carla_y")
        z = target.get("carla_z", 0.2)

        if x is None or y is None:
            return None

        return self.carla.Location(x=float(x), y=float(y), z=float(z or 0.2))

    def should_replan(self, target_loc) -> bool:
        objective_index = None

        if self.mission_payload:
            objective_index = self.mission_payload.get("objective_index")

        if self.last_route_payload is None:
            self.last_replan_reason = "no_previous_route"
            return True

        if objective_index != self.last_objective_index:
            self.last_replan_reason = "objective_changed"
            return True

        if self.last_target_xy is None:
            self.last_replan_reason = "missing_previous_target"
            return True

        moved = math.hypot(
            target_loc.x - self.last_target_xy[0],
            target_loc.y - self.last_target_xy[1],
        )
        if moved >= self.target_replan_threshold_m:
            self.last_replan_reason = "target_moved"
            return True

        ego = self.find_ego_vehicle()
        if ego is not None:
            ego_loc = ego.get_location()
            nearest_index, nearest_dist = self.nearest_route_point(ego_loc)
            self.last_route_nearest_index = nearest_index
            self.last_route_nearest_distance_m = nearest_dist

            if nearest_dist is not None and nearest_dist >= self.off_route_replan_distance_m:
                self.last_replan_reason = "off_route"
                return True

            route_length = len((self.last_route_payload or {}).get("points", []))
            target_distance = math.hypot(ego_loc.x - target_loc.x, ego_loc.y - target_loc.y)
            near_route_end = route_length > 0 and nearest_index >= max(0, route_length - 4)
            if near_route_end and target_distance >= self.route_end_replan_distance_m:
                self.last_replan_reason = "route_end_reached"
                return True

        self.last_replan_reason = "reuse_route"
        return False

    def nearest_route_point(self, ego_loc) -> tuple[int, Optional[float]]:
        points = (self.last_route_payload or {}).get("points", [])
        if not points:
            return 0, None

        nearest_index = 0
        nearest_distance = float("inf")

        for index, point in enumerate(points):
            try:
                dist = math.hypot(float(point["x"]) - ego_loc.x, float(point["y"]) - ego_loc.y)
            except (KeyError, TypeError, ValueError):
                continue

            if dist < nearest_distance:
                nearest_distance = dist
                nearest_index = index

        return nearest_index, nearest_distance if nearest_distance < float("inf") else None

    def waypoint_for_location(self, loc):
        try:
            return self.map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=self.carla.LaneType.Driving,
            )
        except Exception:
            return None

    def waypoint_to_dict(self, waypoint, index: int, road_option: str = "LANEFOLLOW") -> dict:
        transform = waypoint.transform
        return {
            "index": index,
            "x": round(float(transform.location.x), 4),
            "y": round(float(transform.location.y), 4),
            "z": round(float(transform.location.z), 4),
            "yaw_deg": round(float(transform.rotation.yaw), 4),
            "road_id": int(waypoint.road_id),
            "lane_id": int(waypoint.lane_id),
            "lane_width": round(float(waypoint.lane_width), 4),
            "is_junction": bool(waypoint.is_junction),
            "lane_type": str(waypoint.lane_type),
            "road_option": str(road_option),
        }

    def trace_route(self, start_loc, target_loc) -> list[dict]:
        prefix = self.current_lane_prefix(start_loc)

        if self.global_route_planner is not None:
            route = self.global_route_planner.trace_route(start_loc, target_loc)
            global_points = [
                self.waypoint_to_dict(waypoint, index, getattr(option, "name", str(option)))
                for index, (waypoint, option) in enumerate(route)
            ]
            return self.merge_route_prefix(prefix, global_points)

        start_wp = self.map.get_waypoint(
            start_loc,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )
        target_wp = self.map.get_waypoint(
            target_loc,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )

        fallback_points = [
            self.waypoint_to_dict(start_wp, 0, "FALLBACK_START"),
            self.waypoint_to_dict(target_wp, 1, "FALLBACK_TARGET"),
        ]
        return self.merge_route_prefix(prefix, fallback_points)

    def current_lane_prefix(self, start_loc) -> list[dict]:
        start_wp = self.waypoint_for_location(start_loc)
        if start_wp is None:
            return []

        prefix = [self.waypoint_to_dict(start_wp, 0, "INITIAL_LANE_HOLD")]
        current = start_wp
        traveled = 0.0
        index = 1

        while traveled < self.initial_lane_prefix_m:
            try:
                candidates = current.next(self.sampling_resolution_m)
            except Exception:
                break

            same_lane = [
                wp for wp in candidates
                if wp.road_id == start_wp.road_id and wp.lane_id == start_wp.lane_id
            ]
            if not same_lane:
                break

            current = same_lane[0]
            prefix.append(self.waypoint_to_dict(current, index, "INITIAL_LANE_HOLD"))
            traveled += self.sampling_resolution_m
            index += 1

        return prefix

    def merge_route_prefix(self, prefix: list[dict], global_points: list[dict]) -> list[dict]:
        if not prefix:
            return self.reindex_route(global_points)

        merged = list(prefix)
        last = merged[-1]

        for point in global_points:
            try:
                dist = math.hypot(float(point["x"]) - float(last["x"]), float(point["y"]) - float(last["y"]))
            except (KeyError, TypeError, ValueError):
                dist = 0.0

            if dist < self.sampling_resolution_m * 0.75:
                continue
            merged.append(point)

        return self.reindex_route(merged)

    def reindex_route(self, points: list[dict]) -> list[dict]:
        out = []
        for index, point in enumerate(points):
            item = dict(point)
            item["index"] = index
            out.append(item)
        return out

    def build_route_payload(self, ego, target_loc, route_points: list[dict]) -> dict:
        transform = ego.get_transform()
        velocity = ego.get_velocity()
        speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        current_wp = self.waypoint_for_location(transform.location)
        target_wp = self.waypoint_for_location(target_loc)
        objective_index = None
        objective_kind = None
        distance_to_objective_m = None

        if self.mission_payload:
            objective_index = self.mission_payload.get("objective_index")
            objective_kind = self.mission_payload.get("objective_kind")
            distance_to_objective_m = self.mission_payload.get("distance_to_objective_m")

        return {
            "stamp": time.time(),
            "frame": "carla_map",
            "source": "carla_global_route_planner",
            "route_id": f"{objective_index}:{round(time.time(), 3)}",
            "replan_reason": self.last_replan_reason,
            "route_reused": False,
            "objective_index": objective_index,
            "objective_kind": objective_kind,
            "active_mission_target": self.current_target(),
            "distance_to_objective_m": distance_to_objective_m,
            "ego": {
                "x": round(float(transform.location.x), 4),
                "y": round(float(transform.location.y), 4),
                "z": round(float(transform.location.z), 4),
                "yaw_deg": round(float(transform.rotation.yaw), 4),
                "speed_mps": round(float(speed_mps), 4),
            },
            "target": {
                "x": round(float(target_loc.x), 4),
                "y": round(float(target_loc.y), 4),
                "z": round(float(target_loc.z), 4),
                "road_id": int(target_wp.road_id) if target_wp is not None else None,
                "lane_id": int(target_wp.lane_id) if target_wp is not None else None,
            },
            "current_road_id": int(current_wp.road_id) if current_wp is not None else None,
            "current_lane_id": int(current_wp.lane_id) if current_wp is not None else None,
            "target_road_id": int(target_wp.road_id) if target_wp is not None else None,
            "target_lane_id": int(target_wp.lane_id) if target_wp is not None else None,
            "points": route_points,
            "route_length": len(route_points),
        }

    # -------------------------
    # Main state machine / timer
    # -------------------------
    def tick(self):
        ego = self.find_ego_vehicle()
        target_loc = self.target_location()

        if ego is None or target_loc is None:
            return

        if not self.should_replan(target_loc):
            self.publish_route(self.last_route_payload, route_reused=True)
            return

        try:
            start_loc = ego.get_location()
            route_points = self.trace_route(start_loc, target_loc)
            payload = self.build_route_payload(ego, target_loc, route_points)
        except Exception as exc:
            self.get_logger().warning(f"Route planning failed: {exc}")
            return

        self.last_route_payload = payload
        self.last_plan_s = time.time()
        self.last_target_xy = (float(target_loc.x), float(target_loc.y))
        self.last_objective_index = payload.get("objective_index")
        self.publish_route(payload, route_reused=False)

    # -------------------------
    # Debug / publish block
    # -------------------------
    def publish_route(self, payload: Optional[dict], route_reused: bool):
        if not payload:
            return

        payload = dict(payload)
        ego = self.find_ego_vehicle()
        if ego is not None:
            current_wp = self.waypoint_for_location(ego.get_location())
            if current_wp is not None:
                payload["current_road_id"] = int(current_wp.road_id)
                payload["current_lane_id"] = int(current_wp.lane_id)

        payload["route_reused"] = bool(route_reused)
        payload["replan_reason"] = "reuse_route" if route_reused else payload.get("replan_reason")
        payload["route_nearest_index"] = self.last_route_nearest_index
        payload["route_nearest_distance_m"] = self.last_route_nearest_distance_m

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.route_pub.publish(msg)
        self.log_runtime(payload)

    def log_runtime(self, payload: dict):
        ego = payload.get("ego") or {}
        target = payload.get("active_mission_target") or {}
        record = {
            "ego_x": ego.get("x"),
            "ego_y": ego.get("y"),
            "ego_yaw": ego.get("yaw_deg"),
            "current_speed_mps": ego.get("speed_mps"),
            "active_mission_target": target.get("name"),
            "mission_state": (self.mission_payload or {}).get("stage"),
            "route_index": payload.get("objective_index"),
            "route_length": payload.get("route_length", len(payload.get("points", []))),
            "replan_reason": payload.get("replan_reason"),
            "route_reused": payload.get("route_reused"),
            "route_nearest_index": payload.get("route_nearest_index"),
            "route_nearest_distance_m": payload.get("route_nearest_distance_m"),
            "initial_lane_prefix_m": self.initial_lane_prefix_m,
            "current_road_id": payload.get("current_road_id"),
            "current_lane_id": payload.get("current_lane_id"),
            "target_road_id": payload.get("target_road_id"),
            "target_lane_id": payload.get("target_lane_id"),
            "target_x": (payload.get("target") or {}).get("x"),
            "target_y": (payload.get("target") or {}).get("y"),
            "distance_to_goal_m": payload.get("distance_to_objective_m"),
        }
        self.runtime_logger.write(record)

        now = time.time()
        if now - self.last_ros_log_s >= self.ros_log_period_s:
            self.last_ros_log_s = now
            self.get_logger().info(
                "planning "
                f"state={record['mission_state']} target={record['active_mission_target']} "
                f"idx={record['route_index']} len={record['route_length']} "
                f"reused={record['route_reused']} reason={record['replan_reason']} "
                f"dist={record['distance_to_goal_m']} lane={record['current_road_id']}/{record['current_lane_id']}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = RoutePlannerNode()

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
