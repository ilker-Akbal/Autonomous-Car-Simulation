import json
import math
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla
from teknofest_sim.geojson_mission import load_mission_geojson


def resolve_repo_path(path_value: str) -> str:
    raw = os.path.expanduser(str(path_value or "").strip())
    if os.path.isabs(raw) and os.path.exists(raw):
        return os.path.abspath(raw)

    package_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path.cwd() / raw,
        Path.cwd() / "autonomous_driving" / raw,
        package_root / raw,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    raise FileNotFoundError(f"mission_geojson not found: {path_value}")


class Phase1RouteNode(Node):
    def __init__(self):
        super().__init__("phase1_route_node")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 120.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter(
            "mission_geojson",
            "autonomous_driving/missions/teknofest_town03_competition_v4_tasks_only.geojson",
        )
        self.declare_parameter("round_name", "phase1")
        self.declare_parameter("competition_mode", False)
        self.declare_parameter("route_topic", "/adas/phase1/route")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("sampling_resolution_m", 2.0)
        self.declare_parameter("lookahead_distance_m", 8.0)
        self.declare_parameter("target_reached_distance_m", 5.0)
        self.declare_parameter("target_heading_tolerance_deg", 95.0)
        self.declare_parameter("max_route_lateral_error_m", 12.0)
        self.declare_parameter("prefer_right_lane", True)

        self.carla = load_carla(str(self.get_parameter("carla_root").value))
        self.client = self.carla.Client(
            str(self.get_parameter("host").value),
            int(self.get_parameter("port").value),
        )
        self.client.set_timeout(float(self.get_parameter("timeout").value))
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)
        self.ego_vehicle = self.wait_for_ego_vehicle()

        mission_path = resolve_repo_path(str(self.get_parameter("mission_geojson").value))
        self.get_logger().info(f"PHASE1_ROUTE_LOADED_MISSION {mission_path}")
        self.mission = load_mission_geojson(
            mission_path,
            str(self.get_parameter("round_name").value),
            competition_mode=bool(self.get_parameter("competition_mode").value),
        )
        self.targets = self.build_ordered_targets()
        self.route_waypoints = self.build_route_waypoints()
        self.route_index = 0
        self.target_index = 1 if len(self.targets) > 1 else 0
        self.target_route_indices = self.compute_target_route_indices()

        self.lookahead_distance_m = float(self.get_parameter("lookahead_distance_m").value)
        self.target_reached_distance_m = float(
            self.get_parameter("target_reached_distance_m").value
        )
        self.target_heading_tolerance_deg = float(
            self.get_parameter("target_heading_tolerance_deg").value
        )
        self.max_route_lateral_error_m = float(
            self.get_parameter("max_route_lateral_error_m").value
        )
        self.prefer_right_lane = bool(self.get_parameter("prefer_right_lane").value)

        self.pub = self.create_publisher(
            String,
            str(self.get_parameter("route_topic").value),
            10,
        )

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.tick)

        self.get_logger().info(
            "PHASE1_ROUTE_READY "
            f"mission={mission_path} targets={len(self.targets)} "
            f"route_waypoints={len(self.route_waypoints)} map={self.map.name}"
        )

    def wait_for_ego_vehicle(self):
        deadline = time.time() + 30.0
        while time.time() < deadline:
            for vehicle in self.world.get_actors().filter("vehicle.*"):
                if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                    return vehicle
            time.sleep(0.2)
        raise RuntimeError("Phase1 route node ego vehicle not found")

    def build_ordered_targets(self):
        points = [self.mission.start]
        points.extend(list(self.mission.task_points))
        if self.mission.park_entry is not None:
            points.append(self.mission.park_entry)

        return [self.point_to_waypoint(point) for point in points if point.carla_x is not None]

    def point_to_waypoint(self, point):
        location = self.carla.Location(
            x=float(point.carla_x),
            y=float(point.carla_y),
            z=float(point.carla_z or 0.2),
        )
        waypoint = self.map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )
        waypoint = self.select_directional_waypoint(waypoint, point)
        waypoint = self.prefer_right_if_valid(waypoint, point)
        return {
            "name": str(point.name),
            "kind": str(getattr(point, "kind", "") or ""),
            "waypoint": waypoint,
        }

    @staticmethod
    def normalize_angle(angle_deg):
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return angle_deg

    def select_directional_waypoint(self, waypoint, point):
        if waypoint is None:
            return None

        candidates = [waypoint]
        for lane_fn in ("get_left_lane", "get_right_lane"):
            try:
                lane = getattr(waypoint, lane_fn)()
            except Exception:
                lane = None
            if lane is not None and lane.lane_type == self.carla.LaneType.Driving:
                candidates.append(lane)

        road_id = getattr(point, "road_id", None)
        lane_id = getattr(point, "lane_id", None)
        if road_id is not None and lane_id is not None:
            for candidate in candidates:
                if int(candidate.road_id) == int(road_id) and int(candidate.lane_id) == int(lane_id):
                    return candidate

        target_yaw = getattr(point, "carla_yaw", None)
        if target_yaw is None:
            return waypoint

        best = waypoint
        best_error = abs(self.normalize_angle(waypoint.transform.rotation.yaw - float(target_yaw)))
        for candidate in candidates:
            error = abs(self.normalize_angle(candidate.transform.rotation.yaw - float(target_yaw)))
            if error < best_error:
                best = candidate
                best_error = error

        return best

    def prefer_right_if_valid(self, waypoint, point):
        if waypoint is None or not bool(self.get_parameter("prefer_right_lane").value):
            return waypoint

        if getattr(point, "lane_id", None) is not None:
            return waypoint

        try:
            right = waypoint.get_right_lane()
        except Exception:
            return waypoint

        if right is None:
            return waypoint

        if right.lane_type != self.carla.LaneType.Driving:
            return waypoint

        same_direction = (right.lane_id > 0) == (waypoint.lane_id > 0)
        if not same_direction:
            return waypoint

        return right

    def make_global_route_planner(self):
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner

            return GlobalRoutePlanner(
                self.map,
                float(self.get_parameter("sampling_resolution_m").value),
            )
        except Exception as exc:
            self.get_logger().warning(
                f"GlobalRoutePlanner unavailable, using projected targets only: {exc}"
            )
            return None

    def build_route_waypoints(self):
        waypoints = [item["waypoint"] for item in self.targets if item.get("waypoint")]
        if not waypoints:
            return []

        planner = self.make_global_route_planner()
        if planner is None or len(waypoints) < 2:
            return waypoints

        route = []
        for start_wp, end_wp in zip(waypoints, waypoints[1:]):
            try:
                segment = planner.trace_route(
                    start_wp.transform.location,
                    end_wp.transform.location,
                )
            except Exception as exc:
                self.get_logger().warning(f"Route segment failed: {exc}")
                segment = []

            if not segment:
                route.append(start_wp)
                route.append(end_wp)
                continue

            for wp, _road_option in segment:
                if not route or route[-1].id != wp.id:
                    route.append(wp)

        return route

    def compute_target_route_indices(self):
        indices = []
        if not self.route_waypoints:
            return [0 for _ in self.targets]

        last_index = 0
        for target in self.targets:
            target_wp = target.get("waypoint")
            if target_wp is None:
                indices.append(last_index)
                continue

            best_index = last_index
            best_dist = float("inf")
            for index in range(last_index, len(self.route_waypoints)):
                dist = self.distance_xy(
                    target_wp.transform.location,
                    self.route_waypoints[index].transform.location,
                )
                if dist < best_dist:
                    best_dist = dist
                    best_index = index

            last_index = max(last_index, best_index)
            indices.append(last_index)

        return indices

    @staticmethod
    def distance_xy(a, b):
        return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))

    def update_progress(self, ego_location):
        if not self.route_waypoints:
            return

        search_end = min(len(self.route_waypoints), self.route_index + 80)
        best_index = self.route_index
        best_dist = float("inf")

        for index in range(self.route_index, search_end):
            dist = self.distance_xy(ego_location, self.route_waypoints[index].transform.location)
            if dist < best_dist:
                best_dist = dist
                best_index = index

        self.route_index = max(self.route_index, best_index)

        if self.target_index < len(self.targets):
            target_wp = self.targets[self.target_index]["waypoint"]
            target_dist = self.distance_xy(ego_location, target_wp.transform.location)
            target_route_index = self.target_route_indices[self.target_index]
            route_reached = self.route_index >= max(0, target_route_index - 2)
            heading_ok = self.target_heading_ok(target_wp)

            if (target_dist <= self.target_reached_distance_m and heading_ok) or route_reached:
                self.target_index = min(self.target_index + 1, len(self.targets) - 1)

    def target_heading_ok(self, target_wp):
        ego_yaw = float(self.ego_vehicle.get_transform().rotation.yaw)
        target_yaw = float(target_wp.transform.rotation.yaw)
        error = abs(self.normalize_angle(target_yaw - ego_yaw))
        return error <= self.target_heading_tolerance_deg

    def route_distance_to_target(self, target_route_index, ego_location):
        if not self.route_waypoints:
            return None

        target_route_index = max(0, min(int(target_route_index), len(self.route_waypoints) - 1))
        if self.route_index >= target_route_index:
            return 0.0

        total = self.distance_xy(ego_location, self.route_waypoints[self.route_index].transform.location)
        for index in range(self.route_index, target_route_index):
            total += self.distance_xy(
                self.route_waypoints[index].transform.location,
                self.route_waypoints[index + 1].transform.location,
            )
        return total

    def route_lateral_error(self, ego_location):
        if not self.route_waypoints:
            return None
        wp = self.route_waypoints[self.route_index]
        return self.distance_xy(ego_location, wp.transform.location)

    def local_waypoint(self, ego_location):
        if not self.route_waypoints:
            return None

        total = 0.0
        prev = ego_location
        for index in range(self.route_index, len(self.route_waypoints)):
            wp = self.route_waypoints[index]
            total += self.distance_xy(prev, wp.transform.location)
            if total >= self.lookahead_distance_m:
                return wp
            prev = wp.transform.location

        return self.route_waypoints[-1]

    def waypoint_payload(self, waypoint):
        transform = waypoint.transform
        return {
            "x": round(transform.location.x, 3),
            "y": round(transform.location.y, 3),
            "z": round(transform.location.z, 3),
            "yaw": round(transform.rotation.yaw, 3),
            "road_id": int(waypoint.road_id),
            "lane_id": int(waypoint.lane_id),
            "id": int(waypoint.id),
        }

    def tick(self):
        valid = bool(self.route_waypoints)
        reason = "ok" if valid else "empty_route"

        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location

        if valid:
            self.update_progress(ego_location)
            local_wp = self.local_waypoint(ego_location)
            route_wp = self.route_waypoints[self.route_index]
            target = self.targets[self.target_index]
            target_route_index = self.target_route_indices[self.target_index]
            distance_to_target = self.route_distance_to_target(target_route_index, ego_location)
            route_lateral_error = self.route_lateral_error(ego_location)
            if route_lateral_error is not None and route_lateral_error > self.max_route_lateral_error_m:
                valid = False
                reason = "route_lateral_error_too_large"
        else:
            local_wp = None
            route_wp = None
            target = None
            distance_to_target = None
            target_route_index = None
            route_lateral_error = None

        payload = {
            "stamp": time.time(),
            "valid": valid,
            "reason": reason,
            "route_index": int(self.route_index),
            "route_size": int(len(self.route_waypoints)),
            "target_index": int(self.target_index),
            "target_count": int(len(self.targets)),
            "target_name": target["name"] if target else None,
            "target_kind": target["kind"] if target else None,
            "target_route_index": int(target_route_index) if target_route_index is not None else None,
            "distance_to_target_m": (
                round(float(distance_to_target), 3)
                if distance_to_target is not None
                else None
            ),
            "route_lateral_error_m": (
                round(float(route_lateral_error), 3)
                if route_lateral_error is not None
                else None
            ),
            "ego": {
                "x": round(ego_location.x, 3),
                "y": round(ego_location.y, 3),
                "yaw": round(ego_transform.rotation.yaw, 3),
            },
            "route_waypoint": self.waypoint_payload(route_wp) if route_wp else None,
            "local_target": self.waypoint_payload(local_wp) if local_wp else None,
            "prefer_right_lane": bool(self.prefer_right_lane),
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)

        self.get_logger().info(
            "PHASE1_ROUTE "
            f"valid={valid} target={payload['target_name']} "
            f"dist={payload['distance_to_target_m']} idx={self.route_index}/{len(self.route_waypoints)}",
            throttle_duration_sec=2.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = Phase1RouteNode()
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
