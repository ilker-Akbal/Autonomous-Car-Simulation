import json
import math
import os
import time
from typing import Any, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.geojson_mission import load_mission_geojson, mission_to_dict


class MissionRouteManager(Node):
    def __init__(self):
        super().__init__("mission_route_manager")

        self.declare_parameter("carla_root", "/mnt/carla/CARLA_0.9.15")
        self.declare_parameter("mission_geojson", "")
        self.declare_parameter("target_reached_distance_m", 4.0)
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("loop_mission", False)
        self.declare_parameter("competition_mode", True)

        self.carla_root = self.get_parameter("carla_root").value
        self.mission_geojson = self.get_parameter("mission_geojson").value
        self.target_reached_distance_m = float(self.get_parameter("target_reached_distance_m").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.loop_mission = bool(self.get_parameter("loop_mission").value)
        self.competition_mode = bool(self.get_parameter("competition_mode").value)

        self._mission_spec = None
        self._targets: List[dict[str, Any]] = []
        self._current_target_index: int = 0
        self._last_status: Optional[dict[str, Any]] = None
        self._last_status_time: float = 0.0
        self._load_mission()

        self.create_subscription(String, "/adas/carla/status", self._status_callback, 10)

        self.targets_pub = self.create_publisher(String, "/adas/mission/targets", 10)
        self.current_goal_pub = self.create_publisher(String, "/adas/mission/current_goal", 10)
        self.status_pub = self.create_publisher(String, "/adas/mission/status", 10)

        self.timer = self.create_timer(1.0 / max(0.1, self.publish_rate_hz), self._publish)

    def _load_mission(self) -> None:
        if not self.mission_geojson:
            self.get_logger().warn("MissionRouteManager: mission_geojson parameter is empty")
            return

        if not os.path.exists(self.mission_geojson):
            self.get_logger().warn(f"MissionRouteManager: mission_geojson not found: {self.mission_geojson}")
            return

        try:
            self._mission_spec = load_mission_geojson(
                path=self.mission_geojson,
                competition_mode=self.competition_mode,
            )
            self._targets = self._build_targets()
            self._current_target_index = 0
            self.get_logger().info(f"Loaded mission with {len(self._targets)} targets")
        except Exception as exc:
            self._mission_spec = None
            self._targets = []
            self.get_logger().error(f"Failed to load mission geojson: {exc}")

    def _build_targets(self) -> List[dict[str, Any]]:
        if self._mission_spec is None:
            return []

        targets: List[dict[str, Any]] = []
        targets.append({"role": "start", **self._mission_spec.start.__dict__})
        for task in self._mission_spec.task_points:
            targets.append({"role": "task", **task.__dict__})
        targets.append({"role": "park", **self._mission_spec.park_entry.__dict__})
        return targets

    def _status_callback(self, msg: String) -> None:
        try:
            self._last_status = json.loads(msg.data)
            self._last_status_time = time.time()
        except Exception:
            self.get_logger().warn("MissionRouteManager: failed to parse /adas/carla/status JSON")

    def _is_status_fresh(self) -> bool:
        return (time.time() - self._last_status_time) < 2.0

    def _distance_to_point(self, x: float, y: float, point: dict[str, Any]) -> float:
        px = float(point.get("carla_x", point.get("x", 0.0)) or 0.0)
        py = float(point.get("carla_y", point.get("y", 0.0)) or 0.0)
        return math.hypot(px - x, py - y)

    def _select_current_goal(self) -> tuple[Optional[dict[str, Any]], Optional[float]]:
        if not self._targets or self._last_status is None:
            return None, None

        ego_loc = self._last_status.get("location", {})
        ego_x = float(ego_loc.get("x", 0.0))
        ego_y = float(ego_loc.get("y", 0.0))

        if self._current_target_index < 0:
            self._current_target_index = 0

        if self._current_target_index == 0 and len(self._targets) > 1:
            start_point = self._targets[0]
            distance_to_start = self._distance_to_point(ego_x, ego_y, start_point)
            if distance_to_start < self.target_reached_distance_m:
                self._current_target_index = 1

        while self._current_target_index < len(self._targets):
            target = self._targets[self._current_target_index]
            distance = self._distance_to_point(ego_x, ego_y, target)
            if distance < self.target_reached_distance_m:
                if self._current_target_index == len(self._targets) - 1:
                    if self.loop_mission:
                        self._current_target_index = 0
                        continue
                    break
                self._current_target_index += 1
                continue
            return target, distance

        return self._targets[-1], self._distance_to_point(ego_x, ego_y, self._targets[-1])

    def _publish(self) -> None:
        now = time.time()
        mission_targets_payload = {
            "stamp": now,
            "mission_geojson": self.mission_geojson,
            "ok": self._mission_spec is not None,
            "targets": self._targets,
        }

        current_goal, distance_to_goal = self._select_current_goal()
        current_goal_payload = {
            "stamp": now,
            "mission_geojson": self.mission_geojson,
            "ok": self._mission_spec is not None and current_goal is not None,
            "current_index": self._current_target_index,
            "current_goal": current_goal,
            "distance_to_goal_m": distance_to_goal,
            "status_fresh": self._is_status_fresh(),
        }

        status_payload = {
            "stamp": now,
            "mission_geojson": self.mission_geojson,
            "ok": self._mission_spec is not None,
            "status_fresh": self._is_status_fresh(),
            "mission_loaded": self._mission_spec is not None,
            "target_index": self._current_target_index,
            "target_reached_distance_m": self.target_reached_distance_m,
        }

        self.targets_pub.publish(String(data=json.dumps(mission_targets_payload)))
        self.current_goal_pub.publish(String(data=json.dumps(current_goal_payload)))
        self.status_pub.publish(String(data=json.dumps(status_payload)))


def main(args=None):
    rclpy.init(args=args)
    node = MissionRouteManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
