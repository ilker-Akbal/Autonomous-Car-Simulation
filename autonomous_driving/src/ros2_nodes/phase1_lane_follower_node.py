import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


class Phase1LaneFollowerNode(Node):
    def __init__(self):
        super().__init__("phase1_lane_follower_node")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 120.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("route_topic", "/adas/phase1/route")
        self.declare_parameter("lane_command_topic", "/adas/phase1/lane_command")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("route_timeout_s", 1.0)
        self.declare_parameter("heading_kp", 0.020)
        self.declare_parameter("lateral_kp", 0.18)
        self.declare_parameter("max_steer", 0.55)
        self.declare_parameter("steer_smoothing_alpha", 0.35)
        self.declare_parameter("max_route_lateral_error_m", 1.5)

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

        self.latest_route = None
        self.latest_route_time = 0.0
        self.filtered_steer = 0.0

        self.create_subscription(
            String,
            str(self.get_parameter("route_topic").value),
            self.route_callback,
            10,
        )
        self.pub = self.create_publisher(
            String,
            str(self.get_parameter("lane_command_topic").value),
            10,
        )

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.tick)

        self.get_logger().info("PHASE1_LANE_FOLLOWER_READY")

    def wait_for_ego_vehicle(self):
        deadline = time.time() + 30.0
        while time.time() < deadline:
            for vehicle in self.world.get_actors().filter("vehicle.*"):
                if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                    return vehicle
            time.sleep(0.2)
        raise RuntimeError("Phase1 lane follower ego vehicle not found")

    def route_callback(self, msg):
        try:
            self.latest_route = json.loads(msg.data)
            self.latest_route_time = time.time()
        except Exception as exc:
            self.get_logger().warning(f"route parse error: {exc}")

    @staticmethod
    def clamp(value, min_value, max_value):
        return max(min_value, min(float(value), max_value))

    @staticmethod
    def normalize_angle(angle_deg):
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return angle_deg

    @staticmethod
    def local_y(dx, dy, yaw_deg):
        yaw = math.radians(yaw_deg)
        return -math.sin(yaw) * dx + math.cos(yaw) * dy

    def publish_invalid(self, reason):
        payload = {
            "stamp": time.time(),
            "valid": False,
            "reason": reason,
            "steering_target": 0.0,
            "lateral_offset_m": None,
            "heading_error_deg": None,
            "confidence": 0.0,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)

    def tick(self):
        now = time.time()
        route_age = now - self.latest_route_time
        route_timeout = float(self.get_parameter("route_timeout_s").value)

        if self.latest_route is None or route_age > route_timeout:
            self.publish_invalid("route_missing_or_stale")
            return

        if not bool(self.latest_route.get("valid", False)):
            self.publish_invalid("route_invalid")
            return

        route_lateral_error = self.latest_route.get("route_lateral_error_m")
        if route_lateral_error is not None:
            try:
                route_lateral_error = abs(float(route_lateral_error))
            except Exception:
                route_lateral_error = None

        if (
            route_lateral_error is not None
            and route_lateral_error > float(self.get_parameter("max_route_lateral_error_m").value)
        ):
            self.publish_invalid("route_lateral_jump")
            return

        local_target = self.latest_route.get("local_target")
        if not isinstance(local_target, dict):
            self.publish_invalid("local_target_missing")
            return

        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_yaw = float(ego_transform.rotation.yaw)

        try:
            target_x = float(local_target["x"])
            target_y = float(local_target["y"])
            target_yaw = float(local_target.get("yaw", ego_yaw))
        except Exception:
            self.publish_invalid("local_target_bad_fields")
            return

        ego_wp = self.map.get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )

        if ego_wp is None:
            self.publish_invalid("ego_not_on_driving_lane")
            return

        wp_location = ego_wp.transform.location
        lateral_offset = self.local_y(
            ego_location.x - wp_location.x,
            ego_location.y - wp_location.y,
            ego_wp.transform.rotation.yaw,
        )

        bearing = math.degrees(math.atan2(target_y - ego_location.y, target_x - ego_location.x))
        heading_to_target = self.normalize_angle(bearing - ego_yaw)
        lane_heading = self.normalize_angle(target_yaw - ego_yaw)
        heading_error = 0.65 * heading_to_target + 0.35 * lane_heading

        raw_steer = (
            float(self.get_parameter("heading_kp").value) * heading_error
            - float(self.get_parameter("lateral_kp").value) * lateral_offset
        )
        max_steer = float(self.get_parameter("max_steer").value)
        raw_steer = self.clamp(raw_steer, -max_steer, max_steer)

        alpha = self.clamp(float(self.get_parameter("steer_smoothing_alpha").value), 0.0, 1.0)
        self.filtered_steer = alpha * raw_steer + (1.0 - alpha) * self.filtered_steer
        steering_target = self.clamp(self.filtered_steer, -max_steer, max_steer)

        offset_abs = abs(lateral_offset)
        confidence = max(0.0, min(1.0, 1.0 - offset_abs / 2.0))
        reason = "ok"
        if offset_abs > 1.5:
            reason = "large_lateral_offset"
        elif abs(heading_error) > 45.0:
            reason = "large_heading_error"

        payload = {
            "stamp": now,
            "valid": True,
            "reason": reason,
            "steering_target": round(float(steering_target), 4),
            "lateral_offset_m": round(float(lateral_offset), 4),
            "heading_error_deg": round(float(heading_error), 3),
            "confidence": round(float(confidence), 3),
            "route_age": round(route_age, 3),
            "route_lateral_error_m": route_lateral_error,
            "road_id": int(ego_wp.road_id),
            "lane_id": int(ego_wp.lane_id),
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)

        self.get_logger().info(
            "PHASE1_LANE "
            f"steer={steering_target:.3f} offset={lateral_offset:.3f} "
            f"heading={heading_error:.1f} reason={reason}",
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = Phase1LaneFollowerNode()
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
