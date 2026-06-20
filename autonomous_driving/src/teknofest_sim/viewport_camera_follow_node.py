#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from rclpy.node import Node

from teknofest_sim.carla_loader import load_carla


class ViewportCameraFollowNode(Node):
    def __init__(self):
        super().__init__("viewport_camera_follow_node")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("camera_view", "chase")
        self.declare_parameter("camera_follow_rate_hz", 90.0)
        self.declare_parameter("camera_distance_m", 7.0)
        self.declare_parameter("camera_height_m", 3.2)
        self.declare_parameter("camera_pitch_deg", -12.0)
        self.declare_parameter("hood_forward_m", 1.7)
        self.declare_parameter("hood_up_m", 1.5)
        self.declare_parameter("hood_pitch_deg", -5.0)

        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)
        self.camera_view = str(self.get_parameter("camera_view").value).lower()
        self.camera_follow_rate_hz = max(30.0, float(self.get_parameter("camera_follow_rate_hz").value))
        self.timer_period_sec = 1.0 / self.camera_follow_rate_hz
        self.camera_distance_m = float(self.get_parameter("camera_distance_m").value)
        self.camera_height_m = float(self.get_parameter("camera_height_m").value)
        self.camera_pitch_deg = float(self.get_parameter("camera_pitch_deg").value)
        self.hood_forward_m = float(self.get_parameter("hood_forward_m").value)
        self.hood_up_m = float(self.get_parameter("hood_up_m").value)
        self.hood_pitch_deg = float(self.get_parameter("hood_pitch_deg").value)

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()
        self.ego_vehicle = None

        self.create_timer(self.timer_period_sec, self.tick)
        self.get_logger().info("Viewport camera follow ready")
        self.get_logger().info(f"Target role_name={self.ego_role_name}")

    def find_ego_vehicle(self):
        if self.ego_vehicle is not None:
            try:
                if self.ego_vehicle.is_alive:
                    return self.ego_vehicle
            except Exception:
                self.ego_vehicle = None
        try:
            self.world = self.client.get_world()
            self.spectator = self.world.get_spectator()
        except Exception:
            return None
        for vehicle in self.world.get_actors().filter("vehicle.*"):
            if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                self.ego_vehicle = vehicle
                return vehicle
        return None

    def chase_transform(self, ego_transform):
        base = ego_transform.location
        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()
        camera_loc = self.carla.Location(
            x=base.x - forward.x * self.camera_distance_m - right.x * 0.0,
            y=base.y - forward.y * self.camera_distance_m - right.y * 0.0,
            z=base.z + self.camera_height_m,
        )
        rotation = self.carla.Rotation(
            pitch=self.camera_pitch_deg,
            yaw=ego_transform.rotation.yaw,
            roll=0.0,
        )
        return self.carla.Transform(camera_loc, rotation)

    def hood_transform(self, ego_transform):
        base = ego_transform.location
        forward = ego_transform.get_forward_vector()
        camera_loc = self.carla.Location(
            x=base.x + forward.x * self.hood_forward_m,
            y=base.y + forward.y * self.hood_forward_m,
            z=base.z + self.hood_up_m,
        )
        rotation = self.carla.Rotation(
            pitch=self.hood_pitch_deg,
            yaw=ego_transform.rotation.yaw,
            roll=0.0,
        )
        return self.carla.Transform(camera_loc, rotation)

    def tick(self):
        ego_vehicle = self.find_ego_vehicle()
        if ego_vehicle is None:
            return
        try:
            ego_transform = ego_vehicle.get_transform()
            if self.camera_view == "hood":
                transform = self.hood_transform(ego_transform)
            else:
                transform = self.chase_transform(ego_transform)
            self.spectator.set_transform(transform)
        except Exception:
            self.ego_vehicle = None


def main(args=None):
    rclpy.init(args=args)
    node = ViewportCameraFollowNode()
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
