#!/usr/bin/env python3
from __future__ import annotations

import time

import rclpy
from rclpy.node import Node

from teknofest_sim.carla_loader import load_carla


class ViewportCameraFollowNode(Node):
    def __init__(self):
        super().__init__("viewport_camera_follow_node")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("carla_host", "")
        self.declare_parameter("carla_port", 0)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("camera_view", "chase")
        self.declare_parameter("camera_follow_rate_hz", 30.0)
        self.declare_parameter("camera_distance_m", 10.0)
        self.declare_parameter("camera_height_m", 5.0)
        self.declare_parameter("camera_pitch_deg", -18.0)
        self.declare_parameter("follow_update_hz", -1.0)
        self.declare_parameter("follow_distance_m", -1.0)
        self.declare_parameter("follow_height_m", -1.0)
        self.declare_parameter("follow_pitch_deg", -999.0)
        self.declare_parameter("enable_demo_weather", True)
        self.declare_parameter("demo_weather_preset", "clear_sunset")
        self.declare_parameter("enable_ego_lights", True)
        self.declare_parameter("enable_ego_label", True)
        self.declare_parameter("ego_label_text", "ROTA TEKNOFEST")
        self.declare_parameter("ego_retry_timeout_s", 60.0)
        self.declare_parameter("hood_forward_m", 1.7)
        self.declare_parameter("hood_up_m", 1.5)
        self.declare_parameter("hood_pitch_deg", -5.0)

        self.carla_root = str(self.get_parameter("carla_root").value)
        carla_host = str(self.get_parameter("carla_host").value)
        self.host = carla_host if carla_host else str(self.get_parameter("host").value)
        carla_port = int(self.get_parameter("carla_port").value)
        self.port = carla_port if carla_port > 0 else int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)
        self.camera_view = str(self.get_parameter("camera_view").value).lower()
        follow_update_hz = float(self.get_parameter("follow_update_hz").value)
        self.camera_follow_rate_hz = max(
            1.0,
            follow_update_hz
            if follow_update_hz > 0.0
            else float(self.get_parameter("camera_follow_rate_hz").value),
        )
        self.timer_period_sec = 1.0 / self.camera_follow_rate_hz
        follow_distance_m = float(self.get_parameter("follow_distance_m").value)
        follow_height_m = float(self.get_parameter("follow_height_m").value)
        follow_pitch_deg = float(self.get_parameter("follow_pitch_deg").value)
        self.camera_distance_m = (
            follow_distance_m
            if follow_distance_m >= 0.0
            else float(self.get_parameter("camera_distance_m").value)
        )
        self.camera_height_m = (
            follow_height_m
            if follow_height_m >= 0.0
            else float(self.get_parameter("camera_height_m").value)
        )
        self.camera_pitch_deg = (
            follow_pitch_deg
            if follow_pitch_deg > -998.0
            else float(self.get_parameter("camera_pitch_deg").value)
        )
        self.enable_demo_weather = bool(
            self.get_parameter("enable_demo_weather").value
        )
        self.demo_weather_preset = str(
            self.get_parameter("demo_weather_preset").value
        )
        self.enable_ego_lights = bool(self.get_parameter("enable_ego_lights").value)
        self.enable_ego_label = bool(self.get_parameter("enable_ego_label").value)
        self.ego_label_text = str(self.get_parameter("ego_label_text").value)
        self.ego_retry_timeout_s = max(
            0.0,
            float(self.get_parameter("ego_retry_timeout_s").value),
        )
        self.hood_forward_m = float(self.get_parameter("hood_forward_m").value)
        self.hood_up_m = float(self.get_parameter("hood_up_m").value)
        self.hood_pitch_deg = float(self.get_parameter("hood_pitch_deg").value)

        self.carla = None
        self.client = None
        self.world = None
        self.spectator = None
        self.ego_vehicle = None
        self.started_at = time.monotonic()
        self.last_connection_log_time = 0.0
        self.last_missing_ego_log_time = 0.0
        self.last_weather_log_time = 0.0
        self.last_lights_apply_time = 0.0
        self.last_lights_log_time = 0.0
        self.last_label_warn_time = 0.0
        self.spectator_active_logged = False
        self.weather_applied = False
        self.lights_enabled_logged = False
        self.label_enabled_logged = False
        self.missing_ego_timeout_logged = False

        self.create_timer(self.timer_period_sec, self.tick)
        self.get_logger().info(
            "Viewport camera follow ready "
            f"host={self.host} port={self.port} role={self.ego_role_name} "
            f"distance={self.camera_distance_m:.1f}m height={self.camera_height_m:.1f}m "
            f"pitch={self.camera_pitch_deg:.1f}deg hz={self.camera_follow_rate_hz:.1f}"
        )

    def connect_carla(self):
        try:
            if self.carla is None:
                self.carla = load_carla(self.carla_root)
            if self.client is None:
                self.client = self.carla.Client(self.host, self.port)
                self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            self.spectator = self.world.get_spectator()
            self.apply_demo_weather()
            return True
        except Exception as exc:
            now = time.monotonic()
            if now - self.last_connection_log_time > 5.0:
                self.get_logger().warn(f"CARLA spectator connection pending: {exc}")
                self.last_connection_log_time = now
            self.world = None
            self.spectator = None
            return False

    def apply_demo_weather(self):
        if (
            not self.enable_demo_weather
            or self.weather_applied
            or self.demo_weather_preset != "clear_sunset"
            or self.world is None
            or self.carla is None
        ):
            return
        try:
            weather = self.carla.WeatherParameters.ClearSunset
            weather.cloudiness = 10.0
            weather.precipitation = 0.0
            weather.precipitation_deposits = 0.0
            weather.wind_intensity = 0.0
            weather.sun_azimuth_angle = 210.0
            weather.sun_altitude_angle = 14.0
            weather.fog_density = 0.0
            weather.fog_distance = 100000.0
            weather.wetness = 0.0
            self.world.set_weather(weather)
            self.weather_applied = True
            self.get_logger().info("weather clear_sunset applied")
        except Exception as exc:
            now = time.monotonic()
            if now - self.last_weather_log_time > 5.0:
                self.get_logger().warn(f"Demo weather apply pending: {exc}")
                self.last_weather_log_time = now

    def ego_role_candidates(self):
        roles = [self.ego_role_name, "hero", "ego_vehicle"]
        return [
            role
            for index, role in enumerate(roles)
            if role and role not in roles[:index]
        ]

    def find_ego_vehicle(self):
        if self.ego_vehicle is not None:
            try:
                if self.ego_vehicle.is_alive:
                    return self.ego_vehicle
            except Exception:
                self.ego_vehicle = None
        if self.world is None or self.spectator is None:
            if not self.connect_carla():
                return None
        try:
            self.world = self.client.get_world()
            self.spectator = self.world.get_spectator()
            self.apply_demo_weather()
        except Exception:
            self.world = None
            self.spectator = None
            return None
        candidate_roles = self.ego_role_candidates()
        for vehicle in self.world.get_actors().filter("vehicle.*"):
            role_name = vehicle.attributes.get("role_name", "")
            if role_name in candidate_roles:
                self.ego_vehicle = vehicle
                self.missing_ego_timeout_logged = False
                self.get_logger().info(
                    f"Viewport camera following ego id={vehicle.id} role={role_name}"
                )
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

    def enable_lights_if_needed(self, ego_vehicle):
        if not self.enable_ego_lights or self.carla is None:
            return
        now = time.monotonic()
        if now - self.last_lights_apply_time < 1.0:
            return
        self.last_lights_apply_time = now
        try:
            light_state = self.carla.VehicleLightState(
                self.carla.VehicleLightState.Position
                | self.carla.VehicleLightState.LowBeam
            )
            ego_vehicle.set_light_state(light_state)
            if not self.lights_enabled_logged:
                self.get_logger().info("ego lights enabled")
                self.lights_enabled_logged = True
        except Exception as exc:
            if now - self.last_lights_log_time > 5.0:
                self.get_logger().warn(f"Ego lights enable pending: {exc}")
                self.last_lights_log_time = now

    def draw_ego_label(self, ego_vehicle):
        if not self.enable_ego_label or self.world is None or self.carla is None:
            return
        try:
            location = ego_vehicle.get_location()
            label_location = self.carla.Location(
                location.x,
                location.y,
                location.z + 2.8,
            )
            self.world.debug.draw_string(
                label_location,
                self.ego_label_text,
                draw_shadow=True,
                color=self.carla.Color(0, 180, 255),
                life_time=0.25,
                persistent_lines=False,
            )
            if not self.label_enabled_logged:
                self.get_logger().info("ego label enabled")
                self.label_enabled_logged = True
        except Exception as exc:
            now = time.monotonic()
            if now - self.last_label_warn_time > 5.0:
                self.get_logger().warn(f"Ego label draw pending: {exc}")
                self.last_label_warn_time = now

    def tick(self):
        ego_vehicle = self.find_ego_vehicle()
        if ego_vehicle is None:
            now = time.monotonic()
            elapsed = now - self.started_at
            if (
                elapsed >= self.ego_retry_timeout_s
                and not self.missing_ego_timeout_logged
            ):
                self.get_logger().warn(
                    f"Ego vehicle role={self.ego_role_name} not found after "
                    f"{self.ego_retry_timeout_s:.1f}s; spectator follow remains idle"
                )
                self.missing_ego_timeout_logged = True
            elif now - self.last_missing_ego_log_time > 5.0:
                self.get_logger().info(
                    f"Waiting for ego vehicle role={self.ego_role_name}"
                )
                self.last_missing_ego_log_time = now
            return
        self.apply_demo_weather()
        try:
            ego_transform = ego_vehicle.get_transform()
            if self.camera_view == "hood":
                transform = self.hood_transform(ego_transform)
            else:
                transform = self.chase_transform(ego_transform)
            self.spectator.set_transform(transform)
            if not self.spectator_active_logged:
                self.get_logger().info("spectator follow active")
                self.spectator_active_logged = True
            self.enable_lights_if_needed(ego_vehicle)
            self.draw_ego_label(ego_vehicle)
        except Exception as exc:
            self.get_logger().warn(f"Spectator follow update failed: {exc}")
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
