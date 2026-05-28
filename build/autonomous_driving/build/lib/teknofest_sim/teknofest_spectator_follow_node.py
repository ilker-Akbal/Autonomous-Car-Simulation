import math
import time

import carla
import rclpy
from rclpy.node import Node


class TeknofestSpectatorFollowNode(Node):
    def __init__(self):
        super().__init__("teknofest_spectator_follow_node")

        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("distance_m", 8.0)
        self.declare_parameter("height_m", 3.2)
        self.declare_parameter("target_forward_m", 3.0)
        self.declare_parameter("target_height_m", 1.2)
        self.declare_parameter("timer_period_sec", 0.05)

        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)

        self.distance_m = float(self.get_parameter("distance_m").value)
        self.height_m = float(self.get_parameter("height_m").value)
        self.target_forward_m = float(self.get_parameter("target_forward_m").value)
        self.target_height_m = float(self.get_parameter("target_height_m").value)
        self.timer_period_sec = float(self.get_parameter("timer_period_sec").value)

        self.client = None
        self.world = None
        self.spectator = None
        self.last_ego_id = None
        self.last_warn_time = 0.0

        self.get_logger().info("Teknofest spectator follow node başladı.")
        self.get_logger().info(f"CARLA bağlantısı: {self.host}:{self.port}")

        self.connect_to_carla()
        self.timer = self.create_timer(self.timer_period_sec, self.on_timer)

    def warn_throttled(self, text, period_sec=2.0):
        now = time.time()
        if now - self.last_warn_time >= period_sec:
            self.get_logger().warn(text)
            self.last_warn_time = now

    def connect_to_carla(self):
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()
            self.get_logger().info(f"CARLA world bağlandı: {self.world.get_map().name}")
        except Exception as exc:
            self.client = None
            self.world = None
            self.spectator = None
            self.warn_throttled(f"CARLA bağlantısı bekleniyor: {exc}")

    def refresh_world_and_spectator(self):
        if self.client is None or self.world is None:
            self.connect_to_carla()
            return False

        try:
            self.world = self.client.get_world()
        except Exception as exc:
            self.world = None
            self.spectator = None
            self.warn_throttled(f"World alınamadı, tekrar bağlanılacak: {exc}")
            return False

        try:
            self.spectator = self.world.get_spectator()
            return True
        except Exception as exc:
            self.spectator = None
            self.warn_throttled(
                "Spectator bulunamadı. Unreal Editor'da Play açık mı? "
                "Gerekirse Play dropdown içinden New Editor Window seç. "
                f"Detay: {exc}"
            )
            return False

    def find_ego_vehicle(self):
        if self.world is None:
            return None

        vehicles = list(self.world.get_actors().filter("vehicle.*"))
        if not vehicles:
            return None

        preferred_roles = {"hero", "ego", "ego_vehicle", "autopilot"}

        for vehicle in vehicles:
            role_name = vehicle.attributes.get("role_name", "")
            if role_name in preferred_roles:
                return vehicle

        return vehicles[0]

    @staticmethod
    def look_at_rotation(from_loc, to_loc):
        dx = to_loc.x - from_loc.x
        dy = to_loc.y - from_loc.y
        dz = to_loc.z - from_loc.z

        yaw = math.degrees(math.atan2(dy, dx))
        dist_xy = math.sqrt(dx * dx + dy * dy)
        pitch = math.degrees(math.atan2(dz, dist_xy))

        return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)

    def on_timer(self):
        if not self.refresh_world_and_spectator():
            return

        ego = self.find_ego_vehicle()
        if ego is None:
            self.warn_throttled("Takip edilecek araç bulunamadı. Ego spawn bekleniyor.")
            return

        if ego.id != self.last_ego_id:
            self.last_ego_id = ego.id
            self.get_logger().info(
                f"Takip edilen araç: id={ego.id}, "
                f"type={ego.type_id}, "
                f"role={ego.attributes.get('role_name', '-')}"
            )

        try:
            tf = ego.get_transform()
            loc = tf.location
            forward = tf.get_forward_vector()

            cam_loc = carla.Location(
                x=loc.x - forward.x * self.distance_m,
                y=loc.y - forward.y * self.distance_m,
                z=loc.z + self.height_m,
            )

            target_loc = carla.Location(
                x=loc.x + forward.x * self.target_forward_m,
                y=loc.y + forward.y * self.target_forward_m,
                z=loc.z + self.target_height_m,
            )

            cam_rot = self.look_at_rotation(cam_loc, target_loc)
            self.spectator.set_transform(carla.Transform(cam_loc, cam_rot))

        except Exception as exc:
            self.warn_throttled(f"Spectator takip güncellenemedi: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = TeknofestSpectatorFollowNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
