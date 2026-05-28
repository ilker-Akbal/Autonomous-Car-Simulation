import json
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def load_carla(carla_root):
    candidates = [
        os.path.join(carla_root, "PythonAPI", "carla"),
        os.path.join(carla_root, "PythonAPI"),
    ]

    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)

    import carla
    return carla


class CarlaSpectatorFollowNode(Node):
    def __init__(self):
        super().__init__("carla_spectator_follow_node")

        self.declare_parameter("carla_root", "/home/huseyindgn/CARLA_DISK")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 30.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")

        self.declare_parameter("follow_distance", 9.0)
        self.declare_parameter("follow_height", 5.0)
        self.declare_parameter("side_offset", 0.0)
        self.declare_parameter("look_at_height", 1.2)
        self.declare_parameter("tick_s", 0.05)

        self.declare_parameter("status_topic", "/adas/carla/spectator_follow_status")

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value

        self.follow_distance = float(self.get_parameter("follow_distance").value)
        self.follow_height = float(self.get_parameter("follow_height").value)
        self.side_offset = float(self.get_parameter("side_offset").value)
        self.look_at_height = float(self.get_parameter("look_at_height").value)
        self.tick_s = float(self.get_parameter("tick_s").value)

        self.status_topic = self.get_parameter("status_topic").value
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()

        self.ego = None
        self.last_ego_search = 0.0

        self.timer = self.create_timer(self.tick_s, self.timer_cb)

        self.get_logger().info(
            f"CARLA spectator follow hazır. role={self.ego_role_name}, "
            f"distance={self.follow_distance}, height={self.follow_height}"
        )

    def find_ego(self):
        now = time.time()
        if self.ego is not None and self.ego.is_alive:
            return self.ego

        if now - self.last_ego_search < 1.0:
            return None

        self.last_ego_search = now

        for actor in self.world.get_actors().filter("vehicle.*"):
            try:
                if actor.attributes.get("role_name") == self.ego_role_name:
                    self.ego = actor
                    self.get_logger().info(f"Ego bulundu. id={actor.id}, type={actor.type_id}")
                    return actor
            except Exception:
                pass

        self.ego = None
        return None

    def look_at_rotation(self, camera_loc, target_loc):
        dx = target_loc.x - camera_loc.x
        dy = target_loc.y - camera_loc.y
        dz = target_loc.z - camera_loc.z

        yaw = math.degrees(math.atan2(dy, dx))
        dist_xy = max(math.sqrt(dx * dx + dy * dy), 1e-6)
        pitch = math.degrees(math.atan2(dz, dist_xy))

        return self.carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)

    def timer_cb(self):
        try:
            ego = self.find_ego()

            if ego is None:
                self.publish_status(False, "ego_not_found")
                return

            tf = ego.get_transform()
            loc = tf.location
            yaw_rad = math.radians(tf.rotation.yaw)

            forward_x = math.cos(yaw_rad)
            forward_y = math.sin(yaw_rad)

            right_x = math.cos(yaw_rad + math.pi / 2.0)
            right_y = math.sin(yaw_rad + math.pi / 2.0)

            cam_loc = self.carla.Location(
                x=loc.x - forward_x * self.follow_distance + right_x * self.side_offset,
                y=loc.y - forward_y * self.follow_distance + right_y * self.side_offset,
                z=loc.z + self.follow_height,
            )

            target_loc = self.carla.Location(
                x=loc.x,
                y=loc.y,
                z=loc.z + self.look_at_height,
            )

            cam_rot = self.look_at_rotation(cam_loc, target_loc)
            self.spectator.set_transform(self.carla.Transform(cam_loc, cam_rot))

            self.publish_status(
                True,
                "following",
                {
                    "ego_id": ego.id,
                    "ego_type": ego.type_id,
                    "ego_x": round(loc.x, 3),
                    "ego_y": round(loc.y, 3),
                    "ego_yaw": round(tf.rotation.yaw, 2),
                    "cam_x": round(cam_loc.x, 3),
                    "cam_y": round(cam_loc.y, 3),
                    "cam_z": round(cam_loc.z, 3),
                },
            )

        except Exception as e:
            self.publish_status(False, f"error:{repr(e)}")

    def publish_status(self, ok, reason, extra=None):
        payload = {
            "stamp": time.time(),
            "ok": bool(ok),
            "reason": reason,
        }

        if extra:
            payload.update(extra)

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CarlaSpectatorFollowNode()

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
