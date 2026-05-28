import glob
import json
import os
import sys
import time
from typing import Optional


def load_carla(carla_root: str):
    egg_pattern = os.path.join(
        carla_root,
        "PythonAPI",
        "carla",
        "dist",
        "carla-*%d.%d-%s.egg" % (
            sys.version_info.major,
            sys.version_info.minor,
            "linux-x86_64",
        ),
    )

    eggs = glob.glob(egg_pattern)
    if eggs and eggs[0] not in sys.path:
        sys.path.append(eggs[0])

    import carla
    return carla


import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class CarlaWorldManagerNode(Node):
    def __init__(self):
        super().__init__("carla_world_manager_node")

        self.declare_parameter("carla_root", "/mnt/carla/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("town", "")
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("ego_blueprint", "vehicle.tesla.model3")
        self.declare_parameter("spawn_index", 0)
        self.declare_parameter("destroy_on_shutdown", False)
        self.declare_parameter("set_spectator", True)
        self.declare_parameter("enable_sync_mode", False)
        self.declare_parameter("fixed_delta_seconds", 0.05)

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.town = self.get_parameter("town").value
        self.ego_role_name = self.get_parameter("ego_role_name").value
        self.ego_blueprint = self.get_parameter("ego_blueprint").value
        self.spawn_index = int(self.get_parameter("spawn_index").value)
        self.destroy_on_shutdown = bool(self.get_parameter("destroy_on_shutdown").value)
        self.set_spectator = bool(self.get_parameter("set_spectator").value)
        self.enable_sync_mode = bool(self.get_parameter("enable_sync_mode").value)
        self.fixed_delta_seconds = float(self.get_parameter("fixed_delta_seconds").value)

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)

        self.world = self.client.get_world()
        self.cleanup_previous_runtime_actors()

        if self.town:
            current_map = self.world.get_map().name
            if self.town not in current_map:
                self.get_logger().info(f"CARLA haritası yükleniyor: {self.town}")
                self.world = self.client.load_world(self.town)
                time.sleep(2.0)

        self.configure_world()
        self.ego_vehicle = self.get_or_spawn_ego_vehicle()

        if self.set_spectator:
            self.move_spectator_to_ego()

        self.status_pub = self.create_publisher(String, "/adas/carla/status", 10)
        self.timer = self.create_timer(1.0, self.publish_status)

        self.get_logger().info("CARLA world manager hazır")
        self.get_logger().info(f"map={self.world.get_map().name}")
        self.get_logger().info(f"ego_id={self.ego_vehicle.id}")

    def configure_world(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = self.enable_sync_mode

        if self.enable_sync_mode:
            settings.fixed_delta_seconds = self.fixed_delta_seconds
        else:
            settings.fixed_delta_seconds = None

        self.world.apply_settings(settings)

    def find_ego_vehicle(self):
        vehicles = self.world.get_actors().filter("vehicle.*")
        for vehicle in vehicles:
            if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                return vehicle
        return None

    def get_or_spawn_ego_vehicle(self):
        existing = self.find_ego_vehicle()
        if existing is not None:
            self.get_logger().info(f"Mevcut ego araç bulundu: id={existing.id}")
            return existing

        blueprint_library = self.world.get_blueprint_library()
        ego_bp = blueprint_library.find(self.ego_blueprint)
        ego_bp.set_attribute("role_name", self.ego_role_name)

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("CARLA spawn point bulunamadı")

        spawn_index = max(0, min(self.spawn_index, len(spawn_points) - 1))
        spawn_transform = spawn_points[spawn_index]

        ego = self.world.try_spawn_actor(ego_bp, spawn_transform)

        if ego is None:
            for transform in spawn_points:
                ego = self.world.try_spawn_actor(ego_bp, transform)
                if ego is not None:
                    break

        if ego is None:
            raise RuntimeError("Ego araç spawn edilemedi")

        ego.set_autopilot(False)
        return ego

    def move_spectator_to_ego(self):
        spectator = self.world.get_spectator()
        ego_transform = self.ego_vehicle.get_transform()

        spectator_transform = self.carla.Transform(
            ego_transform.location + self.carla.Location(x=-8.0, z=5.0),
            self.carla.Rotation(
                pitch=-25.0,
                yaw=ego_transform.rotation.yaw,
                roll=0.0,
            ),
        )
        spectator.set_transform(spectator_transform)

    def publish_status(self):
        ego_transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5

        payload = {
            "stamp": time.time(),
            "map": self.world.get_map().name,
            "ego_id": self.ego_vehicle.id,
            "role_name": self.ego_vehicle.attributes.get("role_name", ""),
            "location": {
                "x": round(ego_transform.location.x, 3),
                "y": round(ego_transform.location.y, 3),
                "z": round(ego_transform.location.z, 3),
            },
            "rotation": {
                "pitch": round(ego_transform.rotation.pitch, 3),
                "yaw": round(ego_transform.rotation.yaw, 3),
                "roll": round(ego_transform.rotation.roll, 3),
            },
            "speed_mps": round(speed, 3),
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.status_pub.publish(msg)

    def destroy_node(self):
        if self.destroy_on_shutdown and hasattr(self, "ego_vehicle"):
            try:
                self.ego_vehicle.destroy()
            except Exception:
                pass
        super().destroy_node()



    def cleanup_previous_runtime_actors(self):
        patterns = [
            "vehicle.*",
            "sensor.*",
            "walker.pedestrian.*",
            "controller.ai.walker",
            "static.prop.teknofest_sign*",
            "static.prop.streetsign*",
        ]

        actors_to_destroy = []

        for pattern in patterns:
            try:
                actors = list(self.world.get_actors().filter(pattern))
                actors_to_destroy.extend(actors)
            except Exception as exc:
                self.get_logger().warn(f"Runtime cleanup pattern failed: {pattern} | {exc}")

        unique = {}
        for actor in actors_to_destroy:
            unique[actor.id] = actor

        actors_to_destroy = list(unique.values())

        if not actors_to_destroy:
            self.get_logger().info("Runtime cleanup: temizlenecek eski actor yok.")
            return

        self.get_logger().warn(
            f"Runtime cleanup: {len(actors_to_destroy)} eski actor silinecek."
        )

        commands = [
            carla.command.DestroyActor(actor.id)
            for actor in actors_to_destroy
        ]

        try:
            responses = self.client.apply_batch_sync(commands, True)
        except Exception as exc:
            self.get_logger().error(f"Runtime cleanup batch failed: {exc}")
            return

        failed = 0
        for actor, response in zip(actors_to_destroy, responses):
            if response.error:
                failed += 1
                self.get_logger().warn(
                    f"Runtime cleanup failed id={actor.id} "
                    f"type={actor.type_id}: {response.error}"
                )

        self.get_logger().info(
            f"Runtime cleanup finished. success={len(actors_to_destroy) - failed}, failed={failed}"
        )

        try:
            self.world.tick()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = CarlaWorldManagerNode()

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