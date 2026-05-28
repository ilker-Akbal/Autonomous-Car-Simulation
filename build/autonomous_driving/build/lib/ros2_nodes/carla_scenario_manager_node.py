import glob
import json
import os
import random
import sys
import time


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


class CarlaScenarioManagerNode(Node):
    def __init__(self):
        super().__init__("carla_scenario_manager_node")

        self.declare_parameter("carla_root", "/mnt/carla/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")

        self.declare_parameter("scenario", "basic_static")
        self.declare_parameter("spawn_npc_vehicles", True)
        self.declare_parameter("npc_vehicle_count", 8)
        self.declare_parameter("spawn_walkers", True)
        self.declare_parameter("walker_count", 4)
        self.declare_parameter("traffic_manager_port", 8000)
        self.declare_parameter("npc_autopilot", True)
        self.declare_parameter("destroy_on_shutdown", True)

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value

        self.scenario = self.get_parameter("scenario").value
        self.spawn_npc_vehicles = bool(self.get_parameter("spawn_npc_vehicles").value)
        self.npc_vehicle_count = int(self.get_parameter("npc_vehicle_count").value)
        self.spawn_walkers = bool(self.get_parameter("spawn_walkers").value)
        self.walker_count = int(self.get_parameter("walker_count").value)
        self.traffic_manager_port = int(self.get_parameter("traffic_manager_port").value)
        self.npc_autopilot = bool(self.get_parameter("npc_autopilot").value)
        self.destroy_on_shutdown = bool(self.get_parameter("destroy_on_shutdown").value)

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        self.created_actors = []
        self.ego_vehicle = self.wait_for_ego_vehicle()

        self.status_pub = self.create_publisher(String, "/adas/carla/scenario_status", 10)

        self.traffic_manager = self.client.get_trafficmanager(self.traffic_manager_port)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.set_synchronous_mode(False)

        self.setup_scenario()
        self.timer = self.create_timer(1.0, self.publish_status)

        self.get_logger().info(f"CARLA scenario manager hazır: scenario={self.scenario}")

    def wait_for_ego_vehicle(self):
        for _ in range(100):
            vehicles = self.world.get_actors().filter("vehicle.*")
            for vehicle in vehicles:
                if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                    return vehicle
            time.sleep(0.2)

        raise RuntimeError("Ego vehicle bulunamadı")

    def setup_scenario(self):
        if self.spawn_npc_vehicles:
            self.spawn_vehicles()

        if self.spawn_walkers:
            self.spawn_pedestrians()

    def spawn_vehicles(self):
        vehicle_bps = self.bp_lib.filter("vehicle.*")
        vehicle_bps = [
            bp for bp in vehicle_bps
            if bp.has_attribute("number_of_wheels")
            and int(bp.get_attribute("number_of_wheels")) == 4
        ]

        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)

        ego_location = self.ego_vehicle.get_location()
        spawned = 0

        for sp in spawn_points:
            if spawned >= self.npc_vehicle_count:
                break

            if sp.location.distance(ego_location) < 12.0:
                continue

            bp = random.choice(vehicle_bps)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "npc_vehicle")

            actor = self.world.try_spawn_actor(bp, sp)
            if actor is None:
                continue

            if self.npc_autopilot:
                actor.set_autopilot(True, self.traffic_manager_port)

            self.created_actors.append(actor)
            spawned += 1

        self.get_logger().info(f"NPC araç spawn edildi: {spawned}")

    def spawn_pedestrians(self):
        walker_bps = self.bp_lib.filter("walker.pedestrian.*")
        controller_bp = self.bp_lib.find("controller.ai.walker")

        spawned = 0

        for _ in range(self.walker_count * 3):
            if spawned >= self.walker_count:
                break

            location = self.world.get_random_location_from_navigation()
            if location is None:
                continue

            if location.distance(self.ego_vehicle.get_location()) < 8.0:
                continue

            walker_bp = random.choice(walker_bps)
            walker = self.world.try_spawn_actor(
                walker_bp,
                self.carla.Transform(location),
            )

            if walker is None:
                continue

            controller = self.world.try_spawn_actor(
                controller_bp,
                self.carla.Transform(),
                walker,
            )

            if controller is not None:
                controller.start()
                target_location = self.world.get_random_location_from_navigation()
                if target_location is not None:
                    controller.go_to_location(target_location)
                controller.set_max_speed(random.uniform(0.6, 1.4))
                self.created_actors.append(controller)

            self.created_actors.append(walker)
            spawned += 1

        self.get_logger().info(f"Yaya spawn edildi: {spawned}")

    def publish_status(self):
        payload = {
            "stamp": time.time(),
            "scenario": self.scenario,
            "created_actor_count": len(self.created_actors),
            "npc_vehicle_count": len([
                a for a in self.created_actors
                if a.type_id.startswith("vehicle.")
            ]),
            "walker_count": len([
                a for a in self.created_actors
                if a.type_id.startswith("walker.")
            ]),
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.status_pub.publish(msg)

    def destroy_node(self):
        if self.destroy_on_shutdown:
            for actor in reversed(getattr(self, "created_actors", [])):
                try:
                    if actor.is_alive:
                        actor.destroy()
                except Exception:
                    pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarlaScenarioManagerNode()

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