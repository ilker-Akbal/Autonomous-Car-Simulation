import json
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


class CarlaControlAdapter(Node):
    def __init__(self):
        super().__init__("carla_control_adapter_node")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("command_timeout_s", 0.5)

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value
        self.command_timeout_s = float(self.get_parameter("command_timeout_s").value)

        self.vehicle = None
        self.last_cmd = None
        self.last_cmd_time = 0.0

        self.cmd_sub = self.create_subscription(String, "/adas/control/vehicle_command", self._cmd_cb, 10)
        self.debug_pub = self.create_publisher(String, "/adas/control/adapter_debug", 10)

        # connect to CARLA
        try:
            carla = load_carla(self.carla_root)
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
        except Exception as exc:
            self.client = None
            self.world = None
            self.get_logger().warn(f"Control adapter: cannot connect to CARLA: {exc}")

        self.timer = self.create_timer(0.1, self._apply_periodic)

    def _find_ego(self):
        if self.world is None:
            return None
        vehicles = self.world.get_actors().filter("vehicle.*")
        for v in vehicles:
            if v.attributes.get("role_name", "") in (self.ego_role_name, "ego", "ego_vehicle", "hero"):
                return v
        if vehicles:
            return vehicles[0]
        return None

    def _cmd_cb(self, msg: String):
        try:
            self.last_cmd = json.loads(msg.data)
            self.last_cmd_time = time.time()
        except Exception:
            self.get_logger().warn("Control adapter: failed to parse command JSON")

    def _apply_periodic(self):
        now = time.time()
        # find vehicle if not known
        if self.vehicle is None and self.world is not None:
            try:
                self.vehicle = self._find_ego()
                if self.vehicle is not None:
                    self.vehicle.set_autopilot(False)
            except Exception:
                self.vehicle = None

        # determine command to apply
        cmd = None
        if self.last_cmd is not None and (now - self.last_cmd_time) <= self.command_timeout_s:
            cmd = self.last_cmd
        else:
            # safe stop
            cmd = {"throttle": 0.0, "brake": 1.0, "steer": 0.0, "reverse": False, "hand_brake": False}

        # apply if vehicle available
        if self.vehicle is not None:
            try:
                carla = load_carla(self.carla_root)
                VehicleControl = carla.VehicleControl

                throttle = float(cmd.get("throttle", 0.0))
                brake = float(cmd.get("brake", 0.0))
                steer = float(cmd.get("steer", 0.0))
                reverse = bool(cmd.get("reverse", False))
                hand_brake = bool(cmd.get("hand_brake", False))

                throttle = max(0.0, min(1.0, throttle))
                brake = max(0.0, min(1.0, brake))
                steer = max(-1.0, min(1.0, steer))

                control = VehicleControl(throttle=throttle, brake=brake, steer=steer, reverse=reverse, hand_brake=hand_brake)
                self.vehicle.apply_control(control)
            except Exception as exc:
                self.get_logger().warn(f"Control adapter failed to apply control: {exc}")

        dbg = String(); dbg.data = json.dumps({"stamp": now, "applied_cmd_present": cmd is not None})
        self.debug_pub.publish(dbg)

    def destroy_node(self):
        # apply safe stop before shutdown
        try:
            if self.vehicle is None and self.world is not None:
                self.vehicle = self._find_ego()
            if self.vehicle is not None:
                carla = load_carla(self.carla_root)
                VehicleControl = carla.VehicleControl
                control = VehicleControl(throttle=0.0, brake=1.0, steer=0.0, reverse=False, hand_brake=True)
                try:
                    self.vehicle.apply_control(control)
                except Exception:
                    pass
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarlaControlAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
