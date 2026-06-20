import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    import psutil
except ImportError:
    psutil = None


class DiagnosticsNode(Node):
    def __init__(self):
        super().__init__("teknofest_diagnostics_node")
        self.publisher_ = self.create_publisher(String, "/adas/system/status", 10)
        self.timer = self.create_timer(1.0, self._timer_callback)
        self._warned_missing_psutil = False
        self.get_logger().info("Teknofest diagnostics node active")

    def _timer_callback(self) -> None:
        cpu_percent = None
        memory_percent = None
        system_load = None
        status = "Operational"

        if psutil is not None:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_percent = psutil.virtual_memory().percent
                system_load = (
                    psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
                )
            except Exception as exc:
                status = "Diagnostics fallback"
                self.get_logger().warn(f"psutil error, publishing fallback diagnostics: {exc}")
        else:
            if not self._warned_missing_psutil:
                self.get_logger().warn(
                    "psutil not installed; publishing limited diagnostics payload."
                )
                self._warned_missing_psutil = True
            status = "Diagnostics fallback"

        payload = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "system_load": system_load,
            "status": status,
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.publisher_.publish(msg)

        if cpu_percent is not None and cpu_percent > 90:
            self.get_logger().warn(f"Critical CPU load: {cpu_percent}%")


def main(args=None):
    rclpy.init(args=args)
    node = DiagnosticsNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
