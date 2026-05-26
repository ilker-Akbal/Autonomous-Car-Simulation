import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarNode(Node):
    def __init__(self):
        super().__init__("lidar_node")

        self.subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        self.get_logger().info("lidar_node started. Listening /scan")

    def scan_callback(self, msg: LaserScan) -> None:
        valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if not valid:
            return
        min_range = min(valid)
        self.get_logger().info(f"Closest obstacle: {min_range:.2f} m", throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()