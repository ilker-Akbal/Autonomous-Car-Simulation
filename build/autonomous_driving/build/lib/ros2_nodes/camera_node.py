import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        self.declare_parameter("input_topic", "/front_camera")
        self.declare_parameter("output_topic", "/adas/camera/front/image_raw")

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.publisher = self.create_publisher(Image, self.output_topic, 10)

        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10,
        )

        self.get_logger().info(
            f"camera_node başladı: {self.input_topic} -> {self.output_topic}"
        )

    def image_callback(self, msg: Image) -> None:
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()

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