import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import carla
import numpy as np


class CarlaCameraAdapter(Node):
    def __init__(self):
        super().__init__("carla_camera_adapter")

        self.pub = self.create_publisher(Image, "/adas/camera/front/image_raw", 10, )

        self.bridge = CvBridge()

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        self.spawn_camera()

        self.get_logger().info("CARLA camera adapter başlatıldı")

    def spawn_camera(self):
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")

        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")

        vehicle = self.world.get_actors().filter("vehicle.*")[0]

        transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )

        self.camera = self.world.spawn_actor(
            camera_bp,
            transform,
            attach_to=vehicle
        )

        self.camera.listen(self.process_image)

    def process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]

        msg = self.bridge.cv2_to_imgmsg(array, encoding="bgr8")
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CarlaCameraAdapter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()