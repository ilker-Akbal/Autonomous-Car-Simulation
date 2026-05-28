#!/usr/bin/env python3
import math
import time
from typing import Optional

import carla
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, NavSatFix, Imu, PointCloud2, PointField


class CarlaSensorBridgeNode(Node):

    def __init__(self):
        super().__init__("carla_sensor_bridge_node")

        # -------------------------
        # CARLA connection params
        # -------------------------
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 10.0)

        # -------------------------
        # Common camera params
        # -------------------------
        self.declare_parameter("camera_width", 960)
        self.declare_parameter("camera_height", 540)

        # ZED 2i 4mm için simülasyonda dar FOV yaklaşımı.
        # Gerçek ZED kalibrasyonu değildir; CARLA approximation.
        self.declare_parameter("camera_fov", 72.0)

        self.declare_parameter("camera_x", 1.6)
        self.declare_parameter("camera_y", 0.0)
        self.declare_parameter("camera_z", 2.25)
        self.declare_parameter("camera_pitch", -1.0)
        self.declare_parameter("camera_yaw", 0.0)
        self.declare_parameter("camera_roll", 0.0)
        self.declare_parameter("camera_sensor_tick", 0.05)

        # -------------------------
        # Topic params
        # -------------------------
        self.declare_parameter("image_topic", "/adas/camera/front/image_raw")
        self.declare_parameter("camera_info_topic", "/adas/camera/front/camera_info")
        self.declare_parameter("front_rgb_separate_enabled", False)
        self.declare_parameter("front_rgb_from_zed_left", True)

        self.declare_parameter("gnss_topic", "/adas/localization/gnss")
        self.declare_parameter("imu_topic", "/adas/localization/imu")

        # -------------------------
        # Pseudo-ZED params
        # -------------------------
        self.declare_parameter("zed_enabled", True)
        self.declare_parameter("zed_baseline_m", 0.12)

        self.declare_parameter("zed_left_topic", "/zed/zed_node/left/image_rect_color")
        self.declare_parameter("zed_right_topic", "/zed/zed_node/right/image_rect_color")
        self.declare_parameter("zed_left_info_topic", "/zed/zed_node/left/camera_info")
        self.declare_parameter("zed_right_info_topic", "/zed/zed_node/right/camera_info")

        self.declare_parameter("depth_enabled", True)
        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("depth_info_topic", "/zed/zed_node/depth/camera_info")

        self.declare_parameter("zed_point_cloud_enabled", True)
        self.declare_parameter("zed_point_cloud_topic", "/zed/zed_node/point_cloud/cloud_registered")
        self.declare_parameter("zed_point_cloud_decimation", 8)
        self.declare_parameter("zed_point_cloud_max_depth", 80.0)

        # -------------------------
        # LiDAR params - RoboSense Helios approximation
        # -------------------------
        self.declare_parameter("lidar_enabled", True)
        self.declare_parameter("lidar_topic", "/adas/lidar/points")

        self.declare_parameter("lidar_x", 0.8)
        self.declare_parameter("lidar_y", 0.0)
        self.declare_parameter("lidar_z", 2.45)
        self.declare_parameter("lidar_pitch", 0.0)
        self.declare_parameter("lidar_yaw", 0.0)
        self.declare_parameter("lidar_roll", 0.0)

        # Helios 16 benzeri güvenli varsayılan.
        # Helios 32 istenirse launch parametrelerinden channels=32 yapılabilir.
        self.declare_parameter("lidar_channels", 16)
        self.declare_parameter("lidar_range", 100.0)
        self.declare_parameter("lidar_rotation_frequency", 10.0)
        self.declare_parameter("lidar_points_per_second", 320000)
        self.declare_parameter("lidar_upper_fov", 15.0)
        self.declare_parameter("lidar_lower_fov", -15.0)
        self.declare_parameter("lidar_sensor_tick", 0.1)


        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.ego_vehicle: Optional[carla.Actor] = None
        self.sensors = []

        self.width = int(self.get_parameter("camera_width").value)
        self.height = int(self.get_parameter("camera_height").value)
        self.fov = float(self.get_parameter("camera_fov").value)

        self.fx, self.fy, self.cx, self.cy = self.compute_camera_intrinsics(
            self.width, self.height, self.fov
        )


        self.front_image_pub = self.create_publisher(
            Image,
            self.get_parameter("image_topic").value,
            10,
        )
        self.front_info_pub = self.create_publisher(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            10,
        )

        self.gnss_pub = self.create_publisher(
            NavSatFix,
            self.get_parameter("gnss_topic").value,
            10,
        )
        self.imu_pub = self.create_publisher(
            Imu,
            self.get_parameter("imu_topic").value,
            10,
        )

        self.zed_left_pub = self.create_publisher(
            Image,
            self.get_parameter("zed_left_topic").value,
            10,
        )
        self.zed_right_pub = self.create_publisher(
            Image,
            self.get_parameter("zed_right_topic").value,
            10,
        )
        self.zed_left_info_pub = self.create_publisher(
            CameraInfo,
            self.get_parameter("zed_left_info_topic").value,
            10,
        )
        self.zed_right_info_pub = self.create_publisher(
            CameraInfo,
            self.get_parameter("zed_right_info_topic").value,
            10,
        )

        self.depth_pub = self.create_publisher(
            Image,
            self.get_parameter("depth_topic").value,
            10,
        )
        self.depth_info_pub = self.create_publisher(
            CameraInfo,
            self.get_parameter("depth_info_topic").value,
            10,
        )

        self.zed_pc_pub = self.create_publisher(
            PointCloud2,
            self.get_parameter("zed_point_cloud_topic").value,
            10,
        )

        self.lidar_pub = self.create_publisher(
            PointCloud2,
            self.get_parameter("lidar_topic").value,
            10,
        )


        self.connect_to_carla()
        self.find_ego_vehicle()

        if bool(self.get_parameter("front_rgb_separate_enabled").value) or not bool(self.get_parameter("zed_enabled").value):
            self.spawn_front_camera()
        else:
            self.get_logger().info("Separate front RGB camera disabled; /adas/camera/front/image_raw mirrors ZED left.")

        if bool(self.get_parameter("zed_enabled").value):
            self.spawn_zed_left_right()

        if bool(self.get_parameter("depth_enabled").value):
            self.spawn_depth_camera()

        self.spawn_gnss()
        self.spawn_imu()

        if bool(self.get_parameter("lidar_enabled").value):
            self.spawn_lidar()

        self.get_logger().info("CARLA sensor bridge hazır.")
        self.get_logger().info(f"Front RGB     -> {self.get_parameter('image_topic').value}")
        self.get_logger().info(f"GNSS          -> {self.get_parameter('gnss_topic').value}")
        self.get_logger().info(f"IMU           -> {self.get_parameter('imu_topic').value}")
        self.get_logger().info(f"ZED left      -> {self.get_parameter('zed_left_topic').value}")
        self.get_logger().info(f"ZED right     -> {self.get_parameter('zed_right_topic').value}")
        self.get_logger().info(f"ZED depth     -> {self.get_parameter('depth_topic').value}")
        self.get_logger().info(f"ZED pointcloud-> {self.get_parameter('zed_point_cloud_topic').value}")
        self.get_logger().info(f"LiDAR points  -> {self.get_parameter('lidar_topic').value}")


    def connect_to_carla(self):
        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        timeout = float(self.get_parameter("timeout").value)

        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()

        self.get_logger().info(f"CARLA bağlantısı kuruldu: {host}:{port}")
        self.get_logger().info(f"Map: {self.world.get_map().name}")

    def find_ego_vehicle(self):
        assert self.world is not None

        deadline = time.time() + 15.0
        while time.time() < deadline:
            vehicles = self.world.get_actors().filter("vehicle.*")

            for v in vehicles:
                role = v.attributes.get("role_name", "")
                if role in ("ego", "ego_vehicle", "hero"):
                    self.ego_vehicle = v
                    self.get_logger().info(
                        f"Ego vehicle bulundu: id={v.id}, type={v.type_id}, role_name={role}"
                    )
                    return

            if len(vehicles) > 0:
                self.ego_vehicle = vehicles[0]
                self.get_logger().warn(
                    f"role_name=ego bulunamadı. İlk araç ego seçildi: id={self.ego_vehicle.id}, type={self.ego_vehicle.type_id}"
                )
                return

            self.get_logger().warn("Ego vehicle bekleniyor...")
            time.sleep(0.5)

        raise RuntimeError("CARLA içinde ego vehicle bulunamadı. Önce scenario node ego aracı spawn etmeli.")


    @staticmethod
    def compute_camera_intrinsics(width: int, height: int, horizontal_fov_deg: float):
        fov_rad = math.radians(horizontal_fov_deg)
        fx = width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        return fx, fy, cx, cy

    def stamp_now(self):
        return self.get_clock().now().to_msg()

    def make_camera_info(self, frame_id: str) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp = self.stamp_now()
        msg.header.frame_id = frame_id

        msg.width = self.width
        msg.height = self.height
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        msg.k = [
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0,
        ]

        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        msg.p = [
            self.fx, 0.0, self.cx, 0.0,
            0.0, self.fy, self.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return msg


    def carla_rgb_to_ros_image(self, image, frame_id: str) -> Image:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))

        # CARLA raw is BGRA. Convert to RGB.
        rgb = arr[:, :, :3][:, :, ::-1].copy()


        msg = Image()
        msg.header.stamp = self.stamp_now()
        msg.header.frame_id = frame_id
        msg.height = image.height
        msg.width = image.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = image.width * 3
        msg.data = rgb.tobytes()
        return msg

    def depth_raw_to_meters(self, image) -> np.ndarray:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4)).astype(np.float32)

        # CARLA depth raw is BGRA.
        b = arr[:, :, 0]
        g = arr[:, :, 1]
        r = arr[:, :, 2]

        normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
        depth_m = normalized * 1000.0
        return depth_m.astype(np.float32)

    def depth_to_ros_image(self, depth_m: np.ndarray, frame_id: str) -> Image:
        msg = Image()
        msg.header.stamp = self.stamp_now()
        msg.header.frame_id = frame_id
        msg.height = depth_m.shape[0]
        msg.width = depth_m.shape[1]
        msg.encoding = "32FC1"
        msg.is_bigendian = False
        msg.step = msg.width * 4
        msg.data = depth_m.astype(np.float32).tobytes()
        return msg

    def xyz_to_pointcloud2(self, xyz: np.ndarray, frame_id: str) -> PointCloud2:
        if xyz.size == 0:
            xyz = np.zeros((0, 3), dtype=np.float32)

        xyz = np.asarray(xyz, dtype=np.float32).reshape((-1, 3))

        msg = PointCloud2()
        msg.header.stamp = self.stamp_now()
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = int(xyz.shape[0])
        msg.is_bigendian = False
        msg.is_dense = False
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.data = xyz.tobytes()
        return msg

    def xyzi_to_pointcloud2(self, xyzi: np.ndarray, frame_id: str) -> PointCloud2:
        if xyzi.size == 0:
            xyzi = np.zeros((0, 4), dtype=np.float32)

        xyzi = np.asarray(xyzi, dtype=np.float32).reshape((-1, 4))

        msg = PointCloud2()
        msg.header.stamp = self.stamp_now()
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = int(xyzi.shape[0])
        msg.is_bigendian = False
        msg.is_dense = False
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.data = xyzi.tobytes()
        return msg
    
    def get_camera_transform(self, y_offset: float = 0.0):
        return carla.Transform(
            carla.Location(
                x=float(self.get_parameter("camera_x").value),
                y=float(self.get_parameter("camera_y").value) + y_offset,
                z=float(self.get_parameter("camera_z").value),
            ),
            carla.Rotation(
                pitch=float(self.get_parameter("camera_pitch").value),
                yaw=float(self.get_parameter("camera_yaw").value),
                roll=float(self.get_parameter("camera_roll").value),
            ),
        )

    def configure_rgb_camera_bp(self):
        assert self.world is not None
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.width))
        camera_bp.set_attribute("image_size_y", str(self.height))
        camera_bp.set_attribute("fov", str(self.fov))
        camera_bp.set_attribute("sensor_tick", str(float(self.get_parameter("camera_sensor_tick").value)))
        return camera_bp

    def spawn_front_camera(self):
        assert self.world is not None
        assert self.ego_vehicle is not None

        camera_bp = self.configure_rgb_camera_bp()
        transform = self.get_camera_transform(y_offset=0.0)

        camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.ego_vehicle)
        camera.listen(self.front_camera_callback)
        self.sensors.append(camera)

        self.get_logger().info("Front RGB camera spawn edildi.")

    def spawn_zed_left_right(self):
        assert self.world is not None
        assert self.ego_vehicle is not None

        baseline = float(self.get_parameter("zed_baseline_m").value)

        left_bp = self.configure_rgb_camera_bp()
        right_bp = self.configure_rgb_camera_bp()

        left_tf = self.get_camera_transform(y_offset=-baseline / 2.0)
        right_tf = self.get_camera_transform(y_offset=baseline / 2.0)

        left_cam = self.world.spawn_actor(left_bp, left_tf, attach_to=self.ego_vehicle)
        right_cam = self.world.spawn_actor(right_bp, right_tf, attach_to=self.ego_vehicle)

        left_cam.listen(self.zed_left_callback)
        right_cam.listen(self.zed_right_callback)

        self.sensors.extend([left_cam, right_cam])

        self.get_logger().info(f"Pseudo-ZED stereo cameras spawn edildi. baseline={baseline:.3f} m")

    def spawn_depth_camera(self):
        assert self.world is not None
        assert self.ego_vehicle is not None

        bp_lib = self.world.get_blueprint_library()
        depth_bp = bp_lib.find("sensor.camera.depth")
        depth_bp.set_attribute("image_size_x", str(self.width))
        depth_bp.set_attribute("image_size_y", str(self.height))
        depth_bp.set_attribute("fov", str(self.fov))
        depth_bp.set_attribute("sensor_tick", str(float(self.get_parameter("camera_sensor_tick").value)))

        transform = self.get_camera_transform(y_offset=0.0)

        depth_cam = self.world.spawn_actor(depth_bp, transform, attach_to=self.ego_vehicle)
        depth_cam.listen(self.depth_callback)
        self.sensors.append(depth_cam)

        self.get_logger().info("Pseudo-ZED depth camera spawn edildi.")

    def spawn_gnss(self):
        assert self.world is not None
        assert self.ego_vehicle is not None

        bp_lib = self.world.get_blueprint_library()
        gnss_bp = bp_lib.find("sensor.other.gnss")

        transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=2.0),
            carla.Rotation(),
        )

        gnss = self.world.spawn_actor(gnss_bp, transform, attach_to=self.ego_vehicle)
        gnss.listen(self.gnss_callback)
        self.sensors.append(gnss)

        self.get_logger().info("GNSS spawn edildi.")

    def spawn_imu(self):
        assert self.world is not None
        assert self.ego_vehicle is not None

        bp_lib = self.world.get_blueprint_library()
        imu_bp = bp_lib.find("sensor.other.imu")

        transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=2.0),
            carla.Rotation(),
        )

        imu = self.world.spawn_actor(imu_bp, transform, attach_to=self.ego_vehicle)
        imu.listen(self.imu_callback)
        self.sensors.append(imu)

        self.get_logger().info("IMU spawn edildi.")

    def spawn_lidar(self):
        assert self.world is not None
        assert self.ego_vehicle is not None

        bp_lib = self.world.get_blueprint_library()
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")

        lidar_bp.set_attribute("channels", str(int(self.get_parameter("lidar_channels").value)))
        lidar_bp.set_attribute("range", str(float(self.get_parameter("lidar_range").value)))
        lidar_bp.set_attribute("rotation_frequency", str(float(self.get_parameter("lidar_rotation_frequency").value)))
        lidar_bp.set_attribute("points_per_second", str(int(self.get_parameter("lidar_points_per_second").value)))
        lidar_bp.set_attribute("upper_fov", str(float(self.get_parameter("lidar_upper_fov").value)))
        lidar_bp.set_attribute("lower_fov", str(float(self.get_parameter("lidar_lower_fov").value)))
        lidar_bp.set_attribute("sensor_tick", str(float(self.get_parameter("lidar_sensor_tick").value)))

        transform = carla.Transform(
            carla.Location(
                x=float(self.get_parameter("lidar_x").value),
                y=float(self.get_parameter("lidar_y").value),
                z=float(self.get_parameter("lidar_z").value),
            ),
            carla.Rotation(
                pitch=float(self.get_parameter("lidar_pitch").value),
                yaw=float(self.get_parameter("lidar_yaw").value),
                roll=float(self.get_parameter("lidar_roll").value),
            ),
        )

        lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=self.ego_vehicle)
        lidar.listen(self.lidar_callback)
        self.sensors.append(lidar)

        self.get_logger().info(
            "CARLA 3D LiDAR spawn edildi. "
            f"channels={self.get_parameter('lidar_channels').value}, "
            f"hz={self.get_parameter('lidar_rotation_frequency').value}, "
            f"range={self.get_parameter('lidar_range').value}"
        )

    def front_camera_callback(self, image):
        frame_id = "carla_front_camera"

        msg = self.carla_rgb_to_ros_image(image, frame_id)
        self.front_image_pub.publish(msg)

        info = self.make_camera_info(frame_id)
        self.front_info_pub.publish(info)

    def zed_left_callback(self, image):
        frame_id = "zed_left_camera"

        msg = self.carla_rgb_to_ros_image(image, frame_id)
        self.zed_left_pub.publish(msg)

        if bool(self.get_parameter("front_rgb_from_zed_left").value):
            self.front_image_pub.publish(msg)
            self.front_info_pub.publish(self.make_camera_info(frame_id))

        info = self.make_camera_info(frame_id)
        self.zed_left_info_pub.publish(info)

    def zed_right_callback(self, image):
        frame_id = "zed_right_camera"

        msg = self.carla_rgb_to_ros_image(image, frame_id)
        self.zed_right_pub.publish(msg)

        info = self.make_camera_info(frame_id)
        self.zed_right_info_pub.publish(info)

    def depth_callback(self, image):
        frame_id = "zed_depth_camera"

        depth_m = self.depth_raw_to_meters(image)

        depth_msg = self.depth_to_ros_image(depth_m, frame_id)
        self.depth_pub.publish(depth_msg)

        info = self.make_camera_info(frame_id)
        self.depth_info_pub.publish(info)

        if bool(self.get_parameter("zed_point_cloud_enabled").value):
            pc_msg = self.depth_to_sparse_pointcloud(depth_m, frame_id)
            self.zed_pc_pub.publish(pc_msg)

    def depth_to_sparse_pointcloud(self, depth_m: np.ndarray, frame_id: str) -> PointCloud2:
        dec = max(1, int(self.get_parameter("zed_point_cloud_decimation").value))
        max_depth = float(self.get_parameter("zed_point_cloud_max_depth").value)

        d = depth_m[::dec, ::dec]

        h, w = d.shape
        u = np.arange(0, self.width, dec, dtype=np.float32)[:w]
        v = np.arange(0, self.height, dec, dtype=np.float32)[:h]
        uu, vv = np.meshgrid(u, v)

        valid = np.isfinite(d) & (d > 0.2) & (d < max_depth)

        z = d[valid]
        x = (uu[valid] - self.cx) * z / self.fx
        y = (vv[valid] - self.cy) * z / self.fy

        xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        return self.xyz_to_pointcloud2(xyz, frame_id)

    def lidar_callback(self, data):
        frame_id = "carla_lidar"

        points = np.frombuffer(data.raw_data, dtype=np.float32)
        points = points.reshape((-1, 4))


        msg = self.xyzi_to_pointcloud2(points, frame_id)
        self.lidar_pub.publish(msg)

    def gnss_callback(self, data):
        msg = NavSatFix()
        msg.header.stamp = self.stamp_now()
        msg.header.frame_id = "carla_gnss"

        msg.latitude = float(data.latitude)
        msg.longitude = float(data.longitude)
        msg.altitude = float(data.altitude)

        self.gnss_pub.publish(msg)

    def imu_callback(self, data):
        msg = Imu()
        msg.header.stamp = self.stamp_now()
        msg.header.frame_id = "carla_imu"

        msg.angular_velocity.x = float(data.gyroscope.x)
        msg.angular_velocity.y = float(data.gyroscope.y)
        msg.angular_velocity.z = float(data.gyroscope.z)

        msg.linear_acceleration.x = float(data.accelerometer.x)
        msg.linear_acceleration.y = float(data.accelerometer.y)
        msg.linear_acceleration.z = float(data.accelerometer.z)

        msg.orientation_covariance[0] = -1.0

        self.imu_pub.publish(msg)


    def destroy_node(self):
        self.get_logger().info("CARLA sensor bridge kapanıyor. Sensörler temizleniyor...")

        for sensor in self.sensors:
            try:
                if sensor is not None and sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            except Exception as e:
                self.get_logger().warn(f"Sensor cleanup hata: {e}")

        self.sensors.clear()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = CarlaSensorBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[carla_sensor_bridge_node] FATAL: {e}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
