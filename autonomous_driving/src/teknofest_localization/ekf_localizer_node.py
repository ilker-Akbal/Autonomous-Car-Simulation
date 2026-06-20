import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import String


def _quat_to_yaw(q: Quaternion) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _gnss_to_xy(lat: float, lon: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    R = 6_371_000.0
    dlat = math.radians(lat - origin_lat)
    dlon = math.radians(lon - origin_lon)
    x = R * dlon * math.cos(math.radians(origin_lat))
    y = R * dlat
    return x, y


def _has_valid_orientation(msg: Imu) -> bool:
    orientation = msg.orientation
    if orientation is None:
        return False
    if (orientation.x == 0.0 and orientation.y == 0.0 and orientation.z == 0.0 and orientation.w == 0.0):
        return False
    if len(msg.orientation_covariance) > 0 and msg.orientation_covariance[0] < 0:
        return False
    return True


class EKFLocalizer(Node):
    DT = 0.05

    def __init__(self):
        super().__init__("ekf_localizer_node")

        self._x = np.zeros(5)
        self._P = np.eye(5) * 1.0
        q_diag = [0.1, 0.1, 0.05, 0.5, 0.1]
        self._Q = np.diag(q_diag)
        self._R_gnss = np.diag([1.0, 1.0])
        self._R_imu = np.diag([0.02, 0.05])
        self._origin = None
        self._imu_orientation_available = False

        self._odom_pub = self.create_publisher(Odometry, "/adas/localization/odom", 10)
        self.create_subscription(NavSatFix, "/adas/localization/gnss", self._gnss_callback, 10)
        self.create_subscription(Imu, "/adas/localization/imu", self._imu_callback, 10)
        self.create_subscription(String, "/adas/carla/status", self._status_callback, 10)
        self.create_timer(self.DT, self._predict)

        self.get_logger().info("Teknofest EKF localizer node active")

    def _predict(self) -> None:
        v = self._x[3]
        omega = self._x[4]
        yaw = self._x[2]
        dt = self.DT

        self._x[0] += v * math.cos(yaw) * dt
        self._x[1] += v * math.sin(yaw) * dt
        self._x[2] += omega * dt

        F = np.eye(5)
        F[0, 2] = -v * math.sin(yaw) * dt
        F[0, 3] = math.cos(yaw) * dt
        F[1, 2] = v * math.cos(yaw) * dt
        F[1, 3] = math.sin(yaw) * dt
        F[2, 4] = dt

        self._P = F @ self._P @ F.T + self._Q
        self._publish()

    def _gnss_callback(self, msg: NavSatFix) -> None:
        if self._origin is None:
            self._origin = (msg.latitude, msg.longitude)
            self._x[0], self._x[1] = 0.0, 0.0
            self.get_logger().info(f"GNSS origin set: {self._origin}")
            return

        z_x, z_y = _gnss_to_xy(msg.latitude, msg.longitude, *self._origin)
        z = np.array([z_x, z_y])
        H = np.zeros((2, 5))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        self._update(z, H, self._R_gnss)

    def _imu_callback(self, msg: Imu) -> None:
        if not _has_valid_orientation(msg):
            self._imu_orientation_available = False
            return

        self._imu_orientation_available = True
        yaw = _quat_to_yaw(msg.orientation)
        omega = float(msg.angular_velocity.z)
        z = np.array([yaw, omega])
        H = np.zeros((2, 5))
        H[0, 2] = 1.0
        H[1, 4] = 1.0
        self._update(z, H, self._R_imu)

    def _status_callback(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Failed to parse CARLA status JSON: {exc}")
            return

        speed = status.get("speed_mps")
        if speed is not None:
            try:
                self._x[3] = float(speed)
            except (TypeError, ValueError):
                pass

        if not self._imu_orientation_available:
            rotation = status.get("rotation", {})
            yaw_deg = rotation.get("yaw")
            if yaw_deg is not None:
                try:
                    self._x[2] = math.radians(float(yaw_deg))
                except (TypeError, ValueError):
                    pass

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        y = z - H @ self._x
        if H.shape == (2, 5) and len(z) >= 1:
            y[0] = _normalize_angle(y[0])

        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(5) - K @ H) @ self._P

    def _publish(self) -> None:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(self._x[0])
        msg.pose.pose.position.y = float(self._x[1])
        msg.pose.pose.position.z = 0.0

        yaw = float(self._x[2])
        msg.pose.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0),
        )

        msg.twist.twist.linear.x = float(self._x[3])
        msg.twist.twist.angular.z = float(self._x[4])
        self._odom_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFLocalizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
