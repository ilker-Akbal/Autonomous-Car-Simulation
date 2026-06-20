import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


def _canny(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blur)
    lo = int(max(0, 0.67 * v))
    hi = int(min(255, 1.33 * v))
    return cv2.Canny(blur, lo, hi)


class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__("lane_detector_node")
        self.bridge = CvBridge()

        self.declare_parameter("roi_bottom_left", 0.05)
        self.declare_parameter("roi_top_left", 0.40)
        self.declare_parameter("roi_top_right", 0.60)
        self.declare_parameter("roi_bottom_right", 0.95)
        self.declare_parameter("roi_top_y", 0.55)

        self.roi_bottom_left = float(self.get_parameter("roi_bottom_left").value)
        self.roi_top_left = float(self.get_parameter("roi_top_left").value)
        self.roi_top_right = float(self.get_parameter("roi_top_right").value)
        self.roi_bottom_right = float(self.get_parameter("roi_bottom_right").value)
        self.roi_top_y = float(self.get_parameter("roi_top_y").value)

        self.subscription = self.create_subscription(
            Image,
            "/adas/camera/front/image_raw",
            self._image_callback,
            10,
        )
        self.img_pub = self.create_publisher(Image, "/adas/perception/lane_viz", 10)
        self.cte_pub = self.create_publisher(Float32, "/adas/perception/lane_cte", 10)

        self.get_logger().info("Teknofest lane detector node active")

    def _to_bgr(self, msg: Image) -> np.ndarray | None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().error(f"CV bridge conversion failed: {exc}")
            return None

        encoding = msg.encoding.lower()
        if "rgb" in encoding and "bgr" not in encoding:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except cv2.error:
                pass
        elif encoding == "rgba8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif encoding == "bgra8":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif encoding != "bgr8":
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except CvBridgeError as exc:
                self.get_logger().error(f"Unsupported image encoding {msg.encoding}: {exc}")
                return None

        return frame

    def _apply_roi(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros_like(image)
        polygon = np.array(
            [[
                (int(w * self.roi_bottom_left), h),
                (int(w * self.roi_top_left), int(h * self.roi_top_y)),
                (int(w * self.roi_top_right), int(h * self.roi_top_y)),
                (int(w * self.roi_bottom_right), h),
            ]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(image, mask)

    def _fit_line(self, lines):
        left = []
        right = []
        if lines is None:
            return None, None

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if slope < 0:
                left.append((slope, intercept))
            else:
                right.append((slope, intercept))

        def average(group):
            if not group:
                return None
            return tuple(np.mean(group, axis=0))

        return average(left), average(right)

    def _line_coords(self, image: np.ndarray, params):
        if params is None:
            return None

        slope, intercept = params
        h = image.shape[0]
        y1 = h
        y2 = int(h * self.roi_top_y)
        if abs(slope) < 1e-6:
            return None

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return x1, y1, x2, y2

    def _image_callback(self, msg: Image) -> None:
        frame = self._to_bgr(msg)
        if frame is None:
            return

        edges = _canny(frame)
        roi_edges = self._apply_roi(edges)
        lines = cv2.HoughLinesP(
            roi_edges,
            2,
            np.pi / 180,
            100,
            minLineLength=40,
            maxLineGap=5,
        )

        left_params, right_params = self._fit_line(lines)
        left_line = self._line_coords(frame, left_params)
        right_line = self._line_coords(frame, right_params)

        viz = frame.copy()
        lane_center_x = frame.shape[1] // 2
        if left_line is not None:
            cv2.line(viz, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 6)
        if right_line is not None:
            cv2.line(viz, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 6)

        if left_line is not None and right_line is not None:
            lane_center_x = (left_line[0] + right_line[0]) // 2
        elif left_line is not None:
            lane_center_x = left_line[0] + 150
        elif right_line is not None:
            lane_center_x = right_line[0] - 150

        img_center_x = frame.shape[1] // 2
        cte = float(img_center_x - lane_center_x) / (frame.shape[1] / 2)

        cte_msg = Float32()
        cte_msg.data = cte
        self.cte_pub.publish(cte_msg)

        try:
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(viz, "bgr8"))
        except CvBridgeError as exc:
            self.get_logger().error(f"Lane visualization publish failed: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
