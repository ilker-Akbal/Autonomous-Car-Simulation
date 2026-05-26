import cv2
import numpy as np


class LaneDetector:

    def __init__(self):
        pass

    def region_of_interest(self, img):
        h, w = img.shape[:2]

        mask = np.zeros_like(img)

        polygon = np.array([[
            (0, h),
            (w, h),
            (int(w * 0.6), int(h * 0.6)),
            (int(w * 0.4), int(h * 0.6)),
        ]], dtype=np.int32)

        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(img, mask)

    def detect_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def detect_lines(self, edges):
        return cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=100,
        )

    def average_lines(self, frame, lines):
        left = []
        right = []

        if lines is None:
            return None, None

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.5:
                continue

            if slope < 0:
                left.append(line[0])
            else:
                right.append(line[0])

        return left, right

    def make_line(self, frame, lines):
        if len(lines) == 0:
            return None

        x_coords = []
        y_coords = []

        for x1, y1, x2, y2 in lines:
            x_coords += [x1, x2]
            y_coords += [y1, y2]

        poly = np.polyfit(y_coords, x_coords, deg=1)

        y1 = frame.shape[0]
        y2 = int(y1 * 0.6)

        x1 = int(poly[0] * y1 + poly[1])
        x2 = int(poly[0] * y2 + poly[1])

        return (x1, y1, x2, y2)

    def detect(self, frame):
        edges = self.detect_edges(frame)
        roi = self.region_of_interest(edges)
        lines = self.detect_lines(roi)

        left_lines, right_lines = self.average_lines(frame, lines)

        left_lane = self.make_line(frame, left_lines) if left_lines else None
        right_lane = self.make_line(frame, right_lines) if right_lines else None

        return left_lane, right_lane