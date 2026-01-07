from collections import deque

import cv2
import numpy as np
import supervision as sv


class BallAnnotator():
    def __init__(self, color):
        self.color = color

    def annotate(self, frame, detection, ball_speed=None):
        if np.isnan(detection).any():
            return frame

        x, y = detection
        x = int(x)
        y = int(y)

        p1 = (x - 20, y - 20)
        p2 = (x + 20, y - 20)
        p3 = (x, y)

        triangle_points = np.array([p1, p2, p3])

        cv2.drawContours(frame, [triangle_points], 0,
                         self.color, thickness=cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), thickness=2)

        if ball_speed is not None:
            label = f"{ball_speed:.1f} km/h"

            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            rect_x1, rect_y1 = x + 10, y - 25
            rect_x2, rect_y2 = rect_x1 + text_w + 10, rect_y1 - text_h - 10

            overlay = frame.copy()
            cv2.rectangle(overlay, (rect_x1, rect_y1),
                          (rect_x2, rect_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            cv2.putText(frame, label, (rect_x1 + 5, rect_y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame


class BallTracker():
    def __init__(self, buffer_size=5, dt=1):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                             [0, 1, 0, dt],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)

        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.2
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections):
        prediction = self.kf.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])

        if len(detections) == 0:
            return np.array([pred_x, pred_y])

        xy = detections.get_anchors_coordinates(sv.Position.CENTER)

        if self.buffer:
            centroid = np.mean(np.array(self.buffer), axis=0)
            distances = np.linalg.norm(xy - centroid, axis=1)
            index = np.argmin(distances)
        else:
            index = np.argmax(detections.confidence)

        best_detection_x = xy[index][0]
        best_detection_y = xy[index][1]

        measurement = np.array(
            [[best_detection_x], [best_detection_y]], dtype=np.float32)
        corrected = self.kf.correct(measurement)

        final_x = int(corrected[0])
        final_y = int(corrected[1])

        self.buffer.append(np.array([final_x, final_y]))

        return np.array([final_x, final_y])
