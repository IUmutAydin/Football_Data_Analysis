from collections import deque
import cv2
import numpy as np
import pandas as pd
import supervision as sv


class BallAnnotator():
    def __init__(self, color):
        self.color = color

    def annotate(self, frame, detection):
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

        measurement = np.array([[np.float32(best_detection_x)],
                                [np.float32(best_detection_y)]])
        corrected = self.kf.correct(measurement)

        final_x = int(corrected[0])
        final_y = int(corrected[1])

        self.buffer.append(np.array([final_x, final_y]))

        return np.array([final_x, final_y])


# class BallTracker():
#     def __init__(self, buffer_size=5):
#         self.buffer = deque(maxlen=buffer_size)
#         self.ball_coordinates = list()

#     def update(self, detections: sv.Detections):
#         if len(detections) == 0:
#             self.ball_coordinates.append(np.array([np.nan, np.nan]))
#             return

#         xy = detections.get_anchors_coordinates(sv.Position.CENTER)
#         if self.buffer:
#             centroid = np.mean(np.array(self.buffer), axis=0)
#             distances = np.linalg.norm(xy - centroid, axis=1)

#             index = np.argmin(distances)
#         else:
#             index = np.argmax(detections.confidence)

#         self.buffer.append(xy[index])
#         self.ball_coordinates.append(xy[index])

#     def get_ball_coordinates(self):
#         df_ball_positions = pd.DataFrame(
#             self.ball_coordinates, columns=['x', 'y'])

#         df_ball_positions = df_ball_positions.interpolate()
#         df_ball_positions = df_ball_positions.bfill()

#         ball_positions = df_ball_positions.to_numpy()

#         return self.ball_coordinates
