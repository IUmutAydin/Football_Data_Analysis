from collections import deque

import numpy as np


class SpeedEstimator():
    def __init__(self, fps, window_size=8):
        self.fps = fps
        print(f'FPS: {self.fps}')
        self.window_size = window_size
        self.tracker_data = {}
        self.ball_data = {'last_pos': None,
                          'speeds': deque(maxlen=window_size)}

    def update(self, player_id, position):
        if position is None or np.isnan(position).any():
            return 0, 0

        if player_id not in self.tracker_data:
            self.tracker_data[player_id] = {
                'last_pos': position,
                'distance': 0,
                'speeds': deque(maxlen=self.window_size)
            }
            return 0, 0

        last_pos = self.tracker_data[player_id]['last_pos']

        distance = np.linalg.norm(position - last_pos) / 100

        if distance < 0.05:
            distance = 0

        if distance < 0.3:
            current_speed = (distance * self.fps) * 3.6
            self.tracker_data[player_id]['speeds'].append(current_speed)

        speed = sum(self.tracker_data[player_id]['speeds']) / \
            len(self.tracker_data[player_id]['speeds']) if len(
                self.tracker_data[player_id]['speeds']) > 0 else 0

        self.tracker_data[player_id]['last_pos'] = position
        self.tracker_data[player_id]['distance'] += distance

        return speed, self.tracker_data[player_id]['distance']

    def update_ball(self, position):
        if position is None or np.isnan(position).any() or len(position) == 0:
            return 0

        if self.ball_data['last_pos'] is None:
            self.ball_data['last_pos'] = position
            return 0

        distance = np.linalg.norm(position - self.ball_data['last_pos'])

        if distance > 10.0:
            self.ball_data['last_pos'] = position
            return 0

        current_speed = (distance * self.fps) * 3.6
        self.ball_data['speeds'].append(current_speed)

        speed = sum(self.ball_data['speeds']) / len(self.ball_data['speeds'])
        self.ball_data['last_pos'] = position

        return speed
