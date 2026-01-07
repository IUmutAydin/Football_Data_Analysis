import numpy as np


class SpeedEstimator():
    def __init__(self, fps, alpha=0.1):
        self.fps = fps
        self.alpha = alpha
        self.tracker_data = {}
        self.ball_data = {'last_pos': None,
                          'ema_speed': 0}

    def update(self, player_id, position):
        if position is None or np.isnan(position).any():
            return 0, 0

        if player_id not in self.tracker_data:
            self.tracker_data[player_id] = {
                'last_pos': position,
                'distance': 0,
                'ema_speed': 0
            }
            return 0, 0

        last_pos = self.tracker_data[player_id]['last_pos']

        distance = np.linalg.norm(position - last_pos) / 100

        if distance < 0.05:
            distance = 0

        if distance < 0.3:
            current_speed = (distance * self.fps) * 3.6
            prev_ema = self.tracker_data[player_id]['ema_speed']
            self.tracker_data[player_id]['ema_speed'] = (
                self.alpha * current_speed) + (1 - self.alpha) * prev_ema

            self.tracker_data[player_id]['distance'] += distance

        self.tracker_data[player_id]['last_pos'] = position

        return self.tracker_data[player_id]['ema_speed'], self.tracker_data[player_id]['distance']

    def update_ball(self, position):
        if position is None or np.isnan(position).any():
            return 0

        if self.ball_data['last_pos'] is None:
            self.ball_data['last_pos'] = position
            return 0

        distance = np.linalg.norm(position - self.ball_data['last_pos']) / 100

        if distance < 2.0:
            self.ball_data['last_pos'] = position
            current_speed = (distance * self.fps) * 3.6
            prev_ema = self.ball_data['ema_speed']
            self.ball_data['ema_speed'] = (
                self.alpha * current_speed) + (1 - self.alpha) * prev_ema

        self.ball_data['last_pos'] = position

        return self.ball_data['ema_speed']


class BallTerritory():
    def __init__(self):
        self.max_player_ball_distance = 70
        self.team_ball_control = []
        self.player_ids = []

    def assign_player_to_ball(self, player_ids, player_team_ids, player_pitch_xy, ball_pitch_xy):
        minimum_distance = 9999999
        assigned_player = -1
        assigned_team = -1

        for player_id, player_team_id, player_pitch_xy in zip(player_ids, player_team_ids, player_pitch_xy):
            distance = np.linalg.norm(player_pitch_xy - ball_pitch_xy)
            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id
                    assigned_team = player_team_id

        if assigned_team != -1:
            self.team_ball_control.append(assigned_team)
        elif self.team_ball_control:
            self.team_ball_control.append(self.team_ball_control[-1])

        return assigned_player
