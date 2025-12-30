import cv2
import numpy as np
from annotators.football import draw_pitch, draw_points_on_pitch
from configs.football import SoccerPitchConfiguration
import supervision as sv


CONFIG = SoccerPitchConfiguration()


class ViewTransformer():
    def __init__(self, source, target):
        if source.shape != target.shape:
            raise ValueError('Source and target must have the same shape')

        if source.shape[1] != 2:
            raise ValueError('Source and target points must be 2D coordinates')

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        self.m, _ = cv2.findHomography(
            source,
            target
        )

        if self.m is None:
            raise ValueError('Could not find homography')

    def transform_points(self, points):
        if points.size == 0:
            return points

        points = points.reshape(-1, 1, 2).astype(np.float32)

        transformed_points = cv2.perspectiveTransform(points, self.m)

        return transformed_points.reshape(-1, 2)

    def transform_image(self, image, resolution_wh):
        if len(image.shape) not in {2, 3}:
            raise ValueError('Image must be grayscale or color')

        return cv2.warpPerspective(image, self.m, resolution_wh)


def render_radar(transformed_xy, transformed_ball_position, color_lookups, colors):
    radar = draw_pitch(CONFIG)

    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookups == 0], face_color=sv.Color.from_hex(colors[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookups == 1], face_color=sv.Color.from_hex(colors[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookups == 3], face_color=sv.Color.from_hex(colors[3]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_ball_position, face_color=sv.Color.from_hex(colors[2]), radius=20, pitch=radar)

    return radar
