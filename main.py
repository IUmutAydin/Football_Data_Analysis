import os
import pickle
import cv2
from inference import get_model
import numpy as np
import tqdm
from ultralytics import YOLO
import supervision as sv
from annotators.football import draw_pitch, draw_points_on_pitch
from common.ball import BallAnnotator, BallTracker
from common.data import SpeedEstimator
from common.team import get_crops, get_team_classifier, resolve_goalkeeper_team
from common.view import CONFIG, ViewTransformer, render_radar
from configs.football import SoccerPitchConfiguration
from configs.models import API_KEY, BALL_CLASS_ID, GOALKEEPER_CLASS_ID, PITCH_DETECTION_MODEL_ID, PLAYER_BALL_DETECTION_MODEL_ID, PLAYER_BALL_DETECTION_MODEL_PATH, PLAYER_CLASS_ID, REFEREE_CLASS_ID
from utils.video_utils import collect_videos


STRIDE = 60

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)

ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

SPEED_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.Color.from_hex('#000000'),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_position=sv.Position.CENTER_RIGHT,
    text_padding=3,
    text_scale=0.4,
    border_radius=5
)


class FootballAnalysisPip():
    def __init__(self, source_video_path, device):
        self.source_video_path = source_video_path
        self.device = device
        self.frame_generator = sv.get_video_frames_generator(
            source_video_path)
        self.player_ball_detection_model = get_model(
            model_id=PLAYER_BALL_DETECTION_MODEL_ID, api_key=API_KEY)
        self.pitch_detection_model = get_model(
            model_id=PITCH_DETECTION_MODEL_ID, api_key=API_KEY)
        self.tracker = sv.ByteTrack()
        self.team_classifier = team_classifier = get_team_classifier(
            source_video_path, self.player_ball_detection_model, PLAYER_CLASS_ID, STRIDE, device)
        self.ball_tracker = BallTracker(buffer_size=5)
        self.ball_annotator = BallAnnotator(
            sv.Color.from_hex(COLORS[2]).as_bgr())
        self.speed_estimator = SpeedEstimator(
            sv.VideoInfo.from_video_path(source_video_path).fps, 8)

    def process_frame(self, frame):
        # Model results
        result = self.pitch_detection_model.infer(frame)[0]
        keypoints = sv.KeyPoints.from_inference(result)

        result = self.player_ball_detection_model.infer(frame, imgsz=1280)[0]
        detections = sv.Detections.from_inference(result)

        # Creating object groups
        track_mask = np.isin(detections.class_id, [
                             PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID])
        all_players = self.tracker.update_with_detections(
            detections[track_mask])

        # Team classification
        players = all_players[all_players.class_id == PLAYER_CLASS_ID]
        player_team_ids = self.team_classifier.predict(
            get_crops(frame, players))

        goalkeepers = all_players[all_players.class_id == GOALKEEPER_CLASS_ID]
        goalkeeper_team_ids = resolve_goalkeeper_team(
            goalkeepers, players, player_team_ids)

        players = sv.Detections.merge([players, goalkeepers])
        player_team_ids = np.concatenate(
            [player_team_ids, goalkeeper_team_ids]).astype(int)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]
        referees.tracker_id = np.array([None] * len(referees))

        # Ball position estimation
        ball = detections[detections.class_id == BALL_CLASS_ID]
        ball_position = self.ball_tracker.update(ball)

        # Homography transformation
        coord_mask = (keypoints.xy[0][:, 0] > 0) & (
            keypoints.xy[0][:, 1] > 0)
        conf_mask = keypoints.confidence[0] > 0.5
        final_mask = coord_mask & conf_mask

        source_points = keypoints.xy[0][final_mask]

        target_points = np.array(CONFIG.vertices)[final_mask]

        view_transformer = ViewTransformer(
            source=source_points.astype(np.float32),
            target=target_points.astype(np.float32)
        )

        player_pixel_xy = players.get_anchors_coordinates(
            sv.Position.BOTTOM_CENTER)
        player_pitch_xy = view_transformer.transform_points(player_pixel_xy)

        referee_pixel_xy = referees.get_anchors_coordinates(
            sv.Position.BOTTOM_CENTER)
        referee_pitch_xy = view_transformer.transform_points(
            referee_pixel_xy)

        ball_pitch_xy = view_transformer.transform_points(
            ball_position)

        return {
            'players': players,
            'referees': referees,
            'player_team_ids': player_team_ids,
            'ball_position': ball_position,
            'player_pitch_xy': player_pitch_xy,
            'referee_pitch_xy': referee_pitch_xy,
            'ball_pitch_xy': ball_pitch_xy,
            'player_pixel_xy': player_pixel_xy,
            'referee_pixel_xy': referee_pixel_xy,
            'ball_pixel_xy': ball_position,
            'keypoints': keypoints
        }

    def estimate_speed(self, data):
        player_ids = data['players'].tracker_id
        player_pitch_xys = data['player_pitch_xy']
        ball_pitch_xy = data['ball_pitch_xy']

        player_speeds = []
        speed_labels = []

        for player_id, pitch_xy in zip(player_ids, player_pitch_xys):
            # Estimate player speed
            speed, distance = self.speed_estimator.update(player_id, pitch_xy)
            player_speeds.append(speed)
            speed_labels.append(f"{speed:.1f} km/h\n{distance:.1f} m")

        data['player_speeds'] = np.array(player_speeds)

        # Estimate ball speed
        ball_speed = self.speed_estimator.update_ball(ball_pitch_xy)
        data['ball_speed'] = ball_speed

        return speed_labels

    def stream_analysis(self):
        for frame_num, frame in tqdm.tqdm(enumerate(self.frame_generator), desc='Processing frames'):
            # Player and ball detection, team classification, video annotation
            data = self.process_frame(frame)

            detections = sv.Detections.merge(
                [data['players'], data['referees']])

            color_lookup = np.array(
                data['player_team_ids'].tolist() +
                [REFEREE_CLASS_ID] * len(data['referees']))

            labels = [str(
                tracker_id) if tracker_id else 'Referee' for tracker_id in detections.tracker_id]

            annotated_frame = frame.copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, detections, custom_color_lookup=color_lookup)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated_frame, detections, labels, custom_color_lookup=color_lookup)
            annotated_frame = self.ball_annotator.annotate(
                annotated_frame, data['ball_position'])

            # Speed estimation and annotation
            speed_labels = self.estimate_speed(data)

            annotated_frame = SPEED_LABEL_ANNOTATOR.annotate(
                annotated_frame, data['players'], speed_labels)

            # Drawing radar with transformed points
            h, w, _ = frame.shape

            transformed_xy = np.concatenate(
                [data['player_pitch_xy'], data['referee_pitch_xy']])

            transformed_ball_position = data['ball_pitch_xy']

            radar = render_radar(
                transformed_xy, transformed_ball_position, color_lookup, COLORS)
            radar = cv2.resize(radar, (w // 2, h // 2))
            radar_h, radar_w, _ = radar.shape
            rect = sv.Rect(0, 0, radar_w, radar_h)
            annotated_frame = sv.draw_image(
                scene=annotated_frame, image=radar, opacity=0.5, rect=rect)

            yield annotated_frame


def run_model(source_video_path, target_video_path, device='cpu'):
    football_analysis_pip = FootballAnalysisPip(source_video_path, device)
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in football_analysis_pip.stream_analysis():
            sink.write_frame(frame)


def process_videos(video_names):
    OUTPUT_VIDEOS_DIR = 'videos/output_videos'

    videos = collect_videos()
    for video_name in video_names:
        source_video_path = videos[video_name]
        output_video_path = os.path.join(
            OUTPUT_VIDEOS_DIR, f'{video_name}.mp4')
        run_model(source_video_path, output_video_path, 'cuda')


def main() -> None:
    process_videos(['08fd33_4'])


if __name__ == '__main__':
    main()


# def detect(source_video_path, device='cpu', read_from_stub=True, stub_path='stubs/detects.pckl', video=None):
#     frame_data_dict = {}

#     if stub_path is not None and os.path.exists(stub_path):
#         with open(stub_path, 'rb') as f:
#             if os.path.getsize(stub_path) > 0:
#                 frame_data_dict = pickle.load(f)
#         if read_from_stub and video is not None and video in frame_data_dict.keys():
#             data_dict = frame_data_dict[video]
#             return data_dict['frame_data'], data_dict['ball_positions']

#     player_ball_detection_model = get_model(
#         model_id=PLAYER_BALL_DETECTION_MODEL_ID, api_key=API_KEY)
#     pitch_detection_model = get_model(
#         model_id=PITCH_DETECTION_MODEL_ID, api_key=API_KEY)

#     frame_generator = sv.get_video_frames_generator(source_video_path)

#     tracker = sv.ByteTrack()
#     team_classifier = get_team_classifier(
#         source_video_path, player_ball_detection_model, PLAYER_CLASS_ID, STRIDE, device)
#     ball_tracker = BallTracker(buffer_size=5)

#     frame_data = []

#     for frame_num, frame in tqdm.tqdm(enumerate(frame_generator), desc='Extracting datas'):
#         results = pitch_detection_model.infer(frame)[0]
#         keypoints = sv.KeyPoints.from_inference(results)

#         results = player_ball_detection_model.infer(frame, imgsz=1280)[0]
#         detections = sv.Detections.from_inference(results)

#         track_mask = np.isin(detections.class_id, [
#                              PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID])
#         all_players = tracker.update_with_detections(detections[track_mask])

#         players = all_players[all_players.class_id == PLAYER_CLASS_ID]
#         player_team_ids = team_classifier.predict(get_crops(frame, players))

#         goalkeepers = all_players[all_players.class_id == GOALKEEPER_CLASS_ID]
#         goalkeeper_team_ids = resolve_goalkeeper_team(
#             goalkeepers, players, player_team_ids)

#         referees = detections[detections.class_id == REFEREE_CLASS_ID]
#         referees.tracker_id = np.array([None] * len(referees))

#         ball = detections[detections.class_id == BALL_CLASS_ID]
#         ball_tracker.update(ball)

#         frame_data.append({
#             'players': players,
#             'goalkeepers': goalkeepers,
#             'referees': referees,
#             'player_team_ids': player_team_ids,
#             'goalkeeper_team_ids': goalkeeper_team_ids,
#             'keypoints': keypoints
#         })

#     ball_positions = ball_tracker.get_ball_coordinates()

#     if stub_path is not None and video is not None:
#         frame_data_dict[video] = {
#             'frame_data': frame_data,
#             'ball_positions': ball_positions
#         }
#         with open(stub_path, 'wb') as f:
#             pickle.dump(frame_data_dict, f)

#     return frame_data, ball_positions


# def annotate(source_video_path, frame_data, ball_positions):
#     frame_generator = sv.get_video_frames_generator(source_video_path)

#     ball_annotator = BallAnnotator(sv.Color.from_hex(COLORS[2]).as_bgr())

#     for frame_num, frame in tqdm.tqdm(enumerate(frame_generator), desc='Annotating frames'):
#         data = frame_data[frame_num]

#         detections = sv.Detections.merge(
#             [data['players'], data['goalkeepers'], data['referees']])

#         color_lookup = np.array(
#             data['player_team_ids'].tolist() +
#             data['goalkeeper_team_ids'].tolist() +
#             [REFEREE_CLASS_ID] * len(data['referees']))

#         labels = [str(
#             tracker_id) if tracker_id else 'Referee' for tracker_id in detections.tracker_id]

#         annotated_frame = frame.copy()
#         annotated_frame = ELLIPSE_ANNOTATOR.annotate(
#             annotated_frame, detections, custom_color_lookup=color_lookup)
#         annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
#             annotated_frame, detections, labels, custom_color_lookup=color_lookup)
#         annotated_frame = ball_annotator.annotate(
#             annotated_frame, ball_positions[frame_num])

#         h, w, _ = frame.shape

#         radar = render_radar(
#             data, ball_positions[frame_num], color_lookup, COLORS)
#         radar = cv2.resize(radar, (w // 2, h // 2))

#         radar_h, radar_w, _ = radar.shape

#         rect = sv.Rect(0, 0, radar_w, radar_h)

#         annotated_frame = sv.draw_image(
#             scene=annotated_frame, image=radar, opacity=0.5, rect=rect)

#         yield annotated_frame


# def run_model(source_video_path, target_video_path, device='cpu', read_from_stub=True, video=None):
#     frame_data, ball_positions = detect(
#         source_video_path, device, read_from_stub, video=video)

#     frame_generator = annotate(source_video_path, frame_data, ball_positions)

#     video_info = sv.VideoInfo.from_video_path(source_video_path)
#     with sv.VideoSink(target_video_path, video_info) as sink:
#         for frame in frame_generator:
#             sink.write_frame(frame)


# def process_videos(video_names):
#     OUTPUT_VIDEOS_DIR = 'videos/output_videos'

#     videos = collect_videos()
#     for video_name in video_names:
#         source_video_path = videos[video_name]
#         output_video_path = os.path.join(
#             OUTPUT_VIDEOS_DIR, f'{video_name}.mp4')
#         run_model(source_video_path, output_video_path,
#                   'cuda', False, video_name)


# def main() -> None:
#     process_videos(['08fd33_4'])


# if __name__ == '__main__':
#     main()


# def run_pitch_detection(source_video_path: str, device: str):
#     VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
#         color=[sv.Color.from_hex(color) for color in CONFIG.colors],
#         text_color=sv.Color.from_hex('#FFFFFF'),
#         border_radius=5,
#         text_thickness=1,
#         text_scale=0.5,
#         text_padding=5,
#     )

#     pitch_detection_model = get_model(
#         model_id=PITCH_DETECTION_MODEL_ID, api_key=API_KEY)
#     frame_generator = sv.get_video_frames_generator(
#         source_path=source_video_path)

#     for frame in frame_generator:
#         result = pitch_detection_model.infer(frame)[0]
#         keypoints = sv.KeyPoints.from_inference(result)

#         confidences = keypoints.confidence[0] if keypoints.confidence is not None else [
#         ]

#         custom_labels = []
#         for i, name in enumerate(CONFIG.labels):
#             if i < len(confidences):
#                 conf = confidences[i]
#                 custom_labels.append(f"{name} {conf:.2f}")
#             else:
#                 custom_labels.append(name)

#         annotated_frame = frame.copy()
#         annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
#             annotated_frame, keypoints, custom_labels)
#         yield annotated_frame
