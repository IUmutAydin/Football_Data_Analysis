from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
from sympy import use
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(sequence, batch_size):
    batch_size = max(batch_size, 1)

    current_batch = []

    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch

            current_batch = []

        current_batch.append(element)

    if current_batch:
        yield current_batch


class TeamClassifier:
    def __init__(self, device='cpu', batch_size=32):
        self.device = device

        self.batch_size = batch_size

        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)

        self.processor = AutoProcessor.from_pretrained(
            SIGLIP_MODEL_PATH, use_fast=True)

        self.reducer = umap.UMAP(n_components=3)

        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops) -> np.ndarray:
        crops = [sv.cv2_to_pillow(crop) for crop in crops]

        batches = create_batches(crops, self.batch_size)

        data = []

        with torch.no_grad():
            for batch in batches:
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(
                    outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops):
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops):
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)


def get_crops(frame, detections):
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def get_team_classifier(source_video_path, model, class_id, stride=60, device='cpu'):
    frame_generator = sv.get_video_frames_generator(
        source_video_path, stride=stride)

    crops = []

    for frame in frame_generator:
        result = model.infer(frame, imgsz=1280)[0]
        detections = sv.Detections.from_inference(result)
        crops += get_crops(frame,
                           detections[detections.class_id == class_id])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    return team_classifier


def resolve_goalkeeper_team(goalkeepers: sv.Detections, players: sv.Detections, player_team_ids):
    if len(players) == 0 or len(np.unique(player_team_ids)) < 2:
        return np.array([0] * len(goalkeepers))

    goalkeepers_xy = goalkeepers.get_anchors_coordinates(
        sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    team_0_centroid = np.mean(players_xy[player_team_ids == 0], axis=0)
    team_1_centroid = np.mean(players_xy[player_team_ids == 1], axis=0)

    goalkeeper_team_ids = []

    for goalkeeper_xy in goalkeepers_xy:
        dist1 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist2 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeeper_team_ids.append(0 if dist1 < dist2 else 1)

    return np.array(goalkeeper_team_ids)
