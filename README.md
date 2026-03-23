# Football Data Analysis

Real-time football match analysis system powered by computer vision and deep learning. Detects players, ball, and pitch; classifies teams; estimates speeds; and generates tactical visualizations.

## Features

- **Player & Ball Detection** - YOLOv8-based detection with ByteTrack
- **Team Classification** - SigLIP embeddings + UMAP + KMeans clustering
- **Speed Estimation** - Real-time player/ball speed with EMA smoothing
- **Ball Tracking** - Kalman Filter for smooth ball position estimation
- **Ball Possession** - Territory-based possession statistics
- **Tactical View** - Homography-based 2D radar visualization

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)

### Installation

```bash
git clone https://github.com/IUmutAydin/Football_Data_Analysis.git
cd Football_Data_Analysis
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:

```env
API_KEY=your_roboflow_api_key
PLAYER_BALL_DETECTION_MODEL_ID=your_model_id
PITCH_DETECTION_MODEL_ID=your_model_id
```

Get your API key and models from [Roboflow](https://roboflow.com).

### Usage

```python
python main.py
```

Or use the pipeline programmatically:

```python
from main import run_model

run_model(
    source_video_path='videos/input/match.mp4',
    target_video_path='videos/output/analyzed.mp4',
    device='cuda'
)
```

## Project Structure

```
Football_Data_Analysis/
├── main.py                 # Entry point & pipeline orchestration
├── annotators/
│   └── football.py         # Pitch drawing, ball possession overlay
├── common/
│   ├── ball.py             # BallTracker (Kalman), BallAnnotator
│   ├── data.py             # SpeedEstimator, BallTerritory
│   ├── team.py             # TeamClassifier (SigLIP + UMAP + KMeans)
│   └── view.py             # ViewTransformer (homography), radar render
├── configs/
│   ├── football.py         # Pitch dimensions & keypoints config
│   └── models.py           # Model paths & class IDs
├── training/
│   ├── football_training_yolov8.py
│   └── football_pitch_training.py
└── utils/
    └── video_utils.py      # Video collection utilities
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Video Frame                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
    ┌───────────────────┐           ┌───────────────────┐
    │  Player/Ball      │           │   Pitch Keypoint  │
    │  Detection Model  │           │   Detection Model │
    └───────────────────┘           └───────────────────┘
                │                               │
                ▼                               ▼
    ┌───────────────────┐           ┌───────────────────┐
    │  ByteTrack        │           │  Homography       │
    │  Tracking         │           │  Transform        │
    └───────────────────┘           └───────────────────┘
                │                               │
                ▼                               │
    ┌───────────────────┐                       │
    │  Team Classifier  │                       │
    │  (SigLIP+KMeans)  │                       │
    └───────────────────┘                       │
                │                               │
                ▼                               ▼
    ┌─────────────────────────────────────────────────────┐
    │                  Annotation Pipeline                 │
    │  • Speed estimation (EMA smoothed)                  │
    │  • Ball territory & possession                      │
    │  • Radar view (2D tactical visualization)           │
    └─────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │   Output Video    │
                    └───────────────────┘
```

## Key Components

### Team Classification

Uses SigLIP vision transformer for player crop embeddings, reduces dimensions with UMAP, and clusters into two teams with KMeans. Goalkeepers are assigned based on spatial proximity to team centroids.

```python
# common/team.py
team_classifier = TeamClassifier(device='cuda')
team_classifier.fit(player_crops)
team_ids = team_classifier.predict(new_crops)
```

### Speed Estimation

Calculates speed using frame-to-frame displacement with Exponential Moving Average smoothing. Converts pixel distance to real-world units using homography-transformed coordinates.

```python
# common/data.py
speed_estimator = SpeedEstimator(fps=30, alpha=0.1)
speed, distance = speed_estimator.update(player_id, pitch_position)
```

### Ball Tracking

Implements a Kalman Filter with a detection buffer for robust ball tracking, handling occlusions and detection failures.

```python
# common/ball.py
ball_tracker = BallTracker(buffer_size=5)
position = ball_tracker.update(ball_detections)
```

### View Transformation

Computes homography matrix from detected pitch keypoints to standard pitch coordinates, enabling real-world position estimation.

```python
# common/view.py
transformer = ViewTransformer(source=keypoints, target=pitch_vertices)
pitch_coords = transformer.transform_points(pixel_coords)
```

## Models

| Model | Purpose | Source |
|-------|---------|--------|
| Player/Ball Detection | Detect players, goalkeepers, referees, ball | Roboflow |
| Pitch Keypoint Detection | Detect 32 pitch keypoints | Roboflow |
| SigLIP | Player embedding extraction | HuggingFace |

## Training Custom Models

```python
from training.football_training_yolov8 import train_baseline

train_baseline()
```

Prepare your dataset in YOLO format and update the `data.yaml` path.

## Dependencies

```
opencv-python
numpy
supervision
inference
ultralytics
torch
transformers
umap-learn
scikit-learn
python-dotenv
tqdm
```

## Acknowledgments

- [Roboflow](https://roboflow.com) for hosted inference models
- [Supervision](https://github.com/roboflow/supervision) for annotation utilities
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8

## License

MIT License
