import os

from dotenv import load_dotenv


PARENT_DIR = 'training/fine_tuned_models'

PLAYER_BALL_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, 'player_detection_model.pt')

PITCH_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, 'pitch_detection_model.pt')

load_dotenv()
PLAYER_BALL_DETECTION_MODEL_ID = os.getenv('PLAYER_BALL_DETECTION_MODEL_ID')
PITCH_DETECTION_MODEL_ID = os.getenv('PITCH_DETECTION_MODEL_ID')
API_KEY = os.getenv('API_KEY')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3
