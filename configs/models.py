import os


PARENT_DIR = 'training/fine_tuned_models'

PLAYER_BALL_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, 'player_detection_model.pt')

PITCH_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, 'pitch_detection_model.pt')

PLAYER_BALL_DETECTION_MODEL_ID = 'football-players-detection-3zvbc/20'
PITCH_DETECTION_MODEL_ID = 'football-field-detection-f07vi/15'
API_KEY = 'UquXLB1d8JCnXwQVTnKN'

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3
