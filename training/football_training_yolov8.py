from ultralytics import YOLO

model_1 = YOLO('yolov8n.pt')
model_2 = YOLO('yolov8n.pt')


def train_baseline():
    model_1.train(
        task='detect',
        data='C:/neural_networks/FOOTBALL_ANALYSIS/training/football-players-detection-1/data.yaml',
        epochs=15,
        imgsz=640,
        batch=16,
        device="0",
        project='runs',
        name='model_1_baseline_nano',
        save=True,
        exist_ok=True,
        optimizer='SGD'
    )


def train_comparison():
    model_2.train(
        task='detect',
        data='C:/neural_networks/FOOTBALL_ANALYSIS/training/football-players-detection-1/data.yaml',
        epochs=15,
        imgsz=640,
        batch=16,
        device="0",
        project='runs',
        name='model_2_comparison_medium',
        save=True,
        exist_ok=True,
        optimizer='AdamW',
    )


def main():
    train_baseline()
    train_comparison()


if __name__ == '__main__':
    main()
