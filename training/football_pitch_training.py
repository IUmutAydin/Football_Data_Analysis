from ultralytics import YOLO

model = YOLO('yolov8x-pose.pt')


def main():
    model.train(
        data='C:/neural_networks/FOOTBALL_ANALYSIS/training/football-field-detection-1/data.yaml',
        epochs=500,
        imgsz=640,
        batch=48,
        device="0",
        patience=100,
        project='runs',
        name='football_pitch_fine_tuning_results',
        save=True,
        exist_ok=True,
        resume=False
    )


if __name__ == '__main__':
    main()
