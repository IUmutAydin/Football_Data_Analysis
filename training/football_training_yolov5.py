from ultralytics import YOLO

model = YOLO('yolov5x.pt')


def main():
    result = model.train(
        task='detect',
        data='C:/neural_networks/FOOTBALL_ANALYSIS/training/football-players-detection-1/data.yaml',
        epochs=50,
        imgsz=1280,
        batch=16,
        device="0",
        project='runs',
        name='football_fine_tuning_results',
        save=True,
        exist_ok=True,
        resume=False
    )


if __name__ == '__main__':
    main()
