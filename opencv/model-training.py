from ultralytics import YOLO

model = YOLO("/Users/naveen07/Documents/alumnx/project/computer-vision/runs/detect/train2/weights/best.pt")  # load a custom model

model.predict(source="vid.mp4",
              show=True,
              line_width=2,)

if __name__ == '__main__':
    model.train(
        data="data.yaml",
        batch=16,
        workers=1,
        epochs=100,
    )