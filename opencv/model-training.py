from ultralytics import YOLO

model = YOLO("/Users/sumanth/learning/computer-vision/runs/detect/train2/weights/best.pt")  # load a custom model

# model= YOLO("yolo11n.pt")
model.predict(source="/Users/sumanth/Downloads/edited_DJI_20250807102015_0041_D.mp4",
              show=True,
              line_width=2,)

if __name__ == '__main__':
    model.train(
        data="data.yaml",
        batch=16,
        workers=1,
        epochs=100,
    )