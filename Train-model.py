from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8m.pt")  # You can also use yolov8s.pt, yolov8m.pt, etc.

# Train the model
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="arrow_yolo_model",
    verbose=True
)
