# Arrow Direction Detection and Navigation with PiCar-4WD + TFLite

This project enables a Raspberry Pi-powered PiCar-4WD robot to detect **left** and **right** arrows using a trained YOLOv8 model (converted to TensorFlow Lite) and autonomously navigate in that direction using a PiCamera2.

## Features

- Uses a **YOLOv8-trained model (TFLite)** for arrow detection
- Distinguishes between **left and right** arrows using **contour analysis**
- Automatically **moves forward**, **detects the arrow**, then **turns accordingly**
- Supports both **red** and **blue** colored arrows
- Displays **bounding boxes and labels** on the detected arrows

---

## Hardware Requirements

- Raspberry Pi 4 with Raspberry Pi OS
- PiCamera2
- Waveshare or SunFounder PiCar-4WD robot car
- Trained YOLOv8 model (converted to `.tflite`)

---

## Software & Libraries

Install dependencies via:

```bash
sudo apt update
sudo apt install python3-opencv python3-picamera2
pip install tflite-runtime numpy
```

## Dataset & Trained Model

You can download the training dataset and the pre-trained YOLOv8 model in TensorFlow Lite format from the link below:

[ Google Drive: Dataset + TFLite Model](https://drive.google.com/file/d/15_Ndu-UxpKIqCkTnud6SGMLe7qawkzmw/view?usp=sharing)

### Contents of the archive:
- `images/` → Arrow direction images used for training (left/right)
- `labels/` → YOLO-format annotation files
- `data.yaml` → YOLOv8 dataset configuration file
- `best_float32.tflite` → Trained YOLOv8 model converted to TensorFlow Lite format

