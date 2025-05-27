import time
import math
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2, Preview
from libcamera import Transform
import picar_4wd as fc

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="best_float32.tflite") 
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MOVE_DURATION = 2.8

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
    transform=Transform(hflip=True, vflip=True)
)
picam2.configure(config)
picam2.start()
time.sleep(2)

def detect_lr_arrow_by_tip(image, debug=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))
    lower_blue, upper_blue = np.array([100, 150, 50]), np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.bitwise_or(mask_red, mask_blue)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "no arrow detected"
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:
        return "no arrow detected"
    M = cv2.moments(c)
    if M["m00"] == 0:
        return "no arrow detected"
    cx = M["m10"] / M["m00"]
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    if len(approx) < 5:
        leftmost = tuple(c[c[:, :, 0].argmin()][0])
        rightmost = tuple(c[c[:, :, 0].argmax()][0])
        return "left" if abs(leftmost[0] - cx) > abs(rightmost[0] - cx) else "right"

    def angle(p1, p, p2):
        v1 = np.array(p1) - np.array(p)
        v2 = np.array(p2) - np.array(p)
        cosθ = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return math.degrees(math.acos(np.clip(cosθ, -1, 1)))

    best_angle = 180
    tip_pt = None
    pts = [pt[0] for pt in approx]
    for i in range(len(pts)):
        a, b, c2 = pts[(i - 1) % len(pts)], pts[i], pts[(i + 1) % len(pts)]
        ang = angle(a, b, c2)
        if ang < best_angle:
            best_angle = ang
            tip_pt = tuple(b)

    if tip_pt is None:
        return "no arrow detected"
    return "left" if tip_pt[0] < cx else "right"

def get_arrow_direction(frame):
    original = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    input_shape = input_details[0]['shape']
    resized = cv2.resize(original, (input_shape[2], input_shape[1]))
    input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # shape (1, 6, 8400)
    reshaped = output_data[0].T  # shape (8400, 6)

    boxes, scores, raw = [], [], []

    for det in reshaped:
        x_c, y_c, w, h, conf, class_id = det
        if conf < 0.4:
            continue
        x1 = int((x_c - w / 2) * original.shape[1])
        y1 = int((y_c - h / 2) * original.shape[0])
        x2 = int((x_c + w / 2) * original.shape[1])
        y2 = int((y_c + h / 2) * original.shape[0])
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(conf))
        raw.append((x1, y1, x2, y2))

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.4, nms_threshold=0.5)

    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        x1, y1, x2, y2 = raw[i]
        crop = original[y1:y2, x1:x2]
        if crop.size > 0:
            direction = detect_lr_arrow_by_tip(crop)

            # Draw visual output
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original, direction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show annotated image
            cv2.imshow("Arrow Prediction", original)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

            return direction

    return "no arrow detected"


def move_and_decide(car_speed):
    print("Moving forward 0.5m...")
    fc.forward(car_speed)
    time.sleep(MOVE_DURATION)
    fc.stop()

    print("Capturing frame after 10 seconds...")
    time.sleep(3)

    frame = picam2.capture_array()
    direction = get_arrow_direction(frame)
    print("Detected direction:", direction)

    if direction == "right":
        print("Turning right")
        fc.turn_right(15)
        time.sleep(1)
    elif direction == "left":
        print("Turning left")
        fc.turn_left(15)
        time.sleep(1)
    else:
        print("No arrow detected, skipping turn.")
    fc.stop()
    time.sleep(3)

# Run
move_and_decide(28)
move_and_decide(15)
move_and_decide(24)
move_and_decide(28)
print("Final move forward")
fc.forward(30)
time.sleep(MOVE_DURATION)
fc.stop()
cv2.destroyAllWindows()