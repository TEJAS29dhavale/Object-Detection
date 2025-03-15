import cv2
import numpy as np

# Load Model
config = 'ssd_mobilenet_v2_large_coco.pbtxt'
frozen = 'frozen_inference_graph (1).pb'
model = cv2.dnn_DetectionModel(frozen, config)

# Load Labels
with open('labels.txt', 'r') as f:
    classLabels = f.read().rstrip('\n').split('\n')

# Model Settings
model.setInputSize(300, 300)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))  # Corrected
model.setInputSwapRB(True)

# Open Mobile Camera
cap = cv2.VideoCapture("http://192.168.31.57:8080/video")

if not cap.isOpened():
    raise IOError("Cannot open IP Webcam stream")

font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        continue

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.3)

    # Draw bounding boxes
    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if 0 < ClassInd <= len(classLabels):
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                label = f"{classLabels[ClassInd-1]}: {conf:.2f}"
                cv2.putText(frame, label, (boxes[0], boxes[1]-10), font, 1, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
