import cv2
import numpy as np

config='ssd_mobilenet_v2_large_coco.pbtxt'
frozen='frozen_inference_graph (1).pb'

model=cv2.dnn_DetectionModel(frozen,config)

classLabels=[]
file_name='labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(720,720)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127,5,127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture(0)
addr="http://192.168.31.57:8080/video"
cap.open(addr)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



if not cap.isOpened():
    cap = cv2.VideoCapture(addr)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
font_scale = 3
font= cv2.FONT_HERSHEY_PLAIN
while True:
    ret, frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
    print(ClassIndex)

    if (len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if (ClassInd<=80):
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, font_scale, (255, 255, 255), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()