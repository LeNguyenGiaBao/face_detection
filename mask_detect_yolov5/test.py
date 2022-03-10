import torch
import cv2 
from fastapi import FastAPI

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', './weights/best.pt')  # or yolov5m, yolov5l, yolov5x, custom

# Images
# img = '/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/face_1_in.jpg'  # or file, Path, PIL, OpenCV, numpy, list
# img = '/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/Screenshot from 2022-03-07 22-56-19.png'
# img = '/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/Screenshot from 2022-03-07 23-37-45.png'

img_path = '/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/Screenshot from 2022-03-07 23-37-45.png'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

# results.show()
print(results.pred[0])
for face in results.pred[0]:
    x1, x2, y1, y2, conf, cls = face
    print(cls==0)