import torch
import cv2 
from torch2trt import torch2trt, TRTModule

# # create some regular pytorch model...
# model = torch.hub.load('ultralytics/yolov5', 'custom', './weights/best.pt')  # or yolov5m, yolov5l, yolov5x, custom

# # create example data
# x = torch.ones((1, 3, 640, 640)).cuda()

# # convert to TensorRT feeding sample data as input
# model_trt = torch2trt(model, [x])

model_trt = TRTModule()
model_trt.load_state_dict(torch.load('./weights/best.pt'))

img_path = '/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/Screenshot from 2022-03-07 23-37-45.png'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pred = model_trt(img).pred[0]
print(pred)
