import torch
import cv2 
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', './weights/best.pt')  # or yolov5m, yolov5l, yolov5x, custom

video_path = './test_data/video/00000000057004801.mp4'
video = cv2.VideoCapture(video_path)

zoom_in = 1
if zoom_in == 1:
    x = 650
    y = 250
    w = 700
    h = 700
line_width = 2
names = ['NoMask', 'Mask']
index = 0


while video.isOpened():
    ret, frame = video.read()
    index += 1
    if index % 2 != 0:
        continue 

    if ret:
        # zoom in 
        if zoom_in != 0:
            frame = frame[y:y+h, x:x+w]

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = model(frame).pred[0]
        if pred.shape[0] != 0:      # has face
            for face in pred:
                x1, y1, x2, y2, conf, cls = face

                if cls == 0: # no mask
                    color = (0,0,255)
                else:        # mask
                    color = (0, 255, 0)

                c = int(cls) 
                label = f'{names[c]} {conf:.2f}'
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)
                tf = max(line_width- 1, 1)  # font thickness
                # w, h = cv2.getTextSize(label, 0, fontScale=line_width / 3, thickness=tf)[0]  # text width, height
                # outside = p1[1] - h - 3 >= 0  # label fits outside box
                # p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                # cv2.rectangle(frame, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(frame, label, (p1[0], p1[1] - 2), 0, line_width/3, color,
                            thickness=line_width, lineType=cv2.LINE_AA)

        cv2.imshow('1', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
                
    else:
        break
cv2.destroyAllWindows()