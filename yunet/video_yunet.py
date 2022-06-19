import cv2 
import numpy as np


file_name = '00000000043000801'
# file_name = '00000000043001201'
input_video = '../retinaface/face_data_video/mat_vao/{}.mp4'.format(file_name)
vid = cv2.VideoCapture(input_video)

model = cv2.FaceDetectorYN.create(
        model="../api/weight/face_yunet/yunet.onnx",
        config="",
        input_size=(700, 700),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    )
# model.setInputSize(((700, 700 )))

max_det_score = 0
max_face = None

zoom_in = 1

w = 1920
h = 1080
x = 0
y = 0
if zoom_in == 1:
    x = 650
    y = 250
    w = 700
    h = 700

if zoom_in == 2:
    x = 450
    y = 0
    w = 1200
    h = 800

if zoom_in == 2:
    x = 450
    y = 0
    w = 1200
    h = 800

while vid.isOpened():
    ret, frame = vid.read()
    if ret:
        # zoom in
        if zoom_in != 0:
            frame = frame[y:y+h, x:x+w]

        _, face_detect = model.detect(frame)   
        if face_detect is not None:
            for face in face_detect:
                x_box, y_box, w_box, h_box = face[:4].astype(int)
                frame = cv2.rectangle(frame, (x_box, y_box), (x_box+w_box, y_box+h_box), (0,255,0), 2)

        # frame = cv2.resize(frame, (w//2, h//2))
        cv2.imshow('1', frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            if max_face is not None:
                cv2.imwrite('./face_data_video/max_face/{}_{}.jpg'.format(file_name, max_det_score), max_face)

            break
    else:
        break

if max_face is not None:
    cv2.imwrite('./face_data_video/max_face/{}_{}.jpg'.format(file_name, max_det_score), max_face)
print(max_det_score)
cv2.destroyAllWindows()