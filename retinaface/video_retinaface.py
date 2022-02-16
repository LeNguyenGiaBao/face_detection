import cv2 
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# file_name = '00000000043000801'
file_name = '00000000043001201'
input_video = './face_data_video/mat_vao/{}.mp4'.format(file_name)
vid = cv2.VideoCapture(input_video)

model_name = 'buffalo_m'
app = FaceAnalysis(name=model_name, allowed_modules=['detection']) # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))
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

        faces = app.get(frame)
        for face in faces:
            det_score = face['det_score']
            if det_score > max_det_score:
                bbox = face['bbox'].astype(int)
                x1, y1, x2, y2 = bbox
                face_img = frame[y1:y2, x1:x2]
                max_face = face_img
                max_det_score = det_score

        frame = app.draw_on(frame, faces)

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