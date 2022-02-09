import cv2 
import numpy as np
import insightface
from insightface.app import FaceAnalysis

file_name = '00000000043000801'
input_video = './face_data_video/mat_vao/{}.mp4'.format(file_name)
vid = cv2.VideoCapture(input_video)
zoom_in = 0
model_name = 'buffalo_s'
app = FaceAnalysis(name=model_name, allowed_modules=['detection']) # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))
max_det_score = 0
max_face = None

while vid.isOpened():
    ret, frame = vid.read()
    if ret:
        # zoom in
        # if zoom_in != 0:
        #     new_frame = frame[]

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

        frame = cv2.resize(frame, (960, 540))
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