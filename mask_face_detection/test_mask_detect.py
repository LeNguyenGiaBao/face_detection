import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import glob 
import os
import time
import shutil

model_name = 'buffalo_m'
app = FaceAnalysis(name=model_name, allowed_modules=['detection']) # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))

data_path = glob.glob('./img/*.*')
for path in data_path:
    file_name = os.path.split(path)[1]
    img = cv2.imread(path)
    faces = app.get(img)
    
    for face in faces:
        bbox = face['bbox'].astype(int)
        x1, y1, x2, y2 = bbox
        width_face = x2 - x1
        height_face = y2 - y1 

        img_face = np.zeros((height_face*2, width_face*2,3))
        img_face[height_face//2:height_face//2 + height_face, width_face//2:width_face//2+width_face] = img[y1:y2, x1:x2]
        cv2.imwrite('test.jpg', img_face)
        exit()