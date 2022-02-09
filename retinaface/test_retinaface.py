import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import glob 
import os
import time
import shutil

model_name = 'buffalo_s'
output_path = os.path.join('./face_detection_output', model_name)
if os.path.exists(os.path.join('./face_detection_output', model_name)):
    shutil.rmtree(output_path)

os.mkdir(output_path)
# Method-1, use FaceAnalysis
app = FaceAnalysis(name=model_name, allowed_modules=['detection']) # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))

t = time.time()
data_path = glob.glob('./face_data/*.*')
for path in data_path:
    file_name = os.path.split(path)[1]
    img = cv2.imread(path)
    faces = app.get(img)
    rimg = app.draw_on(img, faces)

    new_path = os.path.join(output_path, file_name)
    cv2.imwrite(new_path, rimg)

total_time = time.time() - t
print('Inference time: {:.3f} s/{}'.format(total_time, len(data_path)))
print('Average time: {:.3f} s'.format(total_time/len(data_path)))

# buffalo_m
# Inference time: 4.824 s/36
# Average time: 0.134 s

# buffalo_sc
# Inference time: 4.172 s/36
# Average time: 0.116 s