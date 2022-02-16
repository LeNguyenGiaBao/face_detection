import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import glob 
import os
import time
import shutil

model_name = 'buffalo_m'
output_path = os.path.join('./face_detection_output', model_name)
if os.path.exists(os.path.join('./face_detection_output', model_name)):
    shutil.rmtree(output_path)

os.mkdir(output_path)
# Method-1, use FaceAnalysis
app = FaceAnalysis(name=model_name, allowed_modules=['detection', 'landmark_2d_106']) # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))

t = time.time()
data_path = glob.glob('./face_data/*.*')
for path in data_path:
    file_name = os.path.split(path)[1]
    img = cv2.imread(path)
    faces = app.get(img)
    print(faces)
    exit()
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


# result from face detection
'''
[{
    'bbox': array([848.7424 , 431.87067, 947.8905 , 546.81647], dtype=float32), 
    'kps': array([[865.9018 , 475.89682],
       [907.88257, 473.09155],
       [884.03687, 502.3584 ],
       [876.5345 , 517.968  ],
       [910.0793 , 515.45746]], dtype=float32), 
    'det_score': 0.7496564, 
    'landmark_2d_106': array([[892.3794 , 535.6986 ],
       [854.2152 , 475.13077],
       [866.3828 , 515.269  ],
       [869.2705 , 519.0873 ],
        ...
'''