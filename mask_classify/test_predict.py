import cv2 
import os
import glob 
import numpy as np
import onnxruntime as rt 
import time 


providers = ['CPUExecutionProvider']
m = rt.InferenceSession('./weight/vgg16_18_0.73.onnx')
output_names = ['dense_1']

data_path = glob.glob('/home/giabao/Documents/face/face_verification/data/processed_data/*/*.png')
for path in data_path:
    file_name = os.path.split(path)[1]
    img = cv2.imread(path)
    img = cv2.resize(img, (80, 80))
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    
    t1 = time.time()
    predict = m.run(output_names, {"input": img})[0][0][0]
    t2 = time.time()

    predict_class = round(predict)

    print(t2-t1, predict, predict_class)

    cv2.imwrite(str(predict_class) + '_' + file_name, img[0])