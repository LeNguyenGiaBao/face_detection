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
num_files = len(data_path)
t1 = time.time()

for path in data_path:
    file_name = os.path.split(path)[1]
    img = cv2.imread(path)
    img = cv2.resize(img, (80, 80))
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    
    predict = m.run(output_names, {"input": img})[0][0][0]
    predict_class = round(predict)
    print(predict_class, predict)

t2 = time.time()
print("Total time", t2 - t1)
print("Total file", num_files)
print("Time per image", (t2-t1)/num_files)