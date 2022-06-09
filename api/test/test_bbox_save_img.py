import cv2 
import requests
import glob 
import time 
import os 

url = "http://127.0.0.1:8000/detect/"
num_files = len(glob.glob('../../../../image/*.png'))
t1 = time.time()
for file_path in glob.glob('../../../../image/*.png'):
    full_path = os.path.abspath(file_path)
    file_name = os.path.split(file_path)[1]
    payload={'name_cam': ''}
    files=[
      ('image',(file_name,open(full_path,'rb'),'image/png'))
    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    response = response.json()
    if response['code'] == 200 and response['data'] < 2:
        bbox = response['box1']
        x, y, w, h = bbox.split(',')
        x, y, w, h = int(x), int(y), int(w), int(h)

        img = cv2.imread(full_path)
        face = img[y:y+h, x:x+w]
        cv2.imwrite(full_path.replace('image', 'crop_face'), face)

t2 = time.time()
print("Total time", t2 - t1)
print("Total file", num_files)
print("Time per image", (t2-t1)/num_files)