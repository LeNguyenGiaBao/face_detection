import requests
import glob 
import time 
import os 

url = "http://0.0.0.0:8000/detect/"

num_files = len(glob.glob('/home/giabao/Documents/face/face_verification/data/original_data/*/*.*'))
t1 = time.time()
for file_path in glob.glob('/home/giabao/Documents/face/face_verification/data/original_data/*/*.*'):
    file_name = os.path.split(file_path)[1]
    payload={'name_cam': ''}
    files=[
      ('image',(file_name,open(file_path,'rb'),'image/png'))
    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.text)

t2 = time.time()
print("Total time", t2 - t1)
print("Total file", num_files)
print("Time per image", (t2-t1)/num_files)