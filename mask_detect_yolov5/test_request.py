import requests
import time 

url = "http://127.0.0.1:8000/detect/"
t0 = time.time()

for i in range(50):
    t = time.time()

    payload={'name_cam': ''}
    files=[
    ('image',('Screenshot from 2022-03-07 23-37-31.png',open('/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/Screenshot from 2022-03-07 23-37-31.png','rb'),'image/png'))
    ]
    print(files)
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    t2 = time.time() - t
    print(response.text, t2)

    
    t = time.time()

    payload={'name_cam': ''}
    files=[
    ('image',('face_1_in.jpg',open('/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/face_1_in.jpg','rb'),'image/jpg'))
    ]
    print(files)
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    t2 = time.time() - t
    print(response.text, t2)

print(time.time() - t0)