import requests
import glob 
import time 
import os 

url = "http://127.0.0.1:8000/detect/"

num_files = len(glob.glob(r'F:\a_kltn\code\original_face\*.*'))
t1 = time.time()
for file_path in glob.glob(r'F:\a_kltn\code\original_face\*.*'):
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

# yunet + yolov5
# Total time 8.324703931808472
# Total file 30
# Time per image 0.27749013106028236

# yunet + vgg classify
# Total time 2.7675187587738037
# Total file 30
# Time per image 0.09225062529246013