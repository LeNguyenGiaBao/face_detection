# Mask detection using Yolov5

### Data from [@waittim/mask-detector](https://github.com/waittim/mask-detector/tree/master/modeling/data)

### How to use
- Install dependences
  ```
  pip install -r requirements.txt
  ```
  
- Run API:
  ```
  python app.py
  ```
  The API run on **http://0.0.0.0:8000/**
  
- Test on browser: 
  - Access doc of API in [http://0.0.0.0:8000/docs](http://0.0.0.0:8000/docs)
  - Go to `/detect` API (Post)
  - Upload image and name camera (optional)
  - Receive the result
- Test on Postman:
  - Import collection and upload image to see the result

### API for C# 
```
var client = new RestClient("http://127.0.0.1:8000/detect/");
client.Timeout = -1;
var request = new RestRequest(Method.POST);
request.AddParameter("name_cam", "");
request.AddFile("image", "/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/Screenshot from 2022-03-07 23-37-01.png");
IRestResponse response = client.Execute(request);
Console.WriteLine(response.Content);
```

### Result Format
- Success 
  ```
  {
    "code": 200,  # success
    "data": 0,    # 0 with no mask and 1 with mask
    "msg": "No Mask"
    "x1": x1,
    "y1": y1,
    "x2": x2,
    "y2": y2,
  }
  ```
  
- Fail
  ```
  {
    "code": 201,   # error
    "error_code": 0,
    "msg": "error message"
  }
  ```
  
I try to test with many boundary case, if you test and meet bug that shut down the API, please contact me!!!

### Update 22_03_19:
- add bbox with no mask face
- TODO: make with 2 facé.
