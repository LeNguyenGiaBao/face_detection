# Mask detection using Yolov5

#### Update 22_04_24:
- add landmark to response
- remove padding from response

#### Update 22_04_11:
- add padding to box, default 5%
- send padding value to box1 message (see above)
- load model yolov5 from [cache](https://github.com/LeNguyenGiaBao/face_detection/blob/master/mask_detect_yolov5/app.py#L13), need internet for the first time (to load model)
- Time to run 71 images randomly is 26.67s -> 0.37s/image


#### Update 22_04_05:
- add bbox with mask face
- check face detect: if face detection check fail (because the image is cropped so narrow) -> return "No Face"


#### Update 22_03_19:
- add bbox with no mask face
- TODO: make with 2 faces.



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
    "code": 200,                # success
    "data": 0,                  # 0 with no mask and 1 with mask
    "msg": "No Mask"
    "box1": "408,243,98,114,0.05",   # bbox info with both mask and no mask. 0.05: PADDING_RATIO
    "landmark1": "725,573,780,563,758,605,744,630,784,622"
  }
  ```
  
  ```
  {
    "code": 200,
    "data": 2,
    "msg": "No Face"
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

