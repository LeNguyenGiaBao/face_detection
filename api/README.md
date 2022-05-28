# face_detection

## face_detection API 

### Idea
Use 2 models for 2 tasks:
- Face detection: Yunet
- Mask detection: VGG 

Face detection in parking lot:
- Get face from image
- Using that face for mask classifier (faster and easier when using mask detection model)

### Timing
- Yunet + vgg classifier: 0.09225062529246013 
- Yunet + yolov5: 0.27749013106028236
- [PC]: Yunet + vgg classifier: 0.029399174891923 (reduce 3 times) (CPU 58%, GPU 15%, 300 requests)


Note: 
- [PC]: i5-9400F, 8gb RAM, GTX 1650 4gb
- [Laptop]: i5-8265U, 12gb RAM, MX 230 2gb

### Run on Python
The API run on http://localhost:8000
```
cd api
python app.py
```

### API for C#
```
var client = new RestClient("http://0.0.0.0:8000/detect/");
client.Timeout = -1;
var request = new RestRequest(Method.POST);
request.AddParameter("name_cam", "");
request.AddFile("image", "/home/giabao/Documents/face/face_verification/data/original_data/thay_ra/3444605152_134536_PLATE_4.png");
IRestResponse response = client.Execute(request);
Console.WriteLine(response.Content);
```

Result
```
{
    "code": 200,
    "data": 1,
    "msg": "With Mask",
    "box1": "207,88,87,88",
    "landmark1": "21,31,51,39,24,54,16,64,39,71",
    "face_model": "yunet",
    "mask_model": "vgg"
}
```

Error
```
{
    "code": 201,
    "error_code": 0,
    "msg": "Error Message"
}
```

### Update
#### Update 22_05_26:
- Mask classify: VGG16 onnx model, base on Tensorflow
- Total time 2.7675187587738037
- Total file 30
- Time per image 0.09225062529246013

#### Update 22_05_22:
- Refactor code of api
- Can choose face model and mask model
- Result
    - face model: yunet
    - mask model: yolov5
    - using GPU
    - Total time 8.971619367599487
    - Total file 30
    - Time per image 0.2990539789199829
- TODO: retinaface model (face model) and SSD (mask model)
