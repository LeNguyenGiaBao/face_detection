import torch
import cv2 
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
import asyncio
import uvicorn
from insightface.app import FaceAnalysis

# Model
model_mask_detect = torch.hub.load('ultralytics/yolov5', 'custom', './weights/best.pt')  # or yolov5m, yolov5l, yolov5x, custom
model_name = 'buffalo_m'
model_face_detect = FaceAnalysis(name=model_name, allowed_modules=['detection']) # enable detection model only
model_face_detect.prepare(ctx_id=0, det_size=(640, 640))

# Fast API
app = FastAPI()

@app.get("/")
def index():
    return {"name" : "giabao"}

@app.post("/detect/")
async def detect(name_cam: str = Form(""), image: UploadFile = File(None)):
    try:
        # print(type(image))
        if image == None:
            return jsonable_encoder({
                "code": 201,
                "error_code": 1,
                "msg": "Missing Input Image"
            })

        contents = await asyncio.wait_for(image.read(), timeout=1) 
        if(str(contents) =="b''"):
            return jsonable_encoder({
                "code": 201,
                "error_code": 2,
                "msg": "Not found file"
            })
        
        # check image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonable_encoder({
                "code": 201,
                "error_code": 3,
                "msg": "Input is not an image"
            })

        # model forward
        pred = model_mask_detect(img).pred[0]

        # assumption main face is the biggest face -> sort by face area
        # NOTE: save all face 
        pred = sorted(pred, key=lambda x:((x[2]-x[0])*(x[3]-x[1])), reverse=True)[0]
        cls = pred[5].item()
        
        if cls == 1:    # 1: mask 
            return jsonable_encoder({
                "code": 200,
                "data": 1,
                "msg": "With Mask"
            }) 
        
        elif cls == 0:  # 0: no mask
            face_detect = model_face_detect.get(img)
            print(face_detect[0]['bbox'])
            x1, y1, x2, y2 = face_detect[0]['bbox']
            print(x1)
            return jsonable_encoder({
                "code": 200,
                "data": 0,
                "msg": "No Mask", 
                'x1': int(x1), 
                'y1': int(y1),
                'x2': int(x2), 
                'y2': int(y2)
            })
            # no mask: send result for save image -> face verify

    except Exception:
        return jsonable_encoder({
                "code": 201,
                "error_code": 0,
                "msg": Exception
            })

if __name__ == "__main__":
    # Scale
    x = 650
    y = 250
    w = 700
    h = 700

    # Model
    # model = torch.hub.load('ultralytics/yolov5', 'custom', './weights/best.pt')  # or yolov5m, yolov5l, yolov5x, custom

    # run API
    uvicorn.run('app:app', host="0.0.0.0", port=8000, reload=True)