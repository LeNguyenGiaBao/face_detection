# import torch
import cv2 
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
import asyncio
import uvicorn
import base64
from model import load_face_detect_model, load_mask_detect_model
from get_face import get_face, get_croped_face
from get_mask import get_mask

# config
FACE_DETECT_MODEL = 'retina' # or 'retina'
MASK_DETECT_MODEL = 'vgg' # or 'yolov5'
PADDING_RATIO = 0

# load model 
face_detect_model = load_face_detect_model(FACE_DETECT_MODEL)
# mask_detect_model = load_mask_detect_model(MASK_DETECT_MODEL)

# Fast API
app = FastAPI()

@app.get("/")
def index():
    return {"name" : "giabao"}

@app.post("/detect/")
async def detect(name_cam: str = Form(""), image: str = Form("")):
    try:
        if image == "":
            return jsonable_encoder({
                "code": 201,
                'name_cam': name_cam,
                "error_code": 1,
                "msg": "Missing Input Image"
            })
        
        # # check image
        contents = base64.b64decode(image)
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonable_encoder({
                "code": 201,
                'name_cam': name_cam,
                "error_code": 3,
                "msg": "Input is not an image"
            })
        
        is_face = get_face(FACE_DETECT_MODEL, face_detect_model, img)
        if is_face is None:
            return jsonable_encoder({
                "code": 200,
                'name_cam': name_cam,
                "data": 2,
                "msg": "No Face",
            }) 

        bbox, landmark = is_face
        print(bbox)
        croped_face = get_croped_face(img, bbox)
        if croped_face is None:
            return jsonable_encoder({
                "code": 200,
                'name_cam': name_cam,
                "data": 2,
                "msg": "No Face",
            }) 

        # is_mask = get_mask(MASK_DETECT_MODEL, mask_detect_model, croped_face)
        is_mask = 0
        if is_mask == 1:    # 1: mask 
            return jsonable_encoder({
                "code": 200,
                'name_cam': name_cam,
                "data": 1,
                "msg": "With Mask", 
                'box1': bbox,
                'landmark1': landmark, 
                'face_model': FACE_DETECT_MODEL,
                'mask_model': MASK_DETECT_MODEL,
            }) 
        
        elif is_mask == 0:  # 0: no mask
            return jsonable_encoder({
                "code": 200,
                'name_cam': name_cam,
                "data": 0,
                "msg": "No Mask", 
                'box1': bbox,
                'landmark1': landmark, 
                'face_model': FACE_DETECT_MODEL,
                'mask_model': MASK_DETECT_MODEL,
            }) 
        else:
            return jsonable_encoder({
                "code": 200,
                'name_cam': name_cam,
                "data": is_mask,
                "msg": "", 
                'box1': bbox,
                'landmark1': landmark, 
                'face_model': FACE_DETECT_MODEL,
                'mask_model': MASK_DETECT_MODEL,
            }) 

    except Exception as e:
        print(e)
        return jsonable_encoder({
                "code": 201,
                'name_cam': name_cam,
                "error_code": 0,
                "msg": str(e)
            })

@app.post("/detect_file/")
async def detect_file(name_cam: str = Form(""), image: UploadFile = File(None)):
    try:
        if image == None:
            return jsonable_encoder({
                "code": 201,
                'name_cam': name_cam,
                "error_code": 1,
                "msg": "Missing Input Image"
            })

        contents = await asyncio.wait_for(image.read(), timeout=1) 
        if(str(contents) =="b''"):
            return jsonable_encoder({
                "code": 201,
                'name_cam': name_cam,
                "error_code": 2,
                "msg": "Not found file"
            })
        
        # # check image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonable_encoder({
                "code": 201,
                'name_cam': name_cam,
                "error_code": 3,
                "msg": "Input is not an image"
            })
        
        is_face = get_face(FACE_DETECT_MODEL, face_detect_model, img)
        if is_face is None:
            return jsonable_encoder({
                "code": 200,
                'name_cam': name_cam,
                "data": 2,
                "msg": "No Face",
            }) 

        bbox, landmark = is_face
        croped_face = get_croped_face(img, bbox)
        if croped_face is None:
            return jsonable_encoder({
                "code": 200,
                'name_cam': name_cam,
                "data": 2,
                "msg": "No Face",
            }) 

        # is_mask = get_mask(MASK_DETECT_MODEL, mask_detect_model, croped_face)
        is_mask = 0
        if is_mask == 1:    # 1: mask 
            return jsonable_encoder({
                "code": 200,
                'name_cam': name_cam,
                "data": 1,
                "msg": "With Mask", 
                'box1': bbox,
                'landmark1': landmark, 
                'face_model': FACE_DETECT_MODEL,
                'mask_model': MASK_DETECT_MODEL,
            }) 
        
        elif is_mask == 0:  # 0: no mask
            return jsonable_encoder({
                "code": 200,
                'name_cam': name_cam,
                "data": 0,
                "msg": "No Mask", 
                'box1': bbox,
                'landmark1': landmark, 
                'face_model': FACE_DETECT_MODEL,
                'mask_model': MASK_DETECT_MODEL,
            }) 
        else:
            return jsonable_encoder({
                "code": 200,
                'name_cam': name_cam,
                "data": is_mask,
                "msg": "", 
                'box1': bbox,
                'landmark1': landmark, 
                'face_model': FACE_DETECT_MODEL,
                'mask_model': MASK_DETECT_MODEL,
            }) 

    except Exception as e:
        print(e)
        return jsonable_encoder({
                "code": 201,
                'name_cam': name_cam,
                "error_code": 0,
                "msg": str(e)
            })

if __name__ == "__main__":
    # run API
    uvicorn.run('app:app', host="0.0.0.0", port=8000, reload=True)