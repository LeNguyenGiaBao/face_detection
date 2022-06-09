import cv2
from centerface import CenterFace
import onnxruntime as rt
from insightface.app import FaceAnalysis

def load_retinaface(model_name = 'buffalo_l'):
    model_face_detect = FaceAnalysis(name=model_name, allowed_modules=['detection']) # enable detection model only
    model_face_detect.prepare(ctx_id=0, det_size=(640, 640))

    return model_face_detect

def load_yolov5(weight_path='./weight/mask_yolov5/best.pt'):
    # import torch 
    # # load from torch hub and apply local weight
    # model = torch.hub.load(torch.hub.get_dir() + '/ultralytics_yolov5_master/', 'custom', weight_path, source='local') 
     
    # return model 
    pass

def load_yunet(weight_path='./weight/face_yunet/yunet.onnx'):
    # TODO: modify config for best accuracy
    model = cv2.FaceDetectorYN.create(
        model=weight_path,
        config="",
        input_size=(320, 320),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    )

    return model 

def load_centerface(weight_path='./weight/face_center/centerface.onnx'):
    model = CenterFace(weight_path)
    return model

def load_vgg(weight_path='./weight/mask_vgg/vgg16_18_0.73.onnx'):
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
        }),
        'CPUExecutionProvider',
    ]
    model = rt.InferenceSession(weight_path, providers=providers)

    return model 

def load_face_detect_model(model_name):
    if model_name == 'yunet':
        return load_yunet()
    elif model_name == 'retina':
        return load_retinaface()
    elif model_name == 'center':
        return load_centerface()
    
def load_mask_detect_model(model_name):
    if model_name == 'yolov5':
        return load_yolov5()
    elif model_name == 'vgg':
        return load_vgg()
