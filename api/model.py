import cv2
import torch 

def load_retinaface(weight_path):
    pass

def load_yolov5(weight_path='./weight/mask_yolov5/best.pt'):
    # load from torch hub and apply local weight
    model = torch.hub.load(torch.hub.get_dir() + '/ultralytics_yolov5_master/', 'custom', weight_path, source='local') 
     
    return model 

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

def load_face_detect_model(model_name):
    if model_name == 'yunet':
        return load_yunet()
    elif model_name == 'retina':
        return load_retinaface()
    
def load_mask_detect_model(model_name):
    if model_name == 'yolov5':
        return load_yolov5()
