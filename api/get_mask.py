import cv2 
import numpy as np 

def get_mask_yolov5(model, img):
    predict = model(img).pred[0]

    if predict.shape[0] == 0: # no face
        return None 
    # get the biggest face
    predict_sorted = sorted(predict, key=lambda x:((x[2]-x[0])*(x[3]-x[1])), reverse=True)[0]
    is_mask = int(predict_sorted[5].item())

    return is_mask

def get_mask_vgg(model, img):
    # assume that image is a cropped face
    img = cv2.resize(img, (80, 80)) # (80,80) is input config of vgg
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    output_names = ['dense_1']
    predict = model.run(output_names, {"input": img})[0][0][0]
    is_mask = round(predict)

    return is_mask

def get_mask(model_name, model, img):
    if model_name == 'yolov5':
        return get_mask_yolov5(model, img)
    elif model_name == 'vgg':
        return get_mask_vgg(model, img)

    return None


if __name__=="__main__":
    from model import load_mask_detect_model

    # model_name = 'yolov5'
    # model = load_mask_detect_model(model_name)

    # # img = cv2.imread('/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/noface.png')
    # img = cv2.imread('/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/Screenshot from 2022-02-08 09-54-05.png')
    # # img = cv2.imread('/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/multi_face.jpeg')
    # # img = cv2.imread('/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/Screenshot from 2022-02-08 09-59-55.png')

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # is_mask = get_mask(model_name, model, img)
    # print(is_mask) # 1


    # ----------------------------- vgg -----------------------------

    model_name = 'vgg'
    model = load_mask_detect_model(model_name)

    # img = cv2.imread('/home/giabao/Documents/face/face_verification/data/processed_data/ninh_vao/3857309562_134850_PLATE_1_1_34,33,59,27,58,48,41,63,59,58.png')
    # img = cv2.imread('/home/giabao/Documents/face/face_verification/data/processed_data/ninh_vao/3857309562_134850_PLATE_1_1_34,33,59,27,58,48,41,63,59,58.png')
    # img = cv2.imread('/home/giabao/Documents/face/face_verification/data/processed_data/thay_ra/3444605152_134536_PLATE_0_1_18,37,51,35,35,56,25,69,47,67.png')
    img = cv2.imread('/home/giabao/Documents/face/face_detection/mask_classify_mb2/0_0028802154_134931_PLATE_0_1_12,10,37,10,24,25,17,35,35,35.png')

    is_mask = get_mask(model_name, model, img)
    print(is_mask) # 1