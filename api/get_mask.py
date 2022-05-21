def get_mask(model_name, model, img):
    if model_name == 'yolov5':
        predict = model(img).pred[0]

        if predict.shape[0] == 0: # no face
            return None 

        # get the biggest face
        predict_sorted = sorted(predict, key=lambda x:((x[2]-x[0])*(x[3]-x[1])), reverse=True)[0]
        is_mask = int(predict_sorted[5].item())

        return is_mask

    return None


if __name__=="__main__":
    import cv2 
    from model import load_mask_detect_model

    model_name = 'yolov5'
    model = load_mask_detect_model(model_name)

    # img = cv2.imread('/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/noface.png')
    img = cv2.imread('/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/Screenshot from 2022-02-08 09-54-05.png')
    # img = cv2.imread('/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/multi_face.jpeg')
    # img = cv2.imread('/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/Screenshot from 2022-02-08 09-59-55.png')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    is_mask = get_mask(model_name, model, img)
    print(is_mask) # 1