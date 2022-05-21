def get_relative_landmark(landmark, x_box, y_box):
    landmark = landmark.reshape(-1, 2)
    landmark[:, 0] = landmark[:, 0] - x_box
    landmark[:, 1] = landmark[:, 1] - y_box

    landmark_flatten = landmark.flatten()
    landmark_list = list(landmark_flatten)
    landmark_string = ','.join(map(str,landmark_list))

    return landmark_string

def get_face(model_name, model, img):
    img_height, img_width = img.shape[0], img.shape[1]

    if model_name == 'yunet':
        model.setInputSize(((img_width, img_height)))

        _, face_detect = model.detect(img)   

        if face_detect is None:
            return None
        
        # get the biggest face
        face_detect_sorted = sorted(face_detect, key=lambda x:(x[2]*x[3]), reverse=True)[0]
        x_box, y_box, w_box, h_box = face_detect_sorted[:4].astype(int)
        bbox = '{},{},{},{}'.format(x_box, y_box, w_box, h_box)

        landmark = face_detect_sorted[4:14].astype(int)  # 1-d array

        # TODO: add_padding function
        relative_landmark = get_relative_landmark(landmark, x_box, y_box)
        
        return (bbox, relative_landmark)

    return None



if __name__=="__main__":
    import cv2 
    from model import load_face_detect_model

    model_name = 'yunet'
    model = load_face_detect_model(model_name)

    # img = cv2.imread('/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/noface.png')
    img = cv2.imread('/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/multi_face.jpeg')
    # img = cv2.imread('/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/Screenshot from 2022-02-08 09-58-44.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bbox, relative_landmark = get_face(model_name, model, img)
    print(bbox)
    print(relative_landmark)