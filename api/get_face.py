import numpy as np 

PADDING = 50 # pixel
def get_relative_landmark(landmark, x_box, y_box):
    landmark = landmark.reshape(-1, 2)
    landmark[:, 0] = landmark[:, 0] - x_box
    landmark[:, 1] = landmark[:, 1] - y_box

    landmark_flatten = landmark.flatten()
    landmark_list = list(landmark_flatten)
    landmark_string = ','.join(map(str,landmark_list))

    return landmark_string
    

def get_croped_face(img, bbox):
    x_box, y_box, w_box, h_box = bbox.split(',')
    x_box = int(x_box)
    y_box = int(y_box)
    w_box = int(w_box)
    h_box = int(h_box)
    croped_face = img[y_box:y_box+h_box, x_box:x_box+w_box]

    return croped_face

def convert_x2y2_to_width_height(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1

    x1 = x1 - PADDING
    y1 = y1 - PADDING
    w = w + 2*PADDING
    h = h + 2*PADDING
    
    return x1, y1, w, h


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
        if w_box < 40 or h_box < 40:
            return None
            
        bbox = '{},{},{},{}'.format(x_box, y_box, w_box, h_box)

        landmark = face_detect_sorted[4:14].astype(int)  # 1-d array

        # TODO: add_padding function
        relative_landmark = get_relative_landmark(landmark, x_box, y_box)
        
        return (bbox, relative_landmark)

    if model_name == "center":
        face_detect, lms = model(img, img_height, img_width, threshold=0.35)

        if face_detect.shape[0] == 0:
            return None 

        # get the biggest face
        face_detect_sorted = sorted(face_detect, key=lambda x:((x[2]-x[0])*(x[3]-x[1])), reverse=True)[0]
        rows, cols = np.where(face_detect == face_detect_sorted)
        x1, y1, x2, y2 = face_detect_sorted[:4].astype(int)
        x_box, y_box, w_box, h_box = convert_x2y2_to_width_height(x1,y1,x2,y2)
        if w_box < 40 + 2 * PADDING or h_box < 40 + 2 * PADDING:
            return None
        
        if x_box < 0 or y_box < 0 or x_box + w_box > img_width or y_box + h_box > img_height:
            return None
            
        bbox = '{},{},{},{}'.format(x_box, y_box, w_box, h_box)

        # TODO: make landmark after sort
        landmark = lms[rows[0]].astype(int)
        relative_landmark = get_relative_landmark(landmark, x_box, y_box)

        return (bbox, relative_landmark)

    if model_name == "retina":
        face_detect = model.get(img)

        if face_detect.shape[0] == 0:
            return None 

        face_detect_sorted = sorted(face_detect, key=lambda x:((x['bbox'][2]-x['bbox'][0])* (x['bbox'][3]-x['bbox'][1])), reverse=True)[0]
        x1, y1, x2, y2 = face_detect_sorted['bbox'][:4].astype(int)
        x_box, y_box, w_box, h_box = convert_x2y2_to_width_height(x1,y1,x2,y2)
        bbox = '{},{},{},{}'.format(x_box, y_box, w_box, h_box)

        # TODO: make landmark after sort
        relative_landmark = ""

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