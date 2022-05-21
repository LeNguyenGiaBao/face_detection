import cv2
import glob
import time
from model import SSD, Predictor
from utils.utils import draw_boxes


class_names = ['BACKGROUND', 'no_mask', 'with_mask']
model_path = './weight/vgg16-ssd-Epoch-35-Loss-1.7198709601705724.pth'

net = SSD(len(class_names), is_test=True)
net.load(model_path)
predictor = Predictor(net, nms_method='soft', candidate_size=200)

image_path = '/home/giabao/Documents/face/face_detection/mask_detect_yolov5/test_data/img/*'

t1 = time.time()
for i in glob.glob(image_path):
    img = cv2.imread(i)
    boxes, labels, probs = predictor.predict(img, 100, 0.3)
    img = draw_boxes(img, boxes, labels, probs, class_names)

    new_path = i.replace('mask_detect_yolov5/test_data/img', 'mask_detect_ssd/result_test_data')
    print(new_path)
    cv2.imwrite(new_path, img)

t2 = time.time()

print(t2 - t1, len(glob.glob(image_path)))