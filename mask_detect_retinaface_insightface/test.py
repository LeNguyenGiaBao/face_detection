import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface_cov import RetinaFaceCoV

# https://github.com/lcings/RetinaFaceAntiCov/blob/master/model/mnet_cov2-symbol.json
thresh = 0.2
mask_thresh = 0.2
scales = [640, 1080]

count = 1
gpuid = -1
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

x = 650
y = 250
w = 700
h = 700

# detector = RetinaFaceCoV('./model/mnet_cov1', 0, gpuid, 'net3')
detector = RetinaFaceCoV('./model/mnet_cov2', 0, gpuid, 'net3l')

# data_path = glob.glob('./img/*.png')
data_path = glob.glob('./test.jpg')
output_dir = './output'

for path in data_path:
    file_name = os.path.split(path)[1]
    print(file_name)
    img = cv2.imread(path)
    # try:
    #     img = img[y:y+h, x:x+w]
    # except:
    #     pass
    im_shape = img.shape
    scales = [640, 1080]
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    #im_scale = 1.0
    #if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]
    flip = False

    for c in range(count):
        faces, landmarks = detector.detect(img,
                                        thresh,
                                        scales=scales,
                                        do_flip=flip)

    if faces is not None:
        for i in range(faces.shape[0]):
            #print('score', faces[i][4])
            face = faces[i]
            box = face[0:4].astype(int)
            det_score = face[4]
            mask_score = face[5]
            #color = (255,0,0)
            if mask_score >= mask_thresh:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(img, '{:.2f} {:.2f}'.format(det_score, mask_score), (box[0], box[1]), font, fontScale, color, thickness, cv2.LINE_AA)
            landmark5 = landmarks[i].astype(int)
            #print(landmark.shape)
            for l in range(landmark5.shape[0]):
                color = (255, 0, 0)
                cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

        filename = os.path.join(output_dir, file_name)
        cv2.imwrite(filename, img)
