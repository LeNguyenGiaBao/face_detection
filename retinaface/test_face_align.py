import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from insightface.model_zoo import RetinaFace
from insightface.src. import face_preprocess
# https://github.com/deepinsight/insightface/blob/ce3600a74209808017deaf73c036759b96a44ccb/recognition/arcface_mxnet/common/build_eval_pack.py#L72
# This function is too old, might in old version (15 months ago)

import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from insightface.utils import face_align

if __name__ == '__main__':
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    # img = ins_get_image('t1')
    img = cv2.imread('./face_data/Screenshot from 2022-02-08 10-00-00.png')
    faces = app.get(img)
    # face detect 
    rimg = app.draw_on(img, faces)
    cv2.imwrite('bbox.jpg', rimg)

    # face align
    aimg = face_align.norm_crop(img, landmark=faces[0].kps)
    cv2.imwrite('./aimg.jpg', aimg)

    # draw 68 points
    #assert len(faces)==6
    tim = img.copy()
    color = (200, 160, 75)
    for face in faces:
        lmk = face.landmark_2d_106
        lmk = np.round(lmk).astype(np.int)
        for i in range(lmk.shape[0]):
            p = tuple(lmk[i])
            cv2.circle(tim, p, 1, color, 1, cv2.LINE_AA)
    cv2.imwrite('./68_points.jpg', tim)

    # transform
    cropped, M = face_align.transform(img, faces[0].kps, 300, 1, 0)
    cv2.imwrite('cropped.jpg', cropped)

    # preprocess
    preprocess_img = face_preprocess.preprocess(_npdata, bbox = bbox, landmark=landmark, image_size=self.image_size)