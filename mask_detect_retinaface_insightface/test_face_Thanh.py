from time import time
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from insightface.model_zoo import RetinaFace
# from insightface.src import face_preprocess
# https://github.com/deepinsight/insightface/blob/ce3600a74209808017deaf73c036759b96a44ccb/recognition/arcface_mxnet/common/build_eval_pack.py#L72
# This function is too old, might in old version (15 months ago)
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from insightface.utils import face_align, DEFAULT_MP_NAME
from imutils import face_utils
import matplotlib.pyplot as plt

# from my_utils import *

# thresh = 0.8
# mask_thresh = 0.2

thresh_mask = 165 # set the threshold for mask detection
# scales = [640, 1080]

# count = 1

# gpuid = 0
class MyFaceAnalysis(FaceAnalysis):

    def __init__(self, name=DEFAULT_MP_NAME, root='~/.insightface', allowed_modules=None, **kwargs):
        super(MyFaceAnalysis, self).__init__()
            


def my_norm_crop(img, landmark, image_size=112, mode='arcface'):
    M, pose_index = face_align.estimate_norm(landmark, image_size, mode)
    print('M', M)
    print('pose_index', pose_index)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    print('warped', warped.shape)

    return warped

def compute_sim(feat1, feat2):
    from numpy.linalg import norm
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim

if __name__ == '__main__':
    app = MyFaceAnalysis(allowed_modules=['detection', 'landmark_2d_106', 'recognition'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    # img_1_in = cv2.imread('./face_emb/face 4 out.jpg')
    img_1_in = cv2.imread('./img/face 1 in.jpg')
    # img_2_in = cv2.imread('./face_emb/face 2 in.jpg')
    # img_2_out = cv2.imread('./face_emb/face 2 out.jpg')

    faces_1_in = app.get(img_1_in)
    # faces_1_out = app.get(img_1_out)
    # faces_2_in = app.get(img_2_in)
    # faces_2_out = app.get(img_2_out)

    print("=================================================================================")
    # print(faces_1_in)
    print("=================================================================================")
    
    color = (200, 160, 75)
    if img_1_in is not None:
        for face in faces_1_in:
            img_tmp = np.zeros(img_1_in.shape, dtype="uint8")
            
            x1, y1, x2, y2 = face.bbox
            x1 , y1 = int(x1), int(y1)
            x2 , y2 = int(x2), int(y2)
            print(x1, y1, x2, y2)
            cv2.rectangle(img_1_in, (x1, y1), (x2, y2), color, 2)
            _face = img_1_in[y1:y1+y2, x1:x2+x1]
            lmk = face.landmark_2d_106
            lmk = np.round(lmk).astype(np.int)
            for i in range(lmk.shape[0]):
                p = tuple(lmk[i])
                cv2.circle(img_1_in, p, 1, color, 1, cv2.LINE_AA)
            # lmk = face_utils.shape_to_np(lmk)
            (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
            print("=================================================================")
            print("LANDMARKS_IDSXS: \n", face_utils.FACIAL_LANDMARKS_IDXS)
            print("=================================================================")
            mouth = lmk[mStart:mEnd]
            print('mouth: ', mouth)
            # print('mouth', mouth.shape)
            boundRect = cv2.boundingRect(mouth)
            cv2.rectangle(img_1_in,
                        (int(boundRect[0]), int(boundRect[1])),
                        (int(boundRect[0] + boundRect[2]),  int(boundRect[1] + boundRect[3])), (0,0,255), 2)
            
            hsv = cv2.cvtColor(img_1_in[int(boundRect[1]):int(boundRect[1] + boundRect[3]),int(boundRect[0]):int(boundRect[0] + boundRect[2])], cv2.COLOR_RGB2HSV)
            sum_saturation = np.sum(hsv[:, :, 1])
            area = int(boundRect[2])*int(boundRect[3])
            avg_saturation = sum_saturation / area
            # Check va canh bao voi threshold
            
            print("=================================================================")
            print("Check avg_saturation: ", avg_saturation)
            if avg_saturation > thresh_mask:
                
                print("co khau trang")
                cv2.putText(img_1_in, "coi khau trang ra", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)
            else:
                
                print("khong co khau trang")
    # plt.imshow(tim)
    # plt.show()

    # emb_1_in = faces_1_in.embedding
    # emb_1_out = faces_1_out.embedding
    # emb_2_in = faces_2_in.embedding
    # emb_2_out = faces_2_out.embedding

    # sim_1 = compute_sim(emb_1_in, emb_1_out)
    # sim_2 = compute_sim(emb_2_in, emb_2_out)
    # sim_dif = compute_sim(emb_1_in, emb_2_out)
    # print(sim_1)
    # print(sim_2)
    # print(sim_dif)
    plt.imshow(img_1_in)
    plt.show()