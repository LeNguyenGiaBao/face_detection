import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
import cv2 
import numpy as np


def compute_sim(feat1, feat2):
    from numpy.linalg import norm
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim

img1 = cv2.imread('./anh_BK/nguoi1/face 1 in.jpg')
img2 = cv2.imread('./anh_BK/nguoi1/face 1 out.jpg')
img3 = cv2.imread('./anh_BK/nguoi2/face 2 in.jpg')
img4 = cv2.imread('./anh_BK/nguoi2/face 2 out.jpg')

model_name = 'buffalo_m'
# app = FaceAnalysis(name=model_name, allowed_modules=['detection', 'landmark_2d_106']) # enable detection model only
app = FaceAnalysis(name=model_name, allowed_modules=['detection', 'recognition']) # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))

handler = insightface.model_zoo.get_model('/home/giabao/.insightface/models/buffalo_l/w600k_r50.onnx')
handler.prepare(ctx_id=0)

face1 = app.get(img1)[0]
face2 = app.get(img2)[0]
face3 = app.get(img3)[0]
face4 = app.get(img4)[0]

# cv2.imwrite('./img/face1.jpg', app.draw_on(img1, face1))
# cv2.imwrite('./img/face2.jpg', app.draw_on(img2, face2))
# cv2.imwrite('./img/face3.jpg', app.draw_on(img3, face3))
# cv2.imwrite('./img/face4.jpg', app.draw_on(img4, face4))

emb1 = face1.embedding
emb2 = face2.embedding
emb3 = face3.embedding
emb4 = face4.embedding

sim = compute_sim(emb1, emb2)
sim2 = compute_sim(emb1, emb1)
sim3 = compute_sim(emb1, emb4)
sim4 = compute_sim(emb3, emb4)
print(1- sim, sim2, 1-sim3, 1-sim4)
# model l 0.58802897 0.99999994 0.024464995 0.42700675
# model m 0.5661029 0.99999994 0.013427707 0.4412302
# emb0 = handler.get(img1, face1)
# print(emb0 == emb1)

