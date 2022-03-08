from email.mime import base
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import transform as trans

import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align


img_path = './img/face 1 in.jpg'

Kl = 125
Kh = 188
Ymin = 16
Ymax = 235
Wcb = 46.97
Wcr = 38.76
WLcb = 23
WLcr = 20
WHcb = 14
WHcr = 10
Cx = 109.38
Cy = 152.02
theta = 2.53 * np.pi / 180
ec_x = 1.60
ec_y = 2.41
a = 25.39
b = 14.03

new_width = 256
new_height = 256
color = (0, 0, 0)


def convert_to_YCrCb(img):
    assert img.shape[2] == 3 # RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


def binarization(img):
    # get the size of the image
    # rows, cols, _ = img.shape
    # # convert to YCbCr
    # img_ycbcr = convert_to_YCbCr(img)
    # # get Y channel
    # Y = img_ycbcr[:, :, 0]
    # # get Cb channel
    # Cb = img_ycbcr[:, :, 1]
    # # get Cr channel
    # Cr = img_ycbcr[:, :, 2]
    
    # if Y < Kl:
    #   Wcb_y = WLcb + (WHcb - WLcb) * (Y - Ymin) / (Kl - Ymin)
    #   Wcr_y = WLcr + (WHcr - WLcr) * (Y - Ymin) / (Kl - Ymin)
    #   Cb_y_gach = 108 + (Kl - Y) * (118 - 108) / (Kl - Ymin)
    #   Cr_y_gach = 154 - (Kl - Y) * (154 - 144) / (Kl - Ymin)
    # elif Y > Kh:
    #   Wcb_y = WHcb + (Wcb - WHcb) * (Ymax - Y) / (Ymax - Kh)
    #   Wcr_y = WHcr + (Wcr - WHcr) * (Ymax - Y) / (Ymax - Kh)
    #   Cb_y_gach = 108 + (Y - Kh) * (118 - 108) / (Ymax - Kh)
    #   Cr_y_gach = 154 + (Y - Kh) * (154 - 132) / (Ymax - Kh)
    
    # if Y < Kl or Y > Kh:
    #   Cb_new = (Cb - Cb_y_gach) * Wcb / Wcb_y + Cb_y_gach
    
    # return img_binary

    # convert to YCbCr
    img_ycrcb = convert_to_YCrCb(img)

    img_bin = cv2.inRange(img_ycrcb, (100, 100, 100), (255, 255, 255))
    img_bin = cv2.bitwise_not(img_bin)
    # print(img_bin.shape)
    return img_bin
src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)

def morphological_process(img):    
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # # dilate
    # img_dilate = cv2.dilate(img, kernel, iterations=3)
    # # erode
    # img_erode = cv2.erode(img_dilate, kernel, iterations=3)
    # # morphology
    # img_morphology = cv2.morphologyEx(img_erode, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    kernel = np.ones((3, 3), np.uint8)
    img_morphology = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_morphology = cv2.morphologyEx(img_morphology, cv2.MORPH_CLOSE, kernel)
    img_morphology = cv2.bitwise_not(img_morphology)
    return img_morphology

def my_estimate_norm(lmk, image_size=112, mode='arcface'):
    
    print(lmk.shape)
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        if image_size == 112:
            src = arcface_src
        else:
            src = float(image_size) / 112 * arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index



def my_norm_crop(img, landmark, image_size=112, mode='arcface'):
    M, pose_index = my_estimate_norm(landmark, image_size, mode)
    print('M', M)
    print('pose_index', pose_index)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    print('warped', warped.shape)

    return warped

# DEFINE MODEL
detector = FaceAnalysis()
detector.prepare(ctx_id=0, det_size=(640, 640))


# READ IMAGE
img_1_in = cv2.imread(img_path)
# img_1_in = img_1_in[200:400, 350:550]

# img_1_in = cv2.imread('./face_emb/face 4 out.jpg')
# img_1_in = img_1_in[150:350, 550:750]

# print(img_1_in)
# crop image # y: 100-400, x: 300-600

# Get faces
face = detector.get(img_1_in)[0]
x1, y1, x2, y2 = face.bbox
x1 , y1 = int(x1), int(y1)
x2 , y2 = int(x2), int(y2)

img_1_in = img_1_in[y1:y2, x1:x2]
base_img = img_1_in.copy()
# Binarization
img_YCrCb_thresh = binarization(img_1_in)
# Morphological process
img_YCrCb_thresh = morphological_process(img_YCrCb_thresh)

# add padding to image
# img_padding = np.full((new_height, new_width), 0, dtype=np.uint8)
# print(img_padding.shape)
# # compute offset
# x_center = (new_width - img_YCrCb_thresh.shape[1]) // 2
# y_center = (new_height - img_YCrCb_thresh.shape[0]) // 2
# print("CENTER")
# print(x_center
#       , y_center)
# # copy image to padding
# img_padding[y_center:y_center + img_YCrCb_thresh.shape[0], x_center:x_center + img_YCrCb_thresh.shape[1]] = img_YCrCb_thresh

# img_padding_copy = img_padding.copy()



tmp = np.ones(img_YCrCb_thresh.shape, np.uint8) * 255

# FIND CONTOUR
# area contour of the image
if (int(cv2.__version__[0]) > 3):
  contours, hierarchy = cv2.findContours(img_YCrCb_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
else:
  im2, contours, hierarchy = cv2.findContours(img_YCrCb_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


# # check non zero contour
# extracted = np.zeros(img_YCrCb_thresh.shape, np.uint8)
# contoursSize = []
# for c in contours:
#     area = cv2.contourArea(c)
#     # I have modified these values to make it work for attached picture
#     if 10000 < area < 300000:
#         M = cv2.moments(c)
#         cx = int((M["m10"] / M["m00"]))
#         cy = int((M["m01"] / M["m00"]))
#         extracted.fill(0) 
#         cv2.drawContours(extracted, [c], 0, 255, cv2.FILLED)
#         width = cv2.countNonZero(extracted[cy][:])
#         height = cv2.countNonZero(extracted[:, cx])
#         contoursSize.append((width, height))
# print(contoursSize)


# find contour with max area
if len(contours) != 0:
  # draw in blue the contours that were founded
  cv2.drawContours(tmp, contours, -1, 255, 3)

  # find the biggest countour (c) by the area
  c = max(contours, key = cv2.contourArea)
  # print(c)
  x, y, w, h = cv2.boundingRect(c)

  # draw the biggest contour (c) in green
  cv2.rectangle(tmp,(x,y),(x+w,y+h),(0,255,0),2)

# output = cv2.bitwise_and(img_1_in, img_1_in, mask=img_YCrCb_thresh)

# find center img padding
x_center = (img_YCrCb_thresh.shape[1]) // 2
y_center = (img_YCrCb_thresh.shape[0]) // 2
print("CENTER")
print(x_center, y_center)
# eyes_region = img_padding[y_center - int(new_height / 2):y_center + int(new_height / 2), x_center - int(new_width / 2):x_center + int(new_width / 2)]
eyes_region = img_YCrCb_thresh[0:y_center, :]

# compute all pixel of eyes region
eyes_region_nonzero_pixel = np.count_nonzero(eyes_region)
eyes_region_zero_pixel = eyes_region.size - eyes_region_nonzero_pixel
print('eyes_region_nonzero_pixel', eyes_region_nonzero_pixel)
print('eyes_region_zero_pixel', eyes_region_zero_pixel)
print('eyes_region_size', eyes_region.size)
if eyes_region_nonzero_pixel < 1/2 * eyes_region_zero_pixel:
    indices_1 = np.where(img_YCrCb_thresh != 0)
    indices_0 = np.where(img_YCrCb_thresh == 0)
    img_YCrCb_thresh[indices_0] = 255
    img_YCrCb_thresh[indices_1] = 0

total_nonzero_pixel = np.count_nonzero(img_YCrCb_thresh)
total_zero_pixel = img_YCrCb_thresh.size - total_nonzero_pixel
print('total_nonzero_pixel', total_nonzero_pixel)
print('total_zero_pixel', total_zero_pixel)
print('total_size', img_YCrCb_thresh.size)
if total_nonzero_pixel < 1/2 * total_zero_pixel:
    print("Có khẩu trang")
else:
    print("Không có khẩu trang")


# SHOW IMAGE
plt.subplot(221)
plt.title("Original")
plt.imshow(base_img)
plt.subplot(222)
plt.title("Threshold")
plt.imshow(img_YCrCb_thresh, cmap='gray')
plt.subplot(223)
plt.title("Contour")
plt.imshow(img_YCrCb_thresh, cmap='gray')
plt.subplot(224)
plt.title("Output")
plt.imshow(eyes_region, cmap='gray')
plt.show()
# cv2.imshow('image', im_th)
# cv2.waitKey(0)
# cv2.destroyAllWindows()