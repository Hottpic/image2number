import numpy as np
import cv2
from matplotlib import pyplot as plt
import pyocr
import pyocr.builders
import sys
from PIL import Image

MIN_MATCH_COUNT = 10

img1 = cv2.imread('kanri.png', 0)  # queryImage
img2 = cv2.imread('test_orig.jpg', 0)  # trainImage

# AKAZE検出器の生成
akaze = cv2.AKAZE_create()

# 特徴点の検出
kp1 = akaze.detect(img1, None)
kp2 = akaze.detect(img2, None)

# 特徴量の計算と記述
kp1, des1 = akaze.compute(img1, kp1)
kp2, des2 = akaze.compute(img2, kp2)

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=50)

# ANNで近傍２位までを出力
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    M2 = cv2.getPerspectiveTransform(dst, pts)
    img_warp = cv2.warpPerspective(img2, M2, (w, h))

    img_number = img_warp[30:110, 370:810]
    plt.imshow(img_number, 'gray'), plt.show()
    #
    # img2 = cv2.polylines(img2, [np.int32(dst)], True, 0, 3, cv2.LINE_AA)

#     OCR
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)

    tool = tools[0]

    box = [370, 30, 440, 80]

    # イメージは OpenCV -> PIL に変換する
    txt = tool.image_to_string(Image.fromarray(img_warp[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]), lang="eng",
                               builder=pyocr.builders.TextBuilder(tesseract_layout=7))


    print(txt)

else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None

# draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
#                    singlePointColor=None,
#                    matchesMask=matchesMask,  # draw only inliers
#                    flags=2)
#
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
#
# plt.imshow(img3, 'gray'), plt.show()
