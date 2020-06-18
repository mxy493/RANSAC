# -*- coding: UTF-8 -*-
# author: mxy

import numpy
import random
import cv2
import math


def ransac(pts1, pts2):
    best_inlinenums = 0
    best_f = numpy.zeros([3, 3])
    best_distance = []
    i = 0
    while i < 100:
        # 随机选择8个点
        index = set()
        while len(index) < 8:
            index.add(random.randrange(len(pts1)))
        # 根据这8个点生成矩阵
        a = numpy.zeros([8, 9])
        for j, item in enumerate(index):
            (x1, y1) = pts1[item]
            (x2, y2) = pts2[item]
            a[j][0] = x1 * x2
            a[j][1] = x2 * y1
            a[j][2] = x2
            a[j][3] = x1 * y2
            a[j][4] = y1 * y2
            a[j][5] = y2
            a[j][6] = x1
            a[j][7] = y1
            a[j][8] = 1
        u, d, vt = numpy.linalg.svd(a)
        f = vt[8]
        f = f.reshape(3, 3)
        # 根据F计算内点数，首先计算极线
        one = numpy.ones(len(pts1))
        pts1_new = numpy.insert(pts1, 2, values=one, axis=1)  # 构造齐次坐标系
        elines = [numpy.dot(f, pts1_new[i]) for i in range(len(pts1))]  # 极线
        # 计算pts2中每一个点到对应极线的距离
        pts2_new = numpy.insert(pts2, 2, values=one, axis=1)  # 构造齐次坐标系
        distances = []
        inline_num = 0
        for pt, l in zip(pts2_new, elines):
            div = abs(numpy.dot(numpy.transpose(pt), l))
            dived = math.sqrt(l[0] * l[0] + l[1] * l[1])
            d = div / dived
            if d <= 3:
                inline_num = inline_num + 1
            distances.append(d)
        if inline_num > best_inlinenums:
            best_f = f[:]
            best_inlinenums = inline_num
            best_distance = distances[:]
        i += 1
    return best_f, best_distance


if __name__ == '__main__':
    # 读取两张图片
    img1 = cv2.imread('img/img1.jpg', 0)
    img2 = cv2.imread('img/img2.jpg', 0)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1  # kd树
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []  # 较好的匹配
    pts1 = []  # img1中较好的匹配的坐标
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    best_f, distances = ransac(pts1, pts2)
    matchesMask = [0 for i in range(len(good))]
    for i, k in enumerate(distances):
        if k <= 3:
            matchesMask[i] = 1
    img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, (255, 0, 0), (255, 0, 0), matchesMask, flags=2)
    matchesMask = [0 for i in range(len(good))]
    for i, k in enumerate(distances):
        if k > 3:
            matchesMask[i] = 1
    img = cv2.drawMatches(img1, kp1, img2, kp2, good, img, (0, 0, 255), (0, 0, 255), matchesMask=matchesMask,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG | cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("RANSAC", img)
    cv2.waitKey()
    cv2.destroyAllWindows()  # 清除所有窗口
