import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import imageio
from HistogramEqualization import *
import cv2


# 直方图匹配函数，接受原始图像和目标灰度直方图
def histMatch(grayArray, h_d):
    # 计算累计直方图
    tmp = 0.0
    h_acc = h_d.copy()
    for i in range(256):
        tmp += h_d[i]
        h_acc[i] = tmp

    h1 = arrayToHist(grayArray,256)
    tmp = 0.0
    h1_acc = h1.copy()
    for i in range(256):
        tmp += h1[i]
        h1_acc[i] = tmp
    # 计算映射
    M = np.zeros(256)
    for i in range(256):
        idx = 0
        minv = 1
        for j in h_acc:
            if np.fabs(h_acc[j] - h1_acc[i]) < minv:
                minv = np.fabs(h_acc[j] - h1_acc[i])
                idx = int(j)
        M[i] = idx
    des = M[grayArray]
    return des


def main():
    matplotlib.rcParams['font.sans-serif']=['SimHei']
    imdir = r"D:\test\DIP\raw\DIP3E_Original_Images_CH03\Fig0323(a)(mars_moon_phobos).tif"
    imdir_match = r'D:\test\DIP\raw\DIP3E_Original_Images_CH03\Fig0320(3)(third_from_top).tif'

    # 直方图匹配
    # 打开文件并灰度化
    im_s = imageio.mimread(imdir)
    im_s = np.array(im_s)
    im_s = np.squeeze(im_s)
    print(np.shape(im_s))
    # 打开文件并灰度化
    im_match = imageio.mimread(imdir_match)
    im_match = np.array(im_match)
    im_match = np.squeeze(im_match)
    print(np.shape(im_match))
    # 开始绘图
    plt.figure()

    # 原始图和直方图
    plt.subplot(2, 3, 1)
    plt.title("原始图片")
    plt.imshow(im_s,cmap='gray')

    plt.subplot(2,3,4)
    hist_s = arrayToHist(im_s,256)
    drawHist(hist_s,"原始直方图")

    # match图和其直方图
    plt.subplot(2,3,2)
    plt.title("match图片")
    plt.imshow(im_match,cmap='gray')

    plt.subplot(2,3,5)
    hist_m = arrayToHist(im_match,256)
    drawHist(hist_m,"match直方图")

    # match后的图片及其直方图
    im_d = histMatch(im_s,hist_m)#将目标图的直方图用于给原图做均衡，也就实现了match
    plt.subplot(2,3,3)
    plt.title("match后的图片")
    plt.imshow(im_d,cmap='gray')

    plt.subplot(2,3,6)
    hist_d = arrayToHist(im_d,256)
    drawHist(hist_d,"match后的直方图")

    plt.show()


if __name__ == '__main__':
    main()

