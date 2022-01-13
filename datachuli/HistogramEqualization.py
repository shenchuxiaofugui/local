import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import imageio


# 将灰度数组映射为直方图字典,nums表示灰度的数量级
def arrayToHist(grayArray, nums):
    if len(grayArray.shape) != 2:
        print("length error")
        return None
    w, h = grayArray.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            if hist.get(grayArray[i][j]) is None:
                hist[grayArray[i][j]] = 0
            hist[grayArray[i][j]] += 1
    # normalize
    n = w*h
    for key in hist.keys():
        hist[key] = float(hist[key])/n
    return hist


# 计算累计直方图计算出新的均衡化的图片，nums为灰度数,256
def equalization(grayArray,h_s,nums):
    # 计算累计直方图
    tmp = 0.0
    h_acc = h_s.copy()
    for i in range(256):
        tmp += h_s[i]
        h_acc[i] = tmp
    if len(grayArray.shape) != 2:
        print("length error")
        return None
    w, h = grayArray.shape
    des = np.zeros((w, h), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            des[i][j] = int((nums - 1) * h_acc[grayArray[i][j]] + 0.5)
    return des


# 传入的直方图要求是个字典，每个灰度对应着概率
def drawHist(hist, name):
    keys = hist.keys()
    values = hist.values()
    x_size = len(hist)-1    # x轴长度，也就是灰度级别
    axis_params = []
    axis_params.append(0)
    axis_params.append(x_size)

    # plt.figure()
    if name != None:
        plt.title(name)
    plt.bar(tuple(keys),tuple(values))  # 绘制直方图
    # plt.show()


def main():
    matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文

    # imdir = r"D:\test\DIP\raw\DIP3E_Original_Images_CH03\Fig0316(4)(bottom_left).tif"   # 原始图片的路径
    imdir = r"D:\test\DIP\raw\DIP3E_Original_Images_CH03\Fig0323(a)(mars_moon_phobos).tif"   # 原始图片的路径

    # 打开文件并灰度化
    im_s = imageio.mimread(imdir)
    im_s = np.array(im_s)
    im_s = np.squeeze(im_s)
    print(np.shape(im_s))

    # 开始绘图，分成四个部分
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(im_s, cmap='gray')
    plt.title("原始灰度图")

    # 创建原始直方图
    plt.subplot(2,2,3)
    hist_s = arrayToHist(im_s, 256)
    drawHist(hist_s, "原始直方图")
    # plt.hist(im_s.flatten(), bins=256)

    # 计算均衡化的新的图片，根据累计直方图
    im_d = equalization(im_s,hist_s,256)
    plt.subplot(2,2,2)
    plt.imshow(im_d,cmap="gray")
    plt.title("均衡的灰度图")

    # 根据新的图片的数组，计算新的直方图
    plt.subplot(2,2,4)
    hist_d = arrayToHist(im_d,256)
    drawHist(hist_d,"均衡直方图")

    plt.show()


if __name__ == '__main__':
    main()

