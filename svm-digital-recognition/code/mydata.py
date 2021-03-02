# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import morphology

'''获取边界'''
def getbound(img):
    img_binary=img.convert('1')
    img_array = np.array(img_binary)
    shape = img_array.shape
    sumx = shape[0] - np.sum(img_array, axis=0)/255
    sumy = shape[1] - np.sum(img_array, axis=1)/255
    l, r, h, b = 0, 0, 0, 0
    for i in np.arange(shape[1]):
        if sumx[i] >= 3:
            l = i
            break
    for i in range(shape[1]-1, -1, -1):
        if sumx[i] >= 3:
            r = i
            break
    for i in np.arange(shape[0]):
        if sumy[i] >= 3:
            h = i
            break
    for i in range(shape[0]-1, -1, -1):
        if sumy[i] >= 3:
            b = i
            break
    return l, r, h, b

'''数据预处理'''
def pretreat(filename):
    img = (Image.open(filename).convert('L'))
    l, r, h, b = getbound(img)
    shape = np.array(img).shape
    dx, dy = (r - l)//2, (b - h)//2

    # 获取中心点
    centerx, centery = l+dx, h+dy

    # 改为正方形
    dx = max(dx, dy)
    dy = dx

    # 得到轮廓
    l, r, h, b = max(0, centery-dy*1.2), min(shape[0], centery+dy*1.2), max(0, centerx-dx*1.2), min(shape[1], centerx+dx*1.2)

    # 切片
    img_array = np.array(img)
    the_img = img_array[l:r, h:b]

    # 先膨胀后腐蚀，消除手写字内部可能出现的小泡
    the_img=morphology.closing(the_img, selem=None, out=None)

    # 转灰度
    the_img=Image.fromarray(cv2.cvtColor(the_img, cv2.COLOR_BGR2RGB))
    img_gray=cv2.cvtColor(np.array(the_img),cv2.COLOR_RGB2GRAY)

    # 高斯滤波
    img_gauss = cv2.GaussianBlur(img_gray,(5,5),0)

    # 二值化，反色
    ret,img_binary=cv2.threshold(img_gauss, 0, 1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # 骨架提取
    img_binary=morphology.skeletonize(img_binary)

    # 膨胀
    img_binary=morphology.binary_dilation(img_binary, morphology.disk(29))

    # 调整大小为(28, 28)
    img_binary=Image.fromarray(img_binary)
    img_binary=img_binary.resize((28,28),Image.ANTIALIAS)
    
    # 为1的点转为255
    img_array=np.array(img_binary).astype(np.uint8)
    img_array[img_array==1]=255

    # 数据格式规范
    img_array.reshape((-1,28,28,1))
    img_array=img_array[:,:,np.newaxis]

    return img_array

'''获取数据'''
def getmydata():
    # 获取数据
    my_x_test, my_y_test=[], []
    for i in range(10):
        my_x_test.append(pretreat('./my_img/'+str(i)+'.jpg'))
        my_y_test.append(i)
        plt.subplot(4, 3, 1 + i)
        plt.imshow(my_x_test[i], cmap=plt.get_cmap('gray'))
    plt.savefig('.\\ans_img\\after_pretreat.png', bbox_inches='tight', dpi=300)

    # 数据调整
    my_x_test=np.array(my_x_test)
    my_y_test=np.array(my_y_test)
    my_x_test = my_x_test.reshape(10, 784).astype('float32')
    my_x_test/=255
    
    return my_x_test, my_y_test
