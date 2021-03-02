# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Convolution2D as Conv2D
from tensorflow.keras.layers import MaxPooling2D
from collections import Counter
from sklearn.metrics import confusion_matrix
'''获取数据集并初始化'''


def getdata():
    '''步骤一：下载手写数字数据集，进行初步的数据可视化和统计'''
    # 下载数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 可视化训练集的前9张图片
    plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(331 + i)
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.savefig('.\\ans_img\\sample.png', bbox_inches='tight', dpi=300)

    # 输出数据信息
    print(f'训练集的样本数：{x_train.shape[0]}，测试集的样本数：{x_test.shape[0]}')
    print(f'输入图像的大小：{x_train.shape[1]}*{x_train.shape[2]}')

    # 统计
    label_cnt = Counter(y_train)
    print('训练集的图像类别分布：', label_cnt)
    plt.figure(figsize=(5, 5))
    plt.pie(x=label_cnt.values(), labels=label_cnt.keys(), autopct='%.2f%%')
    plt.savefig('.\\ans_img\\label_distribution.png',
                bbox_inches='tight',
                dpi=300)
    '''步骤二：数据预处理'''
    # 获取总类别
    num_class = len(label_cnt)

    # 格式转换
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # 规范化，将像素值缩至0-1之间
    x_train /= 255
    x_test /= 255

    # 将标签向量转化为one-hot形式的向量
    y_train = tf.keras.utils.to_categorical(y_train, num_class)
    y_test = tf.keras.utils.to_categorical(y_test, num_class)

    return x_train, y_train, x_test, y_test, num_class
