# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Convolution2D as Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import load_model
from collections import Counter
from sklearn.metrics import confusion_matrix
import data
import mydata

# 数据获取
x_train, y_train, x_test, y_test, num_class = data.getdata()

# 加载模型
model = load_model(".\\model\\parameter.h5")
'''步骤五：评估模型'''
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'测试集损失值: {test_loss}, 测试集准确率: {test_acc}')

# 获取自己写的数字
(my_x_test, my_y_test) = mydata.getmydata()
print(0)
my_y_test = tf.keras.utils.to_categorical(my_y_test, num_class)

# 评估模型
print(1)
my_test_loss, my_test_acc = model.evaluate(my_x_test, my_y_test, verbose=0)
print(f'我的测试集损失值: {my_test_loss}, 我的测试集准确率: {my_test_acc}')

# 输出混淆矩阵
print(f"Mnist手写数据集混淆矩阵如下：")
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)

print(f"我的数据集混淆矩阵如下：")
y_pred = model.predict(my_x_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(my_y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)
