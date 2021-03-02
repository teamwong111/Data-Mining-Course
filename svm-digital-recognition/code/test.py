# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import _pickle as pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import data
import mydata

'''步骤五：评估模型'''
# 加载模型
with open('.\\saved_model\\paramater.pkl','rb') as file:
    model = pickle.load(file)

# 获取自己写的数字
(my_x_test,my_y_test) = mydata.getmydata()

# 评估模型
# 输出混淆矩阵
print(f"我的数据集混淆矩阵如下：")
predictions = [int(a) for a in model.predict(my_x_test)]
print(confusion_matrix(my_y_test, predictions))
print(classification_report(my_y_test, np.array(predictions), labels=np.unique(predictions)))
#计算准确度
print('accuracy=', accuracy_score(my_y_test, predictions))