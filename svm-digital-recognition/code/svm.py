# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import _pickle as pickle
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import data

# 数据获取
x_train, y_train, x_test, y_test=data.getdata()

'''步骤三：模型搭建'''
# # 最优超参数组合列表
# params = [
#     {'kernel': ['linear'], 'C': [1, 10, 100, 100]},
#     {'kernel': ['poly'], 'C': [1], 'degree': [2, 3]},
#     {'kernel': ['rbf'], 'C': [1, 10, 100, 100], 'gamma':[1, 0.1, 0.01, 0.001]}
# ]

# # 自动化调参
# model = ms.GridSearchCV(svm.SVC(probability=True), params, refit=True, return_train_score=True,  cv=5)

model = svm.SVC(C=10.0, kernel='rbf', gamma=0.01)

'''步骤四：训练模型'''
h = model.fit(x_train, y_train)

# 保存模型
with open('.\\saved_model\\paramater.pkl','wb') as file:
    pickle.dump(model, file)

# 输出混淆矩阵
print(f"Mnist手写数据集混淆矩阵如下：")
predictions = [int(a) for a in model.predict(x_test)]
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, np.array(predictions)))

# 计算准确度
print('accuracy=', accuracy_score(y_test, predictions))