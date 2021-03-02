# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Convolution2D as Conv2D
from tensorflow.keras.layers import MaxPooling2D
from collections import Counter
from sklearn.metrics import confusion_matrix
import data

# 数据获取
x_train, y_train, x_test, y_test, num_class = data.getdata()
'''步骤三：模型搭建'''
# 序列化模型
model = tf.keras.models.Sequential()

# LeNet
model.add(
    tf.keras.layers.Conv2D(input_shape=(x_train.shape[1], x_train.shape[2],
                                        x_train.shape[3]),
                           filters=8,
                           kernel_size=(5, 5),
                           strides=(1, 1),
                           padding='same',
                           activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(
    tf.keras.layers.Conv2D(filters=20,
                           kernel_size=(5, 5),
                           strides=(1, 1),
                           padding='valid',
                           activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(
    tf.keras.layers.Conv2D(filters=120,
                           kernel_size=(5, 5),
                           strides=(1, 1),
                           padding='valid',
                           activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 查看模型架构
model.summary()

# 定义模型训练细节，包括交叉熵损失函数，Adam优化器和准确率评价指标
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
'''步骤四：训练模型'''
h = model.fit(x_train, y_train, validation_split=0.2, batch_size=128, epochs=5)

# 保存模型
model.save(".\\saved_model\\parameter.h5")

# 训练历史
print(h.history.keys())
accuracy = h.history['accuracy']
val_accuracy = h.history['val_accuracy']
loss = h.history['loss']
val_loss = h.history['val_loss']
epochs = range(len(accuracy))

# 输出训练准确率与损失率
plt.figure()
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'm.', label='validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('.\\ans_img\\accuracy.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'm.', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('.\\ans_img\\loss.png', bbox_inches='tight', dpi=300)
