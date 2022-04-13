# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 23:31
# @Author  : 이합
# @FileName: task.py
# @Software: PyCharm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 不显示等级1以下信息；、log信息四个等级：INFO=0<WARNING=1<ERROR=2<FATAL=3
from tensorflow import keras  # 如果放在上面，黄色提示消失；但是os.environ没有效果；
import data_reader


EPOCHS = 5  

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

# 인공신경망을 제작합니다.
model = keras.Sequential([
    keras.layers.Dense(3),
    keras.layers.Dense(500, activation="softmax"),
    keras.layers.Dense(500, activation="softmax"),
    keras.layers.Dense(3, activation='tanh')
])

# 인공신경망을 컴파일합니다.
model.compile(optimizer="sgd", metrics=["accuracy"],
              loss="sparse_categorical_crossentropy")

# 인공신경망을 학습시킵니다.
print("************ TRAINING START ************")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS,
                    validation_data=(dr.test_X, dr.test_Y),
                    callbacks=[early_stop])

# 학습 결과를 그래프로 출력합니다.
data_reader.draw_graph(history)
