# -*- coding: utf-8 -*-
# @Time    : 2022/4/5 11:29
# @Author  : 이합
# @FileName: task.py
# @Software: PyCharm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow import keras
import data_reader

# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 5  # 예제 기본값은 20입니다.

# 데이터를 읽어옵니다.
dr = data_reader.DataReader()

# 인공신경망을 제작합니다.
model = keras.Sequential([
    keras.layers.Dense(2),                                # Training feature 의 개수를 2개로 줄여서 실험
    keras.layers.Dense(200, activation="relu"),           # Layer 개수 or neuron 개수 변경
    keras.layers.Dense(200, activation="sigmoid"),        # Activation function 변경(relu -> sigmoid)
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(3, activation='softmax')
])

# 인공신경망을 컴파일합니다.
model.compile(optimizer="adam", metrics=["accuracy"],
              loss="sparse_categorical_crossentropy")

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(dr.train_X[0: 100:, :], dr.train_Y[0: 100], epochs=EPOCHS, batch_size=5,
                    # Training data 의 개수를 줄여서 실험
                    validation_data=(dr.test_X, dr.test_Y),
                    callbacks=[early_stop])

# 학습 결과를 그래프로 출력합니다.
data_reader.draw_graph(history)



"""

print(dr.test_X.shape)    # (30, 4)
print(dr.test_Y.shape)    # (30,)
print(dr.train_X.shape)   # (120, 4)
print(dr.train_Y.shape)   # (120,)
print(dr.train_X[0: 50:, :].shape)    # (50, 4)     Training data 의 개수를 50개로 실험
print(dr.train_X[:, 0: 2:].shape)     # (120, 2)    Training feature 의 개수를 2개로 줄여서 실험
print(dr.train_Y[0: 50].shape)        # (50,)
print(dr.train_X[0: 50:, 2:].shape)   # (50,2)      Training data 의 개수를 50개로 실험,
                                                # /Training feature 의 개수를 2개로 줄여서 실험
print(dr.train_X[0: 50:, :], dr.train_X[:, 0: 2:], dr.train_Y[0:50].shape)      # 결과



# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS, batch_size=5,
                    validation_data=(dr.test_X, dr.test_Y),
                    callbacks=[early_stop])

# 학습 결과를 그래프로 출력합니다.
data_reader.draw_graph(history)
"""