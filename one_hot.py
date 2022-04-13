# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 16:30
# @Author  : 이합
# @FileName: one_hot.py
# @Software: PyCharm

from sklearn import preprocessing

encoder = preprocessing.OneHotEncoder()


encoder.fit([
    [0, 2, 7, 1],
    [1, 3, 5, 3],
    [3, 3, 1, 5],
    [1, 2, 4, 5]
])

# 表示有4个特征(看得是每列的值）
# 第一个特征(即：第一列—)为[0,1,3,1]
# 第一个特征有三类特征值[0,1,3]：One-Hot Encoding后采用三个编码：[100,010,001]
# 同理第二个特征列可将两类特征值[2,3]表示为[10,01]
# 第三个特征将4类特征值[1,4,5,7]表示为[1000,0100,0010,0001]
# 第四个特征将3类特征值[1,3,5]表示为[100,010,001]

encoded_vector = encoder.transform([[3, 2, 7, 5]]).toarray()
print("\n Encoded vector =", encoded_vector)

# [[0. 0. 1.    1. 0.      0. 0. 0. 1.     0. 0. 1.]]
