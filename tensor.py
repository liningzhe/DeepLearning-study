# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 22:24
# @Author  : 이합
# @FileName: tensor.py
# @Software: PyCharm

# 根据教材：动手学深度学习整理；
# 课程的直播地址：http://courses.d2l.ai/zh-v2/
# 课程的课件地址：https://zh-v2.d2l.ai/


import torch
# print(dir(torch.distributions))

"""
# 节1：
# ones 函数创建一个具有指定形状的新张量，并将所有元素值设置为 1
t = torch.ones(4)
print('t:', t)   # t: tensor([1., 1., 1., 1.])

# 使⽤arange创建⼀个⾏向量 x, ⾏向量包含以0开始的前12个整数,。张量中的每个值都称为张量的元素（element）
x = torch.arange(12)
print('x:', x)   # x: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# 通过张量的shape属性来访问张量（沿每个轴的⻓度）的形状;
print(x.shape)    # torch.Size([12])
print(x.numel())  # 返回张量中元素的总个数 # 12

# 改变一个张量的形状而不改变元素数量和元素值
y = x.reshape(3, 4)  # 把张量x从形状为（12,）的⾏向量转换为形状为（3,4）的矩阵
y = x.reshape(-1, 4)  # 通过-1来调⽤此⾃动计算出维度的功能
y = x.reshape(3, -1)  # 通过-1来调⽤此⾃动计算出维度的功能
print('y:', y)
"""

"""
# 节2： 
# 创建一个张量，其中所有元素都设置为0
z = torch.zeros(2, 3, 4)
print('z:', z)

# 每个元素都从均值为0、标准差为1的标准高斯（正态）分布中随机采样;
w = torch.randn(2, 3, 4)
print('w:', w)

# 通过提供包含数值的 Python 列表（或嵌套列表）来为所需张量中的每个元素赋予确定值
q = torch.tensor([[1, 2, 3], [4, 3, 2], [7, 4, 3]])
print('q:', q)

# 运算符/张量的运算
x = torch.tensor([1.0, 2, 4, 8]) #1.0 是为了创建浮点数;
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y, torch.exp(x))

"""

"""
# 节3：
# 把多个张量连结(concatenate)，把它们端对端地叠起来形成⼀个更⼤的张量;
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# dim=0表示二维中的行, 此处表示沿⾏（轴-0，形状的第⼀个元素）
print('cat操作 dim=0', torch.cat((X, Y), dim=0))  # 输出张量的轴-0⻓度（6）是两个输⼊张量轴-0⻓度的总和（3 + 3）；
# dim=1在二维矩阵中表示列，此处表示按列（轴-1，形状的第⼆个元素）
print('cat操作 dim=1', torch.cat((X, Y), dim=1))  # 输出张量的轴-1⻓度（8）是两个输⼊张量轴-1⻓度的总和（4 + 4）；

# 通过 逻辑运算符 构建二元张量
print('X == Y', X == Y)
print('X < Y', X < Y)
print('张量所有元素的和:', X.sum())
"""

"""
# 节4：
# ⼴播机制(broadcasting mechanism): 在不同形状的张量上执⾏按元素操作；
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print('a:', a)
print('b:', b)
print('a + b:', a + b)
"""

"""
# 节5：
# 转换为其他 Python对象,比如numpy张量;
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
A = Y.numpy()
print(type(A))    # 打印A的类型 <class 'numpy.ndarray'>
print(A)
B = torch.tensor(A)
print(type(B))    # 打印B的类型 <class 'torch.Tensor'>
print(B)

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))
"""

"""
# 节6：
# 节约内存
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
before = id(Y)    # id()函数提供了内存中引用对象的确切地址
Y = Y + X
print(id(Y) == before)   # False

# 使用 X[:] = X + Y 或 X += Y 来减少操作的内存开销;
before = id(X)
X += Y
print(id(X) == before)  # True

# 使用 X[:] = X + Y 或 X += Y 来减少操作的内存开销;
before = id(X)
X[:] = X + Y
print(id(X) == before)  # True
"""

# 节7：
# 访问元素/索引和切片
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print('X:', X)
print(X[1, 2])   # tensor(6.)
print(X[1, :])   # 访问一行：tensor([4., 5., 6., 7.])
print(X[:, 1])   # 访问一列：tensor([1., 5., 9.])
print(X[1:3, 1:])   # 访问子区域：访问第一行和第二行[1:3,左开右闭；1:]表示第一列开始的所有列；
print(X[::2, ::2])  # 访问子区域：begin:end:step，begin默认为0,end默认值为列尾,step默认为1，即0:size:1
print(X[-1])     # 最后一行 tensor([ 8.,  9., 10., 11.])
print(X[1:3])    # 左开右闭，第一行开始，第二行结束，注意是从第0行数的；

X[1, 2] = 9  # 写入元素: 原先这里为6.,现改为9.
print('X:', X)
X[0:2, :] = 12  # 写入元素: [0:2, :]访问第1⾏和第2⾏，其中“:”代表沿轴1（列）的所有元素(所有列)
print('X:', X)
