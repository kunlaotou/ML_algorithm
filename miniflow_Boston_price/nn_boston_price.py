# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 08:46:16 2017

@author: Administrator
"""

"""
Check out the new network architecture and dataset!

Notice that the weights and biases are
generated randomly.

No need to change anything, but feel free to tweak
to test your network, play around with the epochs, batch size, etc!
"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from miniflow_Boston_price import *

# Load data
data = load_boston()

X_ = data['data']   #(506,13)
y_ = data['target'] #(506,)

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

#获取数据集的列数，即特征个数，并作为W1的行数
n_features = X_.shape[1] 
n_hidden = 10

W1_ = np.random.randn(n_features, n_hidden)  #(13,10) 横轴是特征数(输入个数)，纵轴是隐藏层节点数
b1_ = np.zeros(n_hidden)     #有几个隐藏节点，b的大小就是几 这里b = 10 

W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)


# Neural network
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(y, l2)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}

epochs = 10
# Total number of examples
m = X_.shape[0]
batch_size = 11
steps_per_epoch = m // batch_size  #浮点数除法，其结果进行四舍五入。

graph = topological_sort(feed_dict)
trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

# Step 4  重复第 1-3 步，直到出现收敛情况或者循环被其他机制暂停（即迭代次数）。
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1 从总的数据集中随机抽样一批数据
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # Step 2 前向和后向运行网络，计算梯度（根据第 (1) 步的数据）。
        forward_and_backward(graph)

        # Step 3 应用梯度下降更新。
        sgd_update(trainables)

        loss += graph[-1].value

    print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))
