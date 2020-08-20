# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.io
# import math
# import sklearn
# import sklearn.datasets
#
# from miniAdam import NeuralNetwork  # 参见数据包或者在本文底部copy
# import testCase  # 参见数据包或者在本文底部copy
#
# # %matplotlib inline #如果你用的是Jupyter Notebook请取消注释
# plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
# X_assess,Y_assess,mini_batch_size = testCase.random_mini_batches_test_case()
# plt.plot(X_assess,Y_assess)
# plt.show()
# NN = NeuralNetwork()


import numpy as np
import matplotlib.pyplot as plt
from miniAdam import *

train_fo = open("datasets/zhengqi_train.txt", "r")
test_fo = open("datasets/zhengqi_test.txt", "r")

test_fo.readline()
train_fo.readline()

test = test_fo.readlines()
train = train_fo.readlines()

train_data = []
train_value = []

test_data = []
test_value = []

result_data = []

for i in range(0, int(len(train) * 0.8)):
    line = train[i].split('\t')
    temp = line.pop().split('\n')
    train_value.append(temp[0])
    train_data.append(line)
train_value = np.mat(train_value)
train_value = train_value.astype(float)
train_data = np.mat(train_data).T
train_data = train_data.astype(float)
train_value = (train_value + 3.1) / 6
train_data = train_data / 13

for i in range(int(len(train) * 0.8) + 1, len(train)):
    line = train[i].split('\t')
    temp = line.pop().split('\n')
    test_value.append(temp[0])
    test_data.append(line)
test_value = np.mat(test_value)
test_value = test_value.astype(float)
test_data = np.mat(test_data).T
test_data = test_data.astype(float)
test_value = (test_value + 3.1) / 6
test_data = test_data / 13

for i in range(0, len(test)):
    line = test[i].split('\t')
    temp = line[37].split('\n')
    line[37] = temp[0]
    result_data.append(line)

result_data = np.mat(result_data).T
result_data = result_data.astype(float)
result_data = result_data / 13

test_fo.close()
test_fo.close()
#print(train_data[0])
data = np.array_split(train_data, 38, 0)
d1 = data[0].tolist()
d2 = train_value.tolist()

#print(data[0].shape)
plt.plot((d1, d2))
plt.show()

# NN = NeuralNetwork(train_data, train_value, learning_rate=0.01, num_iterations=6500, lambd=0.3, dropout_rate=1,
#                    layers_dim=[train_data.shape[0], 10, 5, train_value.shape[0]],mini_batch_size=300)
#
# # print(train_value.shape)
#
# NN.learning(test_data, test_value)
# result = NN.predict(test_data)
# SSE = NN.MSE(result, test_value)
# print(f"测试集均方差：{SSE}")
# plt.plot(NN.test_SSE[(len(NN.test_SSE) - 1000):])
# # my_y_ticks = np.arange(0, 0.03, 0.005)
# #
# # plt.yticks(my_y_ticks)
# plt.show()
# np.savetxt("results.txt", (NN.predict(result_data) * 6) - 3.1, delimiter='\n', fmt='%0.8f')
