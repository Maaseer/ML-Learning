import numpy as np
from L2 import *

train_fo = open("datasets/zhengqi_train.txt", "r")
test_fo = open("datasets/zhengqi_test.txt", "r")

test_fo.readline()
train_fo.readline()

test = test_fo.readlines()
train = train_fo.readlines()
train_data = []
train_value = []
test_data = []
for i in range(0, len(train)):
    line = train[i].split('\t')
    temp = line.pop().split('\n')
    train_value.append(temp[0])
    train_data.append(line)
train_value = np.mat(train_value)
train_value = train_value.astype(float)
train_data = np.mat(train_data).T
train_data = train_data.astype(float)
for i in range(0, len(test)):
    line = test[i].split('\t')
    temp = line[37].split('\n')
    line[37] = temp[0]
    test_data.append(line)

test_data = np.mat(test_data).T
test_data = test_data.astype(float)

test_fo.close()
test_fo.close()

train_value = (train_value + 3.1) / 6
train_data = train_data / 13
NN = NeuralNetwork(train_data, train_value, learning_rate=0.05, num_iterations=10000,
                   layers_dim=[train_data.shape[0], 10, 5, train_value.shape[0]])
# print(train_value.shape)
NN.learning()
