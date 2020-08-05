import numpy as np
import matplotlib.pyplot as plt
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


NN = NeuralNetwork(train_data, train_value, learning_rate=0.05, num_iterations=41000,lambd=0.15,
                   layers_dim=[train_data.shape[0], 10, 5, train_value.shape[0]])
# print(train_value.shape)
NN.learning(test_data,test_value)
result = NN.predict(test_data)
SSE = NN.MSE(result, test_value)
print(f"测试集均方差：{SSE}")
plt.plot(NN.test_SSE)
plt.show()
np.savetxt("results.txt",(NN.predict(result_data) * 6) - 3.1,delimiter='\n',fmt='%0.8f')