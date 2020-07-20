from lr_utils import load_dataset
from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y
# print(train_x.shape)
# print(train_y.shape)
NN = NeuralNetwork(train_x, train_y, learning_rate=0.005, num_iterations=2600,
                   layers_dim=[test_x.shape[0], 10, test_y.shape[0]])

NN.learning()
result_test = NN.predict(test_x)
accuracy_test = float(
    (np.dot(test_y, result_test.T) + np.dot(1 - test_y, 1 - result_test.T)) / float(test_y.size) * 100)
accuracy_test = np.round(accuracy_test, 2)
result_train = NN.predict(train_x)
accuracy_train = float(
    (np.dot(train_y, result_train.T) + np.dot(1 - train_y, 1 - result_train.T)) / float(train_y.size) * 100)
accuracy_train = np.round(accuracy_train, 2)

print(f"训练集预测准确率：{accuracy_train}%\n测试集预测准确率：{accuracy_test}%")
plt.plot(NN.lost)
plt.title(f"h_size:{NN.hidden_size},l_rate: {NN.learning_rate},train:{accuracy_train}%, test:{accuracy_test}%")
plt.show()
