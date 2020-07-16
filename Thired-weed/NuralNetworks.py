import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from testCases import *
from ThreeLayerNN import ThreeLayerNeuralNetwork

parameters, X_assess = predict_test_case()
np.random.seed(1)  # 设置一个固定的随机种子，以保证接下来的步骤中我们的结果是一致的。
X, Y = load_planar_dataset()
NN = ThreeLayerNeuralNetwork(X, Y, 0.01, 1200)
NN.learning()
result = NN.predict(X)
print(Y)
print(result)
count = 0
print(Y.shape)
for i in range(Y.shape[1]):
    if Y[0][i] == result[0][i]:
        count = count + 1
print(count / result.shape[1])
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
