import matplotlib.pyplot as plt

from TwoLayerNN import TwoLayerNeuralNetwork
from planar_utils import plot_decision_boundary, load_planar_dataset
from testCases import *

# 加载数据
X, Y = load_planar_dataset()
# 实例化神经网络对象
NN = TwoLayerNeuralNetwork(X, Y, hidden_size=5, learning_rate=0.5, num_iterations=20000)
# 训练
NN.learning()
# 预测
result = NN.predict(X)

# 统计正确率
accuracy = float((np.dot(Y, result.T) + np.dot(1 - Y, 1 - result.T)) / float(Y.size) * 100)
print(f"预测准确率： {accuracy}%")

# 画出lost曲线
title = f"lost learning rate = {NN.learning_rate}, hidden size = {NN.hidden_size}, Accuracy : {accuracy}%"
plt.title(title)
plt.xlabel("num_Iterate")
plt.ylabel("lost")
plt.plot(NN.lost)
plt.show()

# 画出决策边界
plot_decision_boundary(lambda x: NN.predict(x.T), X, np.squeeze(Y))
plt.show()
