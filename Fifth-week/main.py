import matplotlib.pyplot as plt
import init_utils
from regularization import *
import reg_utils

# plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
train_x, train_y, test_x, test_y = reg_utils.load_2D_dataset(is_plot=False)
# # plt.show()
NN = NeuralNetwork(train_x, train_y, learning_rate=0.5, num_iterations=40000, lambd=0, dropout_rate=0.8,
                   layers_dim=[train_x.shape[0], 20, 3, train_y.shape[0]])
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
plt.title(f"l_rate: {NN.learning_rate},train:{accuracy_train}%, test:{accuracy_test}%\n{NN.layers_dim}")
plt.show()
init_utils.plot_decision_boundary(lambda x: NN.predict(x.T), train_x, np.squeeze(train_y))
