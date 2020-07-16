import numpy as np


class ThreeLayerNeuralNetwork:
    """
    各层节点数
    """
    input_size = 0
    hidden_size = 0
    output_size = 0

    """"
    各层权重矩阵
    """
    input_hidden_w = ()
    hidden_output_w = ()
    hidden_b = ()
    output_b = ()

    """
    训练数据
    """
    train_data = ()
    train_value = ()
    m = 0
    cache = {}
    """
    学习率
    """
    learning_rate = 0.01
    num_iterations = 1000

    """
    每次迭代的损失值
    """
    lost = []

    def __init__(self, train_data, train_value, learning_rate=0.01, num_iterations=1000):
        self.input_size, self.hidden_size, self.output_size = self.layer_sizes(train_data, train_value)

        self.input_hidden_w, self.hidden_output_w, self.hidden_b, self.output_b = self.init_para(self.input_size,
                                                                                                 self.hidden_size,
                                                                                                 self.output_size)
        self.train_data = train_data
        self.m = train_data.shape[1]
        self.train_value = train_value
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    @staticmethod
    def layer_sizes(x, y, h=4):
        """
        根据训练集的数据初始化神经网络节点数量
        :param x: input layer size
        :param y: output layer size
        :param h: 隐藏层节点数，默认为4
        :return: 各层的节点数
        """
        n_x = x.shape[0]
        n_h = h
        n_y = y.shape[0]
        return n_x, n_h, n_y

    @staticmethod
    def init_para(input_size, hidden_size, output_size):
        """
        使用方差为0.01的正态分布随机数初始化神经网络各节点
        :param output_size:
        :param input_size: 输入层数量
        :param hidden_size: 隐藏层数量
        :return:输入层、输出层的权重矩阵
        """
        input_hidden_w = np.random.randn(input_size, hidden_size) * 0.01
        hidden_output_w = np.random.randn(hidden_size, output_size) * 0.01
        hidden_b = np.zeros(shape=(hidden_size, 1))
        output_b = np.zeros(shape=(output_size, 1))
        return input_hidden_w, hidden_output_w, hidden_b, output_b

    @staticmethod
    def sigmoid(in_data):
        """
        sigmoid激活函数
        :param in_data:
        :return:
        """
        return 1 / (1 + np.exp(-in_data))

    def forward_propagation(self):
        """
        在hidden层使用tanh激活函数
        在output层使用sigmoid激活函数
        :return:
        """

        z1 = np.dot(self.input_hidden_w.T, self.train_data)
        z1 = z1 + self.hidden_b
        a1 = np.tanh(z1)
        # print("-------------------------")
        # print(a1.shape)
        z2 = np.dot(self.hidden_output_w.T, a1)
        z2 = z2 + self.output_b
        # print("--------------------")
        # print(z2.shape)
        #
        a2 = self.sigmoid(z2)
        self.cache['z1'] = z1
        self.cache['z2'] = z2
        self.cache['a1'] = a1
        self.cache['a2'] = a2
        return a2

    @staticmethod
    def lost_calc(src, tar, m):
        cost = np.multiply(np.log(src), tar) + np.multiply(1 - tar, np.log(1 - src))
        cost = (1 / m) * np.sum(cost, 1)
        return cost

    def back_propagation(self):
        """
        进行反向传播的计算，更新权重矩阵
        :return:
        """
        # lost = self.lost_calc(a2, self.train_value)
        dz2 = self.cache['a2'] - self.train_value
        dz1 = np.multiply(np.dot(self.hidden_output_w, dz2), 1 - np.power(self.cache['a1'], 2))

        # print(dz.shape)

        dw2 = (1 / self.m) * np.dot(dz2, self.cache['a1'].T)
        dw2 = dw2.T

        db2 = (1 / self.m) * np.sum(dz2, 1, keepdims=True)

        # print(db2)
        dw1 = (1 / self.m) * np.dot(dz1, self.train_data.T).T
        db1 = (1 / self.m) * np.sum(dz1, 1, keepdims=True)
        # print("##############")
        # print(dw2.shape)

        self.hidden_output_w = self.hidden_output_w + dw2 * self.learning_rate
        self.output_b = self.output_b - db2 * self.learning_rate

        self.input_hidden_w = self.input_hidden_w + dw1 * self.learning_rate
        self.hidden_b = self.hidden_b - db1 * self.learning_rate
        # self.hidden_b = reduce1 * self.learning_rate

        return 0

    def learning(self):
        for i in range(self.num_iterations):
            self.forward_propagation()
            lost = self.lost_calc(self.cache['a2'], self.train_value, self.m)
            self.lost.append(lost)
            self.back_propagation()
            if i % 100 == 0:
                print(self.hidden_output_w)
                print("迭代次数：", i, "损失值：", lost)

    def predict(self, predict_data):
        z1 = np.dot(self.input_hidden_w.T, predict_data)
        a1 = np.tanh(z1) + self.hidden_b
        z2 = np.dot(self.hidden_output_w.T, a1)
        a2 = self.sigmoid(z2) + self.output_b

        predictions = np.round(a2)
        return predictions
