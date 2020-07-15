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
    input_w = ()
    input_b = ()
    hidden_w = ()
    hidden_b = ()

    """
    训练数据
    """
    train_data = ()
    train_value = ()

    """
    学习率
    """
    learning_rate = 0.01
    num_iterations = 1000

    def __init__(self, train_data, train_value, learning_rate=0.01, num_iterations=1000):
        self.input_size, self.hidden_size, self.output_size = self.layer_sizes(train_data, train_value)

        self.input_w, self.hidden_w, self.hidden_b = self.init_para(self.input_size, self.hidden_size, self.output_size)
        self.train_data = train_data
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
        input_w = np.random.randn(input_size, hidden_size) * 0.01
        hidden_b = np.zeros(shape=(hidden_size, 1))
        hidden_w = np.random.randn(hidden_size, output_size) * 0.01
        return input_w, hidden_w, hidden_b

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

        z1 = np.dot(self.input_w.T, self.train_data) + self.hidden_b
        a1 = np.tanh(z1)
        # print("-------------------------")
        # print(a1.shape)
        z2 = np.dot(self.hidden_w.T, a1)
        # print("--------------------")
        # print(z2.shape)
        #
        a2 = self.sigmoid(z2)

        return a2

    def cost_calc(self, src, tar):
        cost = -1 / src.shape[1] * np.sum(np.multiply(np.log(src), tar) + np.multiply(1 - tar, np.log(1 - src)), 1)
        return cost

    @property
    def back_propagation(self):
        out_put = self.forward_propagation()
        cost1 = self.cost_calc(out_put, self.train_value)

        reduce1 = np.dot( self.hidden_w,cost1,)
        self.hidden_w = self.hidden_w - reduce1 * self.learning_rate

        self.hidden_b = reduce1 * self.learning_rate

        # print(cost1)
        return reduce1
