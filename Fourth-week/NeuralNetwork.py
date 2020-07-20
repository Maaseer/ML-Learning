import numpy as np


class NeuralNetwork:
    """
    各层节点数
    """
    input_size = 0
    hidden_size = 0
    output_size = 0
    layers_dim = None

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
    权重和偏移量矩阵字典
    """
    parameters = {}
    """
    每次迭代的损失值
    """
    lost = []

    def __init__(self, train_data=np.random.randn(2, 2), train_value=np.random.randn(2, 1), hidden_size=4,
                 learning_rate=0.01,
                 num_iterations=1000,
                 layers_dim=None):
        if layers_dim is None:
            self.input_size, self.hidden_size, self.output_size = self.layer_sizes(train_data, train_value, hidden_size)

            self.input_hidden_w, self.hidden_output_w, self.hidden_b, self.output_b = self.init_para(self.input_size,
                                                                                                     self.hidden_size,
                                                                                                     self.output_size)
            self.train_data = train_data
            self.m = train_data.shape[1]
            self.train_value = train_value
            self.learning_rate = learning_rate
            self.num_iterations = num_iterations
        else:
            self.layers_dim = layers_dim
            self.init_para(layers_dim=layers_dim)
            self.train_data = train_data
            self.m = train_data.shape[1]
            self.train_value = train_value
            self.learning_rate = learning_rate
            self.num_iterations = num_iterations

    @staticmethod
    def layer_sizes(x, y, h):
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

    def init_para(self, input_size=0, hidden_size=0, output_size=0, layers_dim=None):
        """
        使用方差为0.01的正态分布随机数初始化神经网络各节点
        :param layers_dim:
        :param output_size:
        :param input_size: 输入层数量
        :param hidden_size: 隐藏层数量
        :return:输入层、输出层的权重矩阵
        """
        if layers_dim is None:
            input_hidden_w = np.random.randn(input_size, hidden_size) * 0.01
            hidden_output_w = np.random.randn(hidden_size, output_size) * 0.01
            hidden_b = np.zeros(shape=(hidden_size, 1))
            output_b = np.zeros(shape=(output_size, 1))
            return input_hidden_w, hidden_output_w, hidden_b, output_b
        else:
            for i in range(1, len(layers_dim)):
                self.parameters['w' + str(i)] = np.random.randn(layers_dim[i - 1], layers_dim[i]) * 0.01
                self.parameters['b' + str(i)] = np.zeros(shape=(layers_dim[i], 1))

    @staticmethod
    def sigmoid(in_data):
        """
        sigmoid激活函数
        :param in_data:
        :return:
        """
        return 1 / (1 + np.exp(-in_data))

    @staticmethod
    def Relu(in_data):
        """
        Relu激活函数
        :param in_data:
        :return: tanh(x)
        """
        return np.tanh(in_data)

    @staticmethod
    def lost_calc(src, tar):
        """
        计算损失值 参照损失函数
        :param src: 预测值
        :param tar: 目标值
        :return:
        """
        # print(tar.shape)
        lost = tar * np.log(src) + (1 - tar) * np.log(1 - src)
        lost = (-1.0 / tar.shape[1]) * np.sum(lost, 1)
        return lost

    @staticmethod
    def liner_forward(A, w, b):
        """
        正向传播的线性部分，计算权重*输入数据+偏移量
        :param A: 输入数据
        :param w: 权重矩阵
        :param b: 偏移量
        :return: 输出数据
        """
        A = np.dot(w.T, A) + b
        return A

    @staticmethod
    def activation_forward(A, activation_fun):
        """
        正向传播激活函数的使用部分，使用传入的激活函数对数据进行激活
        :param A: 传入数据
        :param activation_fun:激活函数
        :return: Z
        """

        return activation_fun(A)

    def forward_propagation(self):
        """
        在hidden层使用tanh激活函数
        在output层使用sigmoid激活函数
        计算正向传播的损失值
        :return:损失值
        """
        length = len(self.layers_dim) - 1
        train_data = self.train_data

        for i in range(1, length):
            w = self.parameters["w" + str(i)]
            b = self.parameters["b" + str(i)]
            z = self.liner_forward(train_data, w, b)
            a = self.activation_forward(z, self.Relu)
            self.cache["z" + str(i)] = z
            self.cache["a" + str(i)] = a
            train_data = a

        # a1 = self.activation_forward(z1, self.Relu)

        h_o_w = self.parameters["w" + str(length)]
        h_o_b = self.parameters["b" + str(length)]
        z = self.liner_forward(train_data, h_o_w, h_o_b)
        a = self.activation_forward(z, self.sigmoid)
        self.cache["z" + str(length)] = z
        self.cache["a" + str(length)] = a

        # z = self.liner_forward(a1, self.hidden_output_w, self.output_b)
        # a = self.activation_forward(z2, self.sigmoid)

        lost = self.lost_calc(a, self.train_value)
        self.lost.append(lost)
        return lost

    def liner_backward(self,dz,layer_index):
        w = self.parameters["w" + str(layer_index)]
        b = self.parameters["b" + str(layer_index)]
        a = self.cache["a" + str(layer_index)]
    def back_propagation(self):
        """
        进行反向传播的计算，更新权重矩阵
        :return:
        """
        # lost = self.lost_calc(a2, self.train_value)
        dz2 = self.cache['a2'] - self.train_value
        dz1 = np.multiply(np.dot(self.hidden_output_w, dz2), 1 - np.power(self.cache['a1'], 2))

        # print(dz.shape)
        dw2 = (1.0 / self.m) * np.dot(dz2, self.cache['a1'].T).T
        db2 = (1.0 / self.m) * np.sum(dz2, 1, keepdims=True)

        # print(db2)
        dw1 = (1.0 / self.m) * np.dot(dz1, self.train_data.T).T
        db1 = (1.0 / self.m) * np.sum(dz1, 1, keepdims=True)
        # print("##############")
        # print(dw2.shape)

        self.hidden_output_w = self.hidden_output_w - dw2 * self.learning_rate
        self.output_b = self.output_b - db2 * self.learning_rate

        self.input_hidden_w = self.input_hidden_w - dw1 * self.learning_rate
        self.hidden_b = self.hidden_b - db1 * self.learning_rate

    def learning(self):
        """
        使神经网络开始学习
        :return:
        """
        for i in range(self.num_iterations):
            lost = self.forward_propagation()
            # lost = self.lost_calc(self.cache['a2'], self.train_value)
            # self.lost.append(lost)
            # self.back_propagation()

            if i % 100 == 0:
                # print(self.hidden_output_w)
                print("迭代次数：", i, "损失值：", lost)

    def predict(self, predict_data):
        """
        预测给定参数的值
        :param predict_data: 2xm的矩阵，意为点的坐标
        :return:1xm的矩阵，由0和1组成，意为该店的预测颜色
        """
        z1 = np.dot(self.input_hidden_w.T, predict_data) + self.hidden_b
        a1 = np.tanh(z1)
        z2 = np.dot(self.hidden_output_w.T, a1) + self.output_b
        a2 = self.sigmoid(z2)
        # print(a2)
        # print(self.input_hidden_w)
        # print(self.hidden_output_w)
        # print(self.hidden_b)
        # print(self.output_b)
        # print(a2)
        predictions = np.round(a2)

        return predictions
