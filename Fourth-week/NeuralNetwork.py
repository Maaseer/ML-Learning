import numpy as np


class NeuralNetwork:
    """
    各层节点数
    """
    layers_dim = None
    """
    训练数据
    """
    train_data = ()
    train_value = ()
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

    def __init__(self, train_data=np.random.randn(2, 2), train_value=np.random.randn(2, 1),
                 learning_rate=0.01,
                 num_iterations=1000,
                 layers_dim=None):

        self.layers_dim = layers_dim
        self.init_para(layers_dim=layers_dim)

        self.train_data = train_data
        self.train_value = train_value
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.cache['a0'] = train_data

    def init_para(self, layers_dim=None):
        """
        使用方差为0.01的正态分布随机数初始化神经网络各节点
        :param layers_dim:各层节点数向量
        :return:权重偏移矩阵字典
        """
        # print(layers_dim)
        # print(len(layers_dim))
        for i in range(1, len(layers_dim)):
            # print(i)
            self.parameters['w' + str(i)] = np.random.randn(layers_dim[i - 1], layers_dim[i]) / np.sqrt(
                layers_dim[i - 1])
            self.parameters['b' + str(i)] = np.zeros(shape=(layers_dim[i], 1))
        return self.parameters

    @staticmethod
    def sigmoid(in_data):
        """
        sigmoid激活函数
        :param in_data:
        :return:
        """
        return 1 / (1 + np.exp(-in_data))

    @staticmethod
    def relu(in_data):
        """
        Relu激活函数
        :param in_data:
        :return: Relu(x)
        """
        return np.maximum(in_data, 0)

    @staticmethod
    def sigmoid_bac(da, z):
        s = 1 / (1 + np.exp(-z))
        dZ = da * s * (1 - s)
        return dZ

    @staticmethod
    def relu_bac(da, z):
        dz = np.array(da, copy=True)
        dz[z <= 0] = 0
        return dz

    @staticmethod
    def lost_calc(src, tar):
        """
        计算损失值 参照损失函数
        :param src: 预测值
        :param tar: 目标值
        :return:
        """
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

    def forward_propagation(self, data=None):
        """
        在hidden层使用Relu激活函数
        在output层使用sigmoid激活函数
        计算正向传播的损失值
        :return:损失值
        """
        if data is None:
            data = self.train_data

        length = len(self.layers_dim) - 1
        # print("forward_propagation")
        for i in range(1, length):
            w = self.parameters["w" + str(i)]
            # print(i)
            b = self.parameters["b" + str(i)]
            # print(b.shape)
            z = self.liner_forward(data, w, b)
            a = self.activation_forward(z, self.relu)
            self.cache["z" + str(i)] = z
            self.cache["a" + str(i)] = a
            data = a

        # a1 = self.activation_forward(z1, self.Relu)
        # print(length)
        h_o_w = self.parameters["w" + str(length)]
        h_o_b = self.parameters["b" + str(length)]
        z = self.liner_forward(data, h_o_w, h_o_b)
        a = self.activation_forward(z, self.sigmoid)
        self.cache["z" + str(length)] = z
        self.cache["a" + str(length)] = a

        lost = self.lost_calc(a, self.train_value)

        return lost

    @staticmethod
    def liner_backward(a_prev, dz, w):
        """
        反向传播的线性部分，根据dz和前一层的输出计算出dw、db
        :param w:
        :param a_prev:
        :param dz: dz
        :return:dw,db
        """
        m = dz.shape[1]
        dw = (1 / m) * np.dot(dz, a_prev.T).T
        db = (1 / m) * np.sum(dz, 1, keepdims=True)
        da_prev = np.dot(w, dz)
        return dw, db, da_prev

    def update_parameters(self, dp):
        dim_index = len(self.layers_dim) - 1
        # print("update_parameters")
        for (dw, db) in dp:
            # print(dw)
            # print(db)
            # print(dim_index)

            self.parameters["w" + str(dim_index)] = self.parameters["w" + str(dim_index)] - self.learning_rate * dw
            self.parameters["b" + str(dim_index)] = self.parameters["b" + str(dim_index)] - self.learning_rate * db
            dim_index -= 1

    def single_layer_back(self, da, dim_index, back_activation):
        a_prev = self.cache["a" + str(dim_index - 1)]
        w = self.parameters["w" + str(dim_index)]
        z = self.cache["z" + str(dim_index)]
        dz = back_activation(da, z)
        dw, db, da = self.liner_backward(a_prev, dz, w)
        # print(dw)
        # print(db)
        return dw, db, da

    def back_propagation(self):
        """
        计算出da，再通过激活函数的反向传播计算出dz。
        根据dz计算出da（下一层）、dw、db，并更新权重矩阵
        :return:
        """
        dim_size = len(self.layers_dim)
        dim_index = dim_size - 1
        dp = []

        a = self.cache["a" + str(dim_index)]
        # print("back_propagation")

        da = - (np.divide(self.train_value, a) - np.divide(1 - self.train_value, 1 - a))
        # da = - (self.train_value / a) - (1 - self.train_value) / (1 - a)
        dw, db, da = self.single_layer_back(da, dim_index, self.sigmoid_bac)

        dp.append((dw, db))
        # print(dim_index)
        dim_index -= 1

        while dim_index != 0:
            dw, db, da = self.single_layer_back(da, dim_index, self.relu_bac)
            dp.append((dw, db))
            # print(dim_index)
            dim_index -= 1

        # 统一更新权重矩阵
        self.update_parameters(dp)

        # dz2 = self.cache['a2'] - self.train_value
        # dz1 = np.multiply(np.dot(self.hidden_output_w, dz2), 1 - np.power(self.cache['a1'], 2))
        #
        # # print(dz.shape)
        # dw2 = (1.0 / self.m) * np.dot(dz2, self.cache['a1'].T).T
        # db2 = (1.0 / self.m) * np.sum(dz2, 1, keepdims=True)
        #
        # # print(db2)
        # dw1 = (1.0 / self.m) * np.dot(dz1, self.train_data.T).T
        # db1 = (1.0 / self.m) * np.sum(dz1, 1, keepdims=True)
        # # print("##############")
        # # print(dw2.shape)
        #
        # self.hidden_output_w = self.hidden_output_w - dw2 * self.learning_rate
        # self.output_b = self.output_b - db2 * self.learning_rate
        #
        # self.input_hidden_w = self.input_hidden_w - dw1 * self.learning_rate
        # self.hidden_b = self.hidden_b - db1 * self.learning_rate

    def learning(self):
        """
        使神经网络开始学习
        :return:
        """
        for i in range(self.num_iterations):
            lost = self.forward_propagation()
            # lost = self.lost_calc(self.cache['a2'], self.train_value)
            # self.lost.append(lost)
            self.back_propagation()

            if i % 100 == 0:
                self.lost.append(lost)
                print("迭代次数：", i, "损失值：", lost)

    def predict(self, predict_data):
        """
        预测给定参数的值
        :param predict_data: 12288xm的矩阵，为猫片的像素点
        :return:1xm的矩阵，由0和1组成，意为该图片的预测结果
        """
        length = len(self.layers_dim) - 1

        for i in range(1, length):
            w = self.parameters["w" + str(i)]
            b = self.parameters["b" + str(i)]
            z = self.liner_forward(predict_data, w, b)
            a = self.activation_forward(z, self.relu)
            predict_data = a

        h_o_w = self.parameters["w" + str(length)]
        h_o_b = self.parameters["b" + str(length)]
        z = self.liner_forward(predict_data, h_o_w, h_o_b)
        result = self.activation_forward(z, self.sigmoid)

        predictions = np.round(result)

        return predictions
