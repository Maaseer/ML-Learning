import numpy as np
import math


class NeuralNetwork:
    """
    各层节点数
    """
    layers_dim = None
    """
    训练数据
    """
    train_mini_batch = {}
    train_data = ()
    mini_batch_size = 0
    mini_batch_num = 0
    train_value = ()
    cache = {}
    """
    超参数
        学习率
        迭代次数
        正则化参数
    """
    learning_rate = 0.01
    num_iterations = 1000
    lambd = 0
    dropout_rate = 1
    """
    权重和偏移量矩阵字典
    """
    parameters = {}
    """
    每次迭代的损失值
    """
    lost = []
    train_SSE = []
    test_SSE = []

    def __init__(self, train_data=np.random.randn(2, 2), train_value=np.random.randn(2, 1),
                 learning_rate=0.01,
                 num_iterations=1000, lambd=0, dropout_rate=1,
                 layers_dim=None, mini_batch_size=0):
        """
        :param train_data: 训练集数据
        :param train_value: 训练集目标值
        :param learning_rate: 学习率，缺省为0.01
        :param num_iterations: 迭代次数，缺省为1000
        :param lambd: L2正则化的参数，缺省则关闭正则化
        :param dropout_rate: dropout概率
        :param layers_dim: 各层节点的数量
        :param mini_batch_size: 每批数据的尺寸
        """
        if layers_dim is None:
            raise ValueError("layers can not be null.")
        self.layers_dim = layers_dim
        self.init_para(layers_dim=layers_dim)
        self.lambd = lambd
        self.dropout_rate = dropout_rate
        self.train_data = train_data
        self.train_value = train_value
        if mini_batch_size != 0:
            self.reshape_data(mini_batch_size)
        else:
            self.cache["a0"] = train_data
        # print(f"train_size:{train_data.shape}")
        # print(f"mini_batch_shape:{mini_batch[0].shape}")
        # print(f"mini_batch_num:{len(mini_batch)}")
        # print(f"self_mini_batch_num:{self.mini_batch_num}")
        # print(f"mini_batch_size:{self.mini_batch_size}")
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.cache['a0'] = train_data

    def reshape_data(self, mini_batch_size):
        """
        根据微批的尺寸将训练数据重新塑形
        :param mini_batch_size: 微批的尺寸
        :return:
        """
        if mini_batch_size <= 0:
            raise ValueError("mini batch size can not smaller than 1")
        self.mini_batch_size = mini_batch_size
        self.mini_batch_num = math.ceil(self.train_value.shape[1] / mini_batch_size)
        data_mini_batch = np.array_split(self.train_data, self.mini_batch_num, 1)
        value_mini_batch = np.array_split(self.train_value, self.mini_batch_num, 1)
        self.train_mini_batch["data"] = data_mini_batch
        self.train_mini_batch["value"] = value_mini_batch

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
            # print(layers_dim)
            t = np.random.randn(layers_dim[i - 1], layers_dim[i]) * np.sqrt(
                (2 / layers_dim[i - 1]))
            t = t % 1
            self.parameters['w' + str(i)] = t
            # print(t)
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
        # print(z.max)
        # print(f"da:{da.shape}")
        # print(f"z:{z.shape}")
        s = 1 / (1 + np.exp(-z))
        # print(f"s:{s.shape}")
        dZ = np.multiply(da, np.multiply(s, (1 - s)))
        # dZ = da * s * (1 - s)
        # print(f"dz:{dZ.shape}")
        return dZ

    @staticmethod
    def relu_bac(da, z):
        dz = np.array(da, copy=True)
        dz[z <= 0] = 0
        return dz

    def lost_calc(self, src, tar):
        """
        计算损失值 参照损失函数
        :param src: 预测值
        :param tar: 目标值
        :return:
        """

        src = src * 0.99

        lost = np.multiply(-np.log(src), tar) + np.multiply(-np.log(1 - src), 1 - tar)
        # 对lost值添加L2正则化

        m = tar.shape[1]
        lost = (1. / m) * np.nansum(lost)
        if self.lambd != 0:
            l2 = 0
            for i in range(1, len(self.layers_dim) - 1):
                l2 = l2 + np.sum(np.square(self.parameters["w" + str(i)]))
            l2 = l2 * (1 / m) * (self.lambd / 2)
            lost = lost + l2
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
        # print(A.shape)
        # print(w.shape)
        # print(b.shape)
        A = np.dot(w.T, A) + b
        return A

    @staticmethod
    def MSE(src, tar):
        """
        计算均方差
        :param src: 源数据
        :param tar: 目标数据
        :return:
        """
        SSE = (1 / src.shape[1]) * np.sum(np.square(tar - src))
        return SSE

    @staticmethod
    def activation_forward(A, activation_fun):
        """
        正向传播激活函数的使用部分，使用传入的激活函数对数据进行激活
        :param A: 传入数据
        :param activation_fun:激活函数
        :return: Z
        """

        return activation_fun(A)

    def forward_propagation(self, data=None, value=None, is_learning=True):
        """
        在hidden层使用Relu激活函数
        在output层使用sigmoid激活函数
        计算正向传播的损失值
        :return:损失值
        """
        if data is None:
            data = self.cache["a0"]

        length = len(self.layers_dim) - 1
        if value is None:
            value = self.cache["a" + str(length + 1)]
        # print("forward_propagation")
        for i in range(1, length):
            w = self.parameters["w" + str(i)]

            # print(i)
            b = self.parameters["b" + str(i)]
            # print(b.shape)
            z = self.liner_forward(data, w, b)
            a = self.activation_forward(z, self.relu)
            if self.dropout_rate != 1 & is_learning:
                drop = np.random.rand(a.shape[0], a.shape[1])
                drop = drop < self.dropout_rate
                a = np.multiply(a, drop)
                a = a / self.dropout_rate
                self.cache["d" + str(i)] = drop
            self.cache["z" + str(i)] = z
            self.cache["a" + str(i)] = a
            data = a

        # a1 = self.activation_forward(z1, self.Relu)
        # print(length)
        h_o_w = self.parameters["w" + str(length)]
        h_o_b = self.parameters["b" + str(length)]
        z = self.liner_forward(data, h_o_w, h_o_b)
        a = self.activation_forward(z, self.sigmoid)
        a = a * 0.999999
        self.cache["z" + str(length)] = z
        self.cache["a" + str(length)] = a

        # print(f"data:{data.shape}")
        # print(f"w:{h_o_w.shape}")
        # print(f"b:{h_o_b.shape}")
        # print(f"z:{z.shape}")
        # print(f"a:{a.shape}")
        lost = self.lost_calc(a, value)

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

        # db = (1 / m) * np.sum(dz, 1, keepdims=True)
        db = (1 / m) * np.sum(dz, 1)
        db = db.reshape(db.shape[0], 1)
        # print(db.shape)
        # print(db)
        da_prev = np.dot(w, dz)
        return dw, db, da_prev

    def update_parameters(self, dp, m):
        dim_index = len(self.layers_dim) - 1
        # print("update_parameters")
        for (dw, db) in dp:
            # print(dw)
            # print(db)
            # print(dim_index)
            w = self.parameters["w" + str(dim_index)]
            if self.lambd != 0:
                # L2正则化
                dw = dw + (self.lambd / m) * w
            self.parameters["w" + str(dim_index)] = w - self.learning_rate * dw
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
        train_value = self.cache["a" + str(dim_size)]
        # print("back_propagation")
        m = a.shape[1]
        da = - (np.divide(train_value, a) - np.divide(1 - train_value, 1 - a))

        dw, db, da = self.single_layer_back(da, dim_index, self.sigmoid_bac)

        dp.append((dw, db))
        # print(dim_index)
        dim_index -= 1

        while dim_index != 0:
            if self.dropout_rate != 1:
                drop = self.cache["d" + str(dim_index)]
                da = np.multiply(da, drop)
                da = da * self.dropout_rate

            dw, db, da = self.single_layer_back(da, dim_index, self.relu_bac)
            dp.append((dw, db))
            # print(dim_index)
            dim_index -= 1

        # 统一更新权重矩阵
        self.update_parameters(dp, m)

    def learning(self, test_data=None, test_value=None):
        """
        使神经网络开始学习
        :return:
        """
        if self.mini_batch_size == 0:
            self.cache["a" + str(len(self.layers_dim))] = self.train_value
            for i in range(self.num_iterations):
                lost = self.forward_propagation()
                # lost = self.lost_calc(self.cache['a2'], self.train_value)
                self.lost.append(lost)
                self.back_propagation()

                if i % 100 == 0:
                    # self.lost.append(lost)
                    print("迭代次数：", i, "损失值：", lost)

                    train_result = self.cache["a" + str(len(self.layers_dim) - 1)]
                    train_SSE = self.MSE(train_result, self.train_value)
                    self.train_SSE.append(train_SSE)
                    print(f"训练集均方差：{train_SSE}")

                    if test_value is not None:
                        test_result = self.predict(test_data)
                        test_SSE = self.MSE(test_result, test_value)
                        self.test_SSE.append(test_SSE)
                        print(f"测试集均方差：{test_SSE}")
        else:  # 使用微批进行梯度下降
            # 提取数据
            print(f"使用mini batch 进行梯度下降，每次使用数据{self.mini_batch_size}份,共{self.mini_batch_num}个包")
            train_data = self.train_mini_batch["data"]
            train_value = self.train_mini_batch["value"]
            for i in range(self.num_iterations):
                for j in range(0, self.mini_batch_num):
                    # 提取当前批数据
                    # print(f"第{i+1}次迭代，第{j+1}个微批")
                    self.cache["a0"] = train_data[j]
                    # print(self.cache["a0"].shape)
                    self.cache["a" + str(len(self.layers_dim))] = train_value[j]
                    # print("a" + str(len(self.layers_dim)))
                    # print(self.cache["a0"].shape)
                    lost = self.forward_propagation()

                    self.lost.append(lost)
                    self.back_propagation()

                    if test_value is not None:
                        test_result = self.predict(test_data)
                        test_SSE = self.MSE(test_result, test_value)
                        self.test_SSE.append(test_SSE)
                if i % 100 == 0:
                    # self.lost.append(lost)
                    print("迭代次数：", i, "损失值：", lost)

                    train_result = self.cache["a" + str(len(self.layers_dim) - 1)]
                    train_SSE = self.MSE(train_result, train_value[j])
                    self.train_SSE.append(train_SSE)
                    print(f"训练集均方差：{train_SSE}")

                    # if test_value is not None:
                    #     test_result = self.predict(test_data)
                    #     test_SSE = self.MSE(test_result, test_value)
                    #     self.test_SSE.append(test_SSE)
                    #     print(f"测试集均方差：{test_SSE}")

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

        # result = np.round(result)

        return result
