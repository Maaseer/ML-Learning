import unittest
import ThreeLayerNN as NN
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_layer_sizes(self):
        in_data = np.random.randn(100, 100)
        out_data = np.random.randn(3, 100) * 10 % 1

        N = NN.ThreeLayerNeuralNetwork(in_data, out_data)
        # self.assertEqual(True, False)

        self.assertEqual(N.input_size, 100)
        self.assertEqual(N.hidden_size, 4)
        self.assertEqual(N.output_size, 3)

        self.assertEqual(N.input_hidden_w.shape, (100, 4))

        self.assertEqual(N.hidden_output_w.shape, (4, 3))

        self.assertEqual((N.hidden_b == np.zeros(shape=N.hidden_size)).all(), True)
        print("------test_init_para------")

        print("------in_w------")
        print(N.input_hidden_w.shape)
        # print(N.input_w)

        print("------hid_b------")
        print(N.hidden_b.shape)
        # print(N.hidden_b)

        print("------hid_w------")
        print(N.hidden_output_w.shape)
        # print(N.hidden_w)
        # print("-----Output----")
        # out = ()
        # for i in range(100):
        #     out = N.forward_propagation()
        # print(out.shape)
        # # print(out)
        # result = N.back_propagation()
        # print(result)
        N.learning()


if __name__ == '__main__':
    unittest.main()
