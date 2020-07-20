import unittest
from NeuralNetwork import NeuralNetwork


class MyTestCase(unittest.TestCase):
    def test_something(self):
        NN = NeuralNetwork(layers_dim=[5, 4, 3])

        for key in NN.parameters:
            print(f"{key} :\n {NN.parameters[key]}")


if __name__ == '__main__':
    unittest.main()
