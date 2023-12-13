from matrix import Matrix
import math


class NeuralNetwork:

    def __init__(self, layers, rand=True):
        self.layers = []
        self.bias = []
        self.weights = []

        self.layers.append(Matrix(layers[0], 1, rand))

        for i in range(1, len(layers) - 1):
            self.layers.append(Matrix(layers[i], 1, rand))

        self.layers.append(Matrix(layers[-1], 1, rand))

        for i in range(1, len(self.layers)):
            self.bias.append(Matrix(self.layers[i].rows, 1, rand))
            self.weights.append(Matrix(self.layers[i].rows, self.layers[i - 1].rows, rand))

    def forward(self, input):
        self.layers[0] = input

        for i in range(1, len(self.layers)):
            self.layers[i] = self.sigmoid((self.weights[i - 1] * self.layers[i - 1]) + self.bias[i - 1])

    def sigmoid(self, m):
        for i in range(m.rows):
            m[i][0] = 1.0 / (1.0 + math.exp(-m[i][0]))

        return m
    
    def backward(self, output):
        pass

    def size(self):
        length = []
        for l in self.layers:
            length.append(l.rows)

        return length

if __name__ == '__main__':
    nn = NeuralNetwork([10, 10, 10])

    print(nn.size())

    m = Matrix(10, 1, rand=True)
    nn.forward(m)

    print(nn.layers[-1])

