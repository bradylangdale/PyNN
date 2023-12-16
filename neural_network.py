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
    
    def backward(self, output, rate_weights=0.001, rate_bias=0.001):
        dsig = self.dsigmoid(self.layers[-1])
        dc = 2 * (self.layers[-1] - output)

        delta = Matrix(dsig.rows, dsig.cols, rand=False)
        for i in range(dsig.rows):
            delta[i][0] = dsig[i][0] * dc[i][0]

        dcost = delta * self.layers[-2].transpose()

        self.weights[-1] -= rate_weights * dcost
        self.bias[-1] -= rate_bias * delta

        for i in range(2, len(self.layers)):
            dsig = self.dsigmoid(self.layers[-i])
            dc = 2 * (self.layers[-i] - delta)

            delta = Matrix(dsig.rows, dsig.cols, rand=False)
            for j in range(dsig.rows):
                delta[j][0] = dsig[j][0] * dc[j][0]

            dcost = delta * self.layers[-(i + 1)].transpose()

            self.weights[-i] -= rate_weights * dcost
            self.bias[-i] -= rate_bias * delta

        return sum(sum(x) for x in dc.data)

    def dsigmoid(self, m):
        result = Matrix(m.rows, m.cols, rand=False)

        for i in range(m.rows):
            result[i][0] = m[i][0] * (1 - m[i][0])

        return result

    def size(self):
        length = []
        for l in self.layers:
            length.append(l.rows)

        return length

if __name__ == '__main__':
    import random
    import mnist_loader
    import numpy as np


    '''nn = NeuralNetwork([784, 32, 10])
    data = list(list(mnist_loader.load_data_wrapper())[0])

    j = 10
    for i in range(len(data)):
        train_set = list(data[i])
        input = Matrix(784, 1, rand=False)
        input.data = train_set[0]

        output = Matrix(10, 1, rand=False)
        output.data = train_set[1]

        nn.forward(input)
        cost = nn.backward(output)

        if j == 10:
            #print('Output', nn.layers[-1][0][0], 'Truth:', output[0][0])
            print('Delta:', cost, 
                  'Guess:', np.argmax(nn.layers[-1].data), 
                  'Answer:', np.argmax(train_set[1]))
            
            print(nn.layers[-1])
            print(train_set[1])
            j = 0

        j += 1'''

    # adding function example
    nn = NeuralNetwork([2, 2, 1])   

    j = 1000
    for i in range(1000000):
        input = Matrix(2, 1, rand=False)
        input[0][0] = random.uniform(0.0, 0.35)
        input[1][0] = random.uniform(0.0, 0.35)
        output = Matrix(1, 1, rand=False)
        output[0][0] = input[0][0] + input[1][0]

        nn.forward(input)
        cost = nn.backward(output)

        if j == 1000:
            print('Output', nn.layers[-1][0][0], 'Truth:', output[0][0])
            print('Delta:', abs(nn.layers[-1][0][0] - output[0][0]))
            j = 0

        j += 1
