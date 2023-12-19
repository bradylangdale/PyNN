import numpy as np
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
        self.layers[0].data = input.data

        for i in range(1, len(self.layers)):
            if i == len(self.layers) - 1:
                self.layers[i] = self.sigmoid((self.weights[i - 1] * self.layers[i - 1]) + self.bias[i - 1])
            else:
                self.layers[i] = self.relu((self.weights[i - 1] * self.layers[i - 1]) + self.bias[i - 1])

    def sigmoid(self, m):
        result = Matrix(m.rows, m.cols, rand=False)

        for i in range(m.rows):
            result[i][0] = 1.0 / (1.0 + math.exp(-m[i][0]))

        return result
    
    def relu(self, m):
        result = Matrix(m.rows, m.cols, rand=False)

        for i in range(m.rows):
            result[i][0] = m[i][0] if m[i][0] > 0 else 0.01 * m[i][0]

        return result
    
    def backward(self, output):
        da_dz = self.dsigmoid(self.layers[-1])
        dc_da = 2 * (self.layers[-1] - output)

        delta = Matrix(da_dz.rows, da_dz.cols, rand=False)
        for i in range(da_dz.rows):
            delta[i][0] = da_dz[i][0] * dc_da[i][0]

        dc_dw = delta * self.layers[-2].transpose()

        gradient = [[dc_dw, delta]]

        for i in range(2, len(self.layers)):
            da_dz = self.drelu(self.layers[-i])
            dc_da = (self.weights[-(i - 1)].transpose() * delta)

            delta = Matrix(da_dz.rows, da_dz.cols, rand=False)
            for j in range(da_dz.rows):
                delta[j][0] = da_dz[j][0] * dc_da[j][0]

            dc_dw = delta * self.layers[-(i + 1)].transpose()

            gradient.append([dc_dw, delta])

        gradient = list(reversed(gradient))

        return gradient
    
    def optimize(self, gradient, batch_size):
        # apply gradient
        for i in range(len(self.weights)):
            self.weights[i] -= (gradient[i][0] / batch_size)
            self.bias[i] -= (gradient[i][1] / batch_size)

    def dsigmoid(self, m):
        result = Matrix(m.rows, m.cols, rand=False)

        for i in range(m.rows):
            result[i][0] = m[i][0] * (1 - m[i][0])

        return result
    
    def drelu(self, m):
        result = Matrix(m.rows, m.cols, rand=False)

        for i in range(m.rows):
            result[i][0] = 1.0 if m[i][0] > 0 else 0.1

        return result

    def sum_grads(self, a, b):
        if len(a) != len(b):
            return a if max(len(a), len(b)) == len(a) else b

        result = []
        
        for i in range(len(a)):
            result.append([a[i][0] + b[i][0], a[i][1] + b[i][1]])

        return result

    def size(self):
        length = []
        for l in self.layers:
            length.append(l.rows)

        return length

if __name__ == '__main__':
    import random


    # adding function example
    nn = NeuralNetwork([2, 2, 1])   
    gradient = []
    error = 0

    j = 0
    for i in range(200000):
        input = Matrix(2, 1, rand=False)
        input[0][0] = random.uniform(0.0, 0.5)
        input[1][0] = random.uniform(0.0, 0.5)
    
        output = Matrix(1, 1, rand=False)
        output[0][0] = input[0][0] + input[1][0]

        nn.forward(input)

        error += abs(output[0][0] - nn.layers[-1][0][0])

        gradient = nn.sum_grads(gradient, nn.backward(output))

        if j == 50:
            print('Output:', nn.layers[-1][0][0], 'Truth:', output[0][0])
            print('Delta:', abs(nn.layers[-1][0][0] - output[0][0]))
            error /= j
            print('Error:', error, 'Accuracy:', (1 - error) * 100)
            
            nn.optimize(gradient, j / math.sqrt(1 - error**2))
            gradient = []
            accuracy = 0
            j = 0

        j += 1
