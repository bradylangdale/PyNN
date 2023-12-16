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
    
    def backward(self, output, rate_weights=0.5, rate_bias=0.5):
        dsig = self.dsigmoid(self.layers[-1])
        dc = 2 * (self.layers[-1] - output)

        delta = Matrix(dsig.rows, dsig.cols, rand=False)
        for i in range(dsig.rows):
            delta[i][0] = dsig[i][0] * dc[i][0]

        dcost = delta * self.layers[-2].transpose()

        self.weights[-1] -= rate_weights * dcost
        self.bias[-1] -= rate_bias * delta

        for i in range(2, len(self.layers)):
            dsig = self.drelu(self.layers[-i])
            dc = 2 * (self.weights[-(i - 1)].transpose() * delta)

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
    
    def drelu(self, m):
        result = Matrix(m.rows, m.cols, rand=False)

        for i in range(m.rows):
            result[i][0] = 1.0 if m[i][0] > 0 else 0.1

        return result

    def size(self):
        length = []
        for l in self.layers:
            length.append(l.rows)

        return length

if __name__ == '__main__':
    import random


    # adding function example
    nn = NeuralNetwork([2, 4, 4, 4, 1])   

    j = 1000
    for i in range(200000):
        input = Matrix(2, 1, rand=False)
        input[0][0] = random.uniform(0.0, 0.5)
        input[1][0] = random.uniform(0.0, 0.5)
        
        output = Matrix(1, 1, rand=False)
        output[0][0] = input[0][0] + input[1][0]

        nn.forward(input)

        rate = output[0][0] - nn.layers[-1][0][0]
        rate = max(0.1, min(rate, 0.99)) * 0.1

        cost = nn.backward(output, rate, rate)

        if j == 1000:
            print('Output', nn.layers[-1][0][0], 'Truth:', output[0][0])
            print('Delta:', abs(nn.layers[-1][0][0] - output[0][0]))
            j = 0

        j += 1
