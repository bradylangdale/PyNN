import math
from PyQt6.QtCore import pyqtSignal, QObject, QThread

from matrix import Matrix
from neural_network import NeuralNetwork
import random
import pickle
import mnist_loader
import numpy as np


class QNeuralNetwork(QObject):

    trainingProgress = pyqtSignal(list)

    def __init__(self, network_size=None):
        QObject.__init__(self)

        if not network_size is None:
            self.nn = NeuralNetwork(network_size)

        self.alive = False

        self.training_thread = QThread()
        self.moveToThread(self.training_thread)
        self.training_thread.started.connect(self.run)

    def inference(self, m):
        self.nn.forward(m)

        return self.nn.layers[-1].transpose()[0]

    def start_training(self):
        self.alive = True
        self.training_thread.start()

    def stop_training(self):
        self.alive = False
        self.training_thread.exit()

    def load_network(self, filename):
        with open(filename, 'rb') as f:
            self.nn = pickle.load(f)

    def save_network(self, filename=None):
        if filename is None or filename == '':
            filename = 'network-' + str(self.nn.size())

        with open(filename, 'wb') as f:
            pickle.dump(self.nn ,f)

    def run(self):
        data = list(list(mnist_loader.load_data_wrapper())[0])
        j = 0
        i = random.randint(0, len(data))
        right = []
        wrong = []
        gradient = []
        batch_size = 25

        #tested = [i]
        while self.alive:
            train_set = list(data[i])
            input = Matrix(784, 1, rand=False)
            input.data = np.array(train_set[0])

            output = Matrix(10, 1, rand=False)
            output.data = np.array(train_set[1])

            self.nn.forward(input)

            if np.argmax(self.nn.layers[-1].data) == np.argmax(train_set[1]):
                right.append(1)
                wrong.append(0)
            else:
                right.append(0)
                wrong.append(1)

            gradient = self.nn.sum_grads(gradient, self.nn.backward(output, batch_size))

            if j == batch_size:
                accuracy = (sum(right)/(sum(right) + sum(wrong)))
                if accuracy == 1:
                    accuracy = 0.99

                self.trainingProgress.emit([sum(right), sum(wrong),
                                            (sum(right)/(sum(right) + sum(wrong))) * 100])
                
                self.nn.optimize(gradient, 1 / math.sqrt(1 - accuracy**2))
                gradient = []
                j = 0

            j += 1

            i = random.randint(0, len(data) - 1)

            if len(right) >= 200:
                right.pop(0)
                wrong.pop(0)
