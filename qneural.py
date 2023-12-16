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
        j = 10
        i = random.randint(0, len(data))
        right = 0
        wrong = 0
        cost = 0

        tested = [i]
        while self.alive:
            train_set = list(data[i])
            input = Matrix(784, 1, rand=False)
            input.data = train_set[0]

            output = Matrix(10, 1, rand=False)
            output.data = train_set[1]

            self.nn.forward(input)

            if np.argmax(self.nn.layers[-1].data) == np.argmax(train_set[1]):
                right += 1
            else:
                wrong += 1

            index = np.argmax(train_set[1])
            rate = 1 - self.nn.layers[-1][index][0]
            rate = max(0.1, min(rate, 0.99)) * 0.01
            cost = self.nn.backward(output, rate, rate)

            if j == 10:
                self.trainingProgress.emit([cost, right, wrong, (right/(right + wrong)) * 100])
                j = 0

            j += 1

            i = random.randint(0, len(data) - 1)
            while i in tested:
                i = random.randint(0, len(data) - 1)
            tested.append(i)

            if len(data) - 1 == len(tested):
                i = random.randint(0, len(data) - 1)
                tested = [i]
