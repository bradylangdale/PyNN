from PyQt6.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton
from PyQt6 import uic
from pyqtgraph import PlotWidget
from qneural import QNeuralNetwork

from tele_graph import TelemetryGraph


class UI(QWidget):

    def __init__(self):
        super().__init__()

        # loading the ui file with uic module
        uic.loadUi('main.ui', self)

        self.neural = QNeuralNetwork([784, 8, 4, 10])

        # setup graphs
        self.accuracyGraph = TelemetryGraph(self.findChild(PlotWidget, 'accuracyGraph'))
        self.accuracyGraph.setTitle('Accuracy')
        self.accuracyGraph.addLine()
        self.accuracyGraph.x_limit = 100

        self.neural.trainingProgress.connect(self.on_training_update)

        self.findChild(QPushButton, 'startTraining').clicked.connect(self.start_training)
        self.findChild(QPushButton, 'stopTraining').clicked.connect(self.stop_training)
        self.findChild(QPushButton, 'loadNetwork').clicked.connect(self.load_network)
        self.findChild(QPushButton, 'saveNetwork').clicked.connect(self.save_network)

        self.networkName = self.findChild(QLineEdit, 'networkName')

        self.t_count = 998

    def start_training(self):
        self.neural.start_training()

    def stop_training(self):
        self.neural.stop_training()

    def load_network(self):
        self.neural.load_network(self.networkName.text())

    def save_network(self):
        if self.neural.alive:
            self.neural.stop_training()

        self.neural.save_network(self.networkName.text())

    def on_training_update(self, data):
        self.accuracyGraph.plotData(data[-1])

        self.t_count += 1

        if self.t_count == 1000:
            self.stop_training()
            self.save_network()
            self.start_training()
            self.t_count = 0

    def closeEvent(self, event):
        if self.neural.alive:
            self.neural.stop_training()


app = QApplication([])
window = UI()
window.show()
app.exec()
