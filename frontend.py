from PyQt6.QtWidgets import (QApplication, QWidget, QLineEdit, QPushButton,
                             QFrame, QHBoxLayout, QLabel, QVBoxLayout, QProgressBar)
from PyQt6.QtCore import Qt
from PyQt6 import uic
from pyqtgraph import PlotWidget
from digitdrawer import DigitDrawer
from qneural import QNeuralNetwork

from tele_graph import TelemetryGraph


class UI(QWidget):

    def __init__(self):
        super().__init__()

        # loading the ui file with uic module
        uic.loadUi('main.ui', self)

        self.neural = QNeuralNetwork([784, 256, 128, 64, 32, 16, 10])

        # setup graphs
        self.accuracyGraph = TelemetryGraph(self.findChild(PlotWidget, 'accuracyGraph'))
        self.accuracyGraph.setTitle('Accuracy')
        self.accuracyGraph.addLine()
        self.accuracyGraph.x_limit = 100

        self.neural.trainingProgress.connect(self.on_training_update)
        self.t_count = 0

        self.findChild(QPushButton, 'startTraining').clicked.connect(self.start_training)
        self.findChild(QPushButton, 'stopTraining').clicked.connect(self.stop_training)
        self.findChild(QPushButton, 'loadNetwork').clicked.connect(self.load_network)
        self.findChild(QPushButton, 'saveNetwork').clicked.connect(self.save_network)

        # setup digit drawer
        self.networkName = self.findChild(QLineEdit, 'networkName')
        self.digitDrawer = self.findChild(DigitDrawer, 'digitDrawer')
        self.findChild(QPushButton, 'clearButton').clicked.connect(self.digitDrawer.clearImage)

        # setup prediction viewer
        self.outputFrame = self.findChild(QFrame, 'outputFrame')
        self.predictionLayout = self.outputFrame.layout()
        self.outputBars = []

        for i in range(10):
            self.horiz = QWidget()
            self.horiz.setLayout(QHBoxLayout())
            label = QLabel(str(i))
            label.setMinimumWidth(24)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.horiz.layout().addWidget(label)
            self.outputBars.append(QProgressBar())
            self.horiz.layout().addWidget(self.outputBars[-1])
            self.predictionLayout.addWidget(self.horiz)

        self.digitDrawer.edited.connect(self.on_drawer_edited)

    def on_drawer_edited(self):
        output = self.neural.inference(self.digitDrawer.getMatrix())

        for i in range(len(output)):
            self.outputBars[i].setValue(int(output[i] * 100))

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

        #self.t_count += 1

        #if self.t_count == 1000:
        #    self.stop_training()
        #    self.save_network()
        #    self.start_training()
        #    self.t_count = 0

    def closeEvent(self, event):
        if self.neural.alive:
            self.neural.stop_training()

app = QApplication([])
window = UI()
window.show()
app.exec()
