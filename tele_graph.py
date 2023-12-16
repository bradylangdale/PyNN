import pyqtgraph
from pyqtgraph import PlotWidget


class TelemetryGraph:

    def __init__(self, graph, legend=False):
        self._graph = graph
        self._graph.setBackground('black')

        self.styles = {'color': '#FFFFFF', 'font-size': '10px'}

        self.x_limit = 30
        self.last_x = 0

        self._x = dict()
        self._y = dict()
        self._lines = dict()
        self._pen = dict()

        if legend:
            self._graph.addLegend(offset=(0, 0))

    def addLine(self, name='default', color='white'):
        self._x[name] = [0]
        self._y[name] = [0]
        self._pen[name] = pyqtgraph.mkPen(color=color, width=2)
        self._lines[name] = self._graph.plot(self._x[name], self._y[name], name=name)

    def plotData(self, y, x=None, name='default'):
        if x is None:
            x = self.last_x + 1

        self.last_x = x

        self._x[name].append(x)
        self._y[name].append(y)

        if len(self._x[name]) > self.x_limit:
            self._x[name] = self._x[name][1:]
            self._y[name] = self._y[name][1:]

        self._lines[name].setData(self._x[name], self._y[name], name=name, pen=self._pen[name])

    def setBackgroundColor(self, color='w'):
        self._graph.setBackground(color)

    def setTitle(self, name, color='white'):
        self._graph.setTitle(name, color=color, size='12pt')

    def setYLabel(self, name):
        self._graph.setLabel('left', name, **self.styles)

    def setXLabel(self, name):
        self._graph.setLabel('bottom', name, **self.styles)
