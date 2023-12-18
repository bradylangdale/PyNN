from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPoint, QSize, QRect, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor
from matrix import Matrix


class DigitDrawer(QWidget):

    edited = pyqtSignal()
    
    def __init__(self, parent=None):
        super(DigitDrawer, self).__init__(parent)

        self.modified = False
        self.drawing = False
        self.myPenWidth = 1
        self.myPenColor = QColor.fromRgb(255, 255, 255)
        self.image = QPixmap(QSize(28, 28))
        self.lastPoint = QPoint()

        self.clearImage()

    def clearImage(self):
        self.image.fill(QColor.fromRgb(0, 0, 0))
        self.modified = True
        self.update()

    def drawLineTo(self, endPoint):
        painter = QPainter(self.image)
        painter.setPen(QPen(self.myPenColor, self.myPenWidth))
        painter.drawLine(self.lastPoint, endPoint)
        self.modified = True

        rad = int(self.myPenWidth / 2 + 2)
        self.update(QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
        self.lastPoint = QPoint(endPoint)
        self.edited.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.lastPoint = event.pos() * (28.0/self.width())
            self.drawing = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            self.drawLineTo(event.pos() * (28.0/self.width()))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawLineTo(event.pos() * (28.0/self.width()))
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(0, 0, self.width(), self.width(), self.image)
        self.update()

    def resizeEvent(self, event):
        super(DigitDrawer, self).resizeEvent(event)
        self.update()

    def isModified(self):
        return self.modified
    
    def getMatrix(self):
        m = Matrix(784, 1, rand=False)
        pixels = []

        for x in range(28):
            for y in range(28):
                v = self.image.toImage().pixelColor(y, x).getHsv()[2]
                pixels.append([v / 255.0])
        
        m.data = pixels
        return m