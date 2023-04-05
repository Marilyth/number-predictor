import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import torch
from typing import *


def start_drawing(prediction_callback: Callable[[torch.Tensor], str]):
    """Opens a window in which the user can draw numbers and let them be predicted.

    Args:
        prediction_callback (Callable[[torch.Tensor], str]): The function to call to convert an image tensor into a prediction string.
    """
    app = QApplication(sys.argv)
    w = QWidget()
    predictionLabel = QLabel()
    btnClear = QPushButton("Predict")
    drawer = Drawer()

    w.setLayout(QVBoxLayout())
    w.layout().addWidget(predictionLabel)
    w.layout().addWidget(btnClear)
    w.layout().addWidget(drawer)

    # Convert the QImage to a 28x28 torch tensor.
    def get_drawing_tensor():
        scaled_drawing = drawer.image.scaled(28, 28, Qt.KeepAspectRatio, Qt.FastTransformation).bits().asarray(28 * 28)
        return torch.tensor(np.frombuffer(scaled_drawing, dtype=np.uint8).reshape((28, 28))).to(torch.float) / 255

    # Set label to prediction.
    btnClear.clicked.connect(lambda: (predictionLabel.setText(prediction_callback(get_drawing_tensor())), drawer.clearImage()))

    w.show()
    app.exec_()


class Drawer(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setAttribute(Qt.WA_StaticContents)
        h = 308
        w = 308
        self.myPenWidth = 22
        self.myPenColor = Qt.white
        self.image = QImage(w, h, QImage.Format_Grayscale8)
        self.path = QPainterPath()
        self.clearImage()

    def setPenColor(self, newColor):
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth

    def clearImage(self):
        self.path = QPainterPath()
        self.image.fill(Qt.black)  ## switch it to else
        self.update()

    def saveImage(self, fileName, fileFormat):
        self.image.save(fileName, fileFormat)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image, self.rect())

    def mousePressEvent(self, event):
        self.path.moveTo(event.pos())

    def mouseMoveEvent(self, event):
        self.path.lineTo(event.pos())
        p = QPainter(self.image)
        p.setPen(QPen(self.myPenColor,
                      self.myPenWidth, Qt.SolidLine, Qt.RoundCap,
                      Qt.RoundJoin))
        p.drawPath(self.path)
        p.end()
        self.update()

    def sizeHint(self):
        return QSize(308, 308)
    
