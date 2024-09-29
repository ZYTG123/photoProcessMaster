import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QMainWindow

import cutterLabel


class cutDialog(QMainWindow):
    signalCut = pyqtSignal(list)  # 定义信号

    def __init__(self, image, width=800, height=600):
        self.counter = 0
        self.coor = [[0 for i in range(2)] for j in range(4)]
        super().__init__()
        self.image = image

        self.setWindowTitle("图片裁剪器")
        self.resize(width, height)

        self.centralwidget = QtWidgets.QWidget()
        self.centralwidget.setObjectName("centralwidget")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")

        self.label = cutterLabel.cutterLabel(image)
        self.label.setObjectName("label")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: 2px solid black")
        # 标签中的信号绑定槽
        self.label.signalChoose.connect(self.cutCounter)
        self.label.signalUnChoose.connect(self.cutBackspace)
        self.verticalLayout.addWidget(self.label)

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)

        self.verticalLayout.setStretch(0, 10)
        self.verticalLayout.setStretch(1, 1)

        self.setCentralWidget(self.centralwidget)

        self.retranslateUi(self)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "TextLabel"))
        self.label_2.setText(_translate("Dialog", "请在图像中左键点击四点以选取要截取的四边形，右键选择上一个点"))

    def cutCounter(self, xp, yp):
        self.showImg()
        self.coor[self.counter][0] = xp
        self.coor[self.counter][1] = yp
        self.counter += 1
        if self.counter == 4:
            self.signalCut.emit(self.coor)
            self.destroy()

    def cutBackspace(self):
        self.counter -= 1

    def showImg(self):
        img0 = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        blue = cv2.split(img0)[0]
        img1 = QtGui.QImage(img0, img0.shape[1], img0.shape[0], blue.shape[1] * 3, QtGui.QImage.Format_RGB888)
        img2 = QPixmap(img1).scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(img2)
