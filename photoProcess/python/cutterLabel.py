import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import QLabel


class cutterLabel(QLabel):
    signalChoose = pyqtSignal(int, int)  # 左键信号信号
    signalUnChoose = pyqtSignal()  # 右键信号

    def __init__(self, image):
        super(cutterLabel, self).__init__()
        self.image = image
        self.isShow = False  # 防止自动触发
        self.counter = 0
        self.coor = [[0 for i in range(2)] for j in range(4)]

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter()
        if self.isShow:
            painter.begin(self)
            painter.setPen(QPen(QColor(255, 0, 0), 7))
            for i in range(self.counter):
                painter.drawPoint(self.coor[i][0], self.coor[i][1])
            painter.end()

    def mousePressEvent(self, event):
        QLabel.mousePressEvent(self, event)
        # self.clear()
        if event.buttons() == QtCore.Qt.LeftButton:
            # print("左键按下")
            globalPoint = event.globalPos()
            self.coor[self.counter][0] = event.x()
            self.coor[self.counter][1] = event.y()
            self.counter += 1
            self.isShow = True
            # 触发信号
            picturePoint = self.mapGp(event.x(), event.y())
            self.signalChoose.emit(picturePoint[0], picturePoint[1])
        elif event.buttons() == QtCore.Qt.RightButton:
            # print("右键按下")
            if self.counter > 0:
                self.counter -= 1
            self.signalUnChoose.emit()
        # 触发画图
        self.update()

    def mapGp(self, x, y):
        wLabel = self.width()
        hLabel = self.height()
        rLabel = wLabel/hLabel
        wLocal = self.image.shape[1]
        hLocal = self.image.shape[0]
        rLocal = wLocal/hLocal
        # 标签宽高比小于等于实际图片宽高比
        if rLabel <= rLocal:
            wPicture = wLabel
            hPicture = wPicture / rLocal
            remainder = (hLabel - hPicture) / 2
            yPicture = y - remainder
            R = wLocal / wPicture
            # print("标签尺寸：", wLabel, hLabel, '\n'
            #                                "显示尺寸：", wPicture, hPicture, '\n'
            #                                                             "实际尺寸：", wLocal, hLocal, '\n'
            #                                                                                      "标签坐标：", x,
            #       y, '\n'
            #                  "图片坐标：", x, yPicture, '\n'
            #                                                "实际坐标：", x * R, yPicture * R)
            return [x*R, yPicture*R]
        else:
            hPicture = hLabel
            wPicture = hPicture * rLocal
            remainder = (wLabel - wPicture) / 2
            xPicture = x - remainder
            R = hLocal / hPicture
            # print("标签尺寸：", wLabel, hLabel, '\n'
            #                                "显示尺寸：", wPicture, hPicture, '\n'
            #                                                             "实际尺寸：", wLocal, hLocal, '\n'
            #                                                                                      "标签坐标：", x,
            #       y, '\n'
            #                  "图片坐标：", xPicture, y, '\n'
            #                                                "实际坐标：", xPicture * R, y * R)
            return [xPicture*R, y*R]

