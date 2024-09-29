import os
import sys

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox

import function as DYC
import cutter

APP_NAME = "拍照图像处理大师"
APP_SIZE_WIDTH = 1200
APP_SIZE_HEIGHT = 750
APP_ICON_PATH = "res/icon/icon.png"


def toQPixmap(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blue = cv2.split(image)[0]
    img = QtGui.QImage(image, image.shape[1], image.shape[0], blue.shape[1] * 3, QtGui.QImage.Format_RGB888)
    return QPixmap(img).scaled()


class FirstWin(QMainWindow):
    image = 0  # 用于存储图片
    path = ''

    def __init__(self):
        super().__init__()
        self.cwd = os.getcwd()  # 当前工作目录

        # 设置窗口基本属性
        self.setObjectName('MainWindow')
        self.setWindowTitle(APP_NAME)
        self.resize(APP_SIZE_WIDTH, APP_SIZE_HEIGHT)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(APP_ICON_PATH), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

        # 主水平布局
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")

        # 图片展示区label
        self.label = QtWidgets.QLabel(self.centralwidget)
        # self.label = testLabel.MyLabel(self.centralwidget)
        self.label.setText("")
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setObjectName("label")
        self.label.setPixmap(QPixmap("res/icon/source.png"))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: 2px solid black")

        # # 滚动区
        # self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        # self.scrollArea.setWidgetResizable(True)
        # self.scrollArea.setObjectName("scrollArea")
        # self.scrollAreaWidgetContents = QtWidgets.QWidget()
        # self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 333, 596))
        # self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        # self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        #
        # # 备用组件
        # # 用于接收文字识别信息
        # self.plainTextEdit = QtWidgets.QPlainTextEdit(self.scrollArea)
        # self.plainTextEdit.setObjectName("plainTextEdit")
        # self.plainTextEditContents = QtWidgets.QWidget()

        # 用文字区代替滚动区
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.plainTextEditContents = QtWidgets.QWidget()
        self.funHelp()

        # 设置水平布局属性
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.addWidget(self.plainTextEdit)
        self.horizontalLayout.setStretch(0, 5)
        self.horizontalLayout.setStretch(1, 2)
        self.setCentralWidget(self.centralwidget)

        # 初始化工具栏
        self.toolBar = QtWidgets.QToolBar(self)
        self.toolBar.setIconSize(QtCore.QSize(50, 50))
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolBar.setObjectName("toolBar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        # 初始化状态栏
        self.statusBar = QtWidgets.QStatusBar(self)
        self.statusBar.setObjectName("statusBar")
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("欢迎使用拍照图像处理大师", 5000)

        # 初始化菜单栏
        self.menuBar = QtWidgets.QMenuBar(self)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, APP_SIZE_WIDTH, 26))
        self.menuBar.setObjectName("menuBar")
        self.menu = QtWidgets.QMenu(self.menuBar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menuBar)
        self.menu_2.setObjectName("menu_2")
        self.setMenuBar(self.menuBar)

        # 设置动作并绑定槽
        # 1、打开
        self.actionOpen = QtWidgets.QAction(self)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("res/icon/ico11.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpen.setIcon(icon1)
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen.triggered.connect(self.funOpen)
        # 2、保存
        self.actionSave = QtWidgets.QAction(self)
        self.actionSave.setEnabled(False)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("res/icon/ico12.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave.setIcon(icon2)
        self.actionSave.setObjectName("actionSave")
        self.actionSave.triggered.connect(self.funSave)
        # 3、纹理1
        self.actionTexture1 = QtWidgets.QAction(self)
        self.actionTexture1.setEnabled(False)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("res/icon/ico21.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTexture1.setIcon(icon3)
        self.actionTexture1.setObjectName("actionTexture1")
        self.actionTexture1.triggered.connect(self.funTexture1)
        # 4、纹理2
        self.actionTexture2 = QtWidgets.QAction(self)
        self.actionTexture2.setEnabled(False)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("res/icon/ico22.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTexture2.setIcon(icon4)
        self.actionTexture2.setObjectName("actionTexture2")
        self.actionTexture2.triggered.connect(self.funTexture2)
        # 5、纹理3
        self.actionTexture3 = QtWidgets.QAction(self)
        self.actionTexture3.setEnabled(False)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("res/icon/ico23.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTexture3.setIcon(icon5)
        self.actionTexture3.setObjectName("actionTexture3")
        self.actionTexture3.triggered.connect(self.funTexture3)
        # 6、顺时针
        self.actionClockwiseRotation = QtWidgets.QAction(self)
        self.actionClockwiseRotation.setEnabled(False)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("res/icon/ico31.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionClockwiseRotation.setIcon(icon6)
        self.actionClockwiseRotation.setObjectName("actionClockwiseRotation")
        self.actionClockwiseRotation.triggered.connect(self.funClockwise)
        # 7、逆时针
        self.actionAntiClockwiseRotation = QtWidgets.QAction(self)
        self.actionAntiClockwiseRotation.setEnabled(False)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("res/icon/ico32.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAntiClockwiseRotation.setIcon(icon7)
        self.actionAntiClockwiseRotation.setObjectName("actionAntiClockwiseRotation")
        self.actionAntiClockwiseRotation.triggered.connect(self.funAntiClockwise)
        # 8、自由旋转
        self.actionFreeRotation = QtWidgets.QAction(self)
        self.actionFreeRotation.setEnabled(False)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("res/icon/ico33.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFreeRotation.setIcon(icon8)
        self.actionFreeRotation.setObjectName("actionFreeRotation")
        self.actionFreeRotation.triggered.connect(self.funFreeRotation)
        # 9、水平翻转
        self.actionFlipHorizontal = QtWidgets.QAction(self)
        self.actionFlipHorizontal.setEnabled(False)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("res/icon/ico41.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFlipHorizontal.setIcon(icon9)
        self.actionFlipHorizontal.setObjectName("actionFlipHorizontal")
        self.actionFlipHorizontal.triggered.connect(self.funFlipHorizontal)
        # 10、垂直翻转
        self.actionFlipVertical = QtWidgets.QAction(self)
        self.actionFlipVertical.setEnabled(False)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("res/icon/ico42.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFlipVertical.setIcon(icon10)
        self.actionFlipVertical.setObjectName("actionFlipVertical")
        self.actionFlipVertical.triggered.connect(self.funFlipVertical)
        # 11、放大
        self.actionMagnify = QtWidgets.QAction(self)
        self.actionMagnify.setEnabled(False)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap("res/icon/ico51.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionMagnify.setIcon(icon11)
        self.actionMagnify.setObjectName("actionMagnify")
        self.actionMagnify.triggered.connect(self.funMagnify)
        # 12、缩小
        self.actionShrink = QtWidgets.QAction(self)
        self.actionShrink.setEnabled(False)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap("res/icon/ico52.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionShrink.setIcon(icon12)
        self.actionShrink.setObjectName("actionShrink")
        self.actionShrink.triggered.connect(self.funShrink)
        # 13、裁剪
        self.actionTaylor = QtWidgets.QAction(self)
        self.actionTaylor.setEnabled(False)
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap("res/icon/ico61.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTaylor.setIcon(icon13)
        self.actionTaylor.setObjectName("actionTaylor")
        self.actionTaylor.triggered.connect(self.funTaylor)
        # 14、二值化
        self.actionBinarizaton = QtWidgets.QAction(self)
        self.actionBinarizaton.setEnabled(False)
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap("res/icon/ico62.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionBinarizaton.setIcon(icon14)
        self.actionBinarizaton.setObjectName("actionBinarizaton")
        self.actionBinarizaton.triggered.connect(self.funBinarizaton)
        # 15、平整
        self.actionLeveling = QtWidgets.QAction(self)
        self.actionLeveling.setEnabled(False)
        icon15 = QtGui.QIcon()
        icon15.addPixmap(QtGui.QPixmap("res/icon/ico63.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionLeveling.setIcon(icon15)
        self.actionLeveling.setObjectName("actionLeveling")
        self.actionLeveling.triggered.connect(self.funLeveling)
        # 16、一键处理
        self.actionSuperman = QtWidgets.QAction(self)
        self.actionSuperman.setEnabled(False)
        icon16 = QtGui.QIcon()
        icon16.addPixmap(QtGui.QPixmap("res/icon/ico71.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSuperman.setIcon(icon16)
        self.actionSuperman.setObjectName("actionSuperman")
        self.actionSuperman.triggered.connect(self.funSuperman)
        # 17、退出
        self.actionQuit = QtWidgets.QAction(self)
        self.actionQuit.setObjectName("actionQuit")
        self.actionQuit.triggered.connect(self.funQuit)
        # 18、帮助
        self.actionHelp = QtWidgets.QAction(self)
        self.actionHelp.setObjectName("actionHelp")
        self.actionHelp.triggered.connect(self.funHelp)
        # 19、关于
        self.actionAbout = QtWidgets.QAction(self)
        self.actionAbout.setObjectName("actionAbout")
        self.actionAbout.triggered.connect(self.funAbout)
        # 20、文字识别
        self.actionCharacter = QtWidgets.QAction(self)
        self.actionCharacter.setEnabled(False)
        icon20 = QtGui.QIcon()
        icon20.addPixmap(QtGui.QPixmap("res/icon/ico72.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionCharacter.setIcon(icon20)
        self.actionCharacter.setObjectName("actionCharacter")
        self.actionCharacter.triggered.connect(self.funCharacter)

        # 添加动作到工具栏与菜单栏
        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionSave)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionTexture1)
        self.toolBar.addAction(self.actionTexture2)
        self.toolBar.addAction(self.actionTexture3)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionClockwiseRotation)
        self.toolBar.addAction(self.actionAntiClockwiseRotation)
        # self.toolBar.addAction(self.actionFreeRotation)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionFlipHorizontal)
        self.toolBar.addAction(self.actionFlipVertical)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionMagnify)
        self.toolBar.addAction(self.actionShrink)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionTaylor)
        self.toolBar.addAction(self.actionBinarizaton)
        self.toolBar.addAction(self.actionLeveling)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionSuperman)
        self.menu.addAction(self.actionOpen)
        self.menu.addAction(self.actionSave)
        self.menu.addAction(self.actionQuit)
        self.menu_2.addAction(self.actionHelp)
        self.menu_2.addAction(self.actionAbout)
        self.menuBar.addAction(self.menu.menuAction())
        self.menuBar.addAction(self.menu_2.menuAction())
        self.toolBar.addAction(self.actionCharacter)

        self.retranslateUi()
        # QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.menu.setTitle(_translate("MainWindow", "开始"))
        self.menu_2.setTitle(_translate("MainWindow", "帮助"))
        self.actionOpen.setText(_translate("MainWindow", "打开"))
        self.actionOpen.setToolTip(_translate("MainWindow", "打开文件"))
        self.actionSave.setText(_translate("MainWindow", "保存"))
        self.actionSave.setToolTip(_translate("MainWindow", "保存文件"))
        self.actionTexture1.setText(_translate("MainWindow", "纹理1"))
        self.actionTexture1.setToolTip(_translate("MainWindow", "迭代最小二乘法"))
        self.actionTexture2.setText(_translate("MainWindow", "纹理2"))
        self.actionTexture2.setToolTip(_translate("MainWindow", "L0平滑"))
        self.actionTexture3.setText(_translate("MainWindow", "增强"))
        self.actionTexture3.setToolTip(_translate("MainWindow", "纹理增强"))
        self.actionClockwiseRotation.setText(_translate("MainWindow", "顺时针"))
        self.actionClockwiseRotation.setToolTip(_translate("MainWindow", "顺时针旋转"))
        self.actionAntiClockwiseRotation.setText(_translate("MainWindow", "逆时针"))
        self.actionAntiClockwiseRotation.setToolTip(_translate("MainWindow", "逆时针旋转"))
        self.actionFreeRotation.setText(_translate("MainWindow", "自由旋转"))
        self.actionFreeRotation.setToolTip(_translate("MainWindow", "自由旋转"))
        self.actionFlipHorizontal.setText(_translate("MainWindow", "水平"))
        self.actionFlipHorizontal.setToolTip(_translate("MainWindow", "水平翻转"))
        self.actionFlipVertical.setText(_translate("MainWindow", "垂直"))
        self.actionFlipVertical.setToolTip(_translate("MainWindow", "垂直翻转"))
        self.actionMagnify.setText(_translate("MainWindow", "放大"))
        self.actionMagnify.setToolTip(_translate("MainWindow", "放大图片"))
        self.actionShrink.setText(_translate("MainWindow", "缩小"))
        self.actionShrink.setToolTip(_translate("MainWindow", "缩小图片"))
        self.actionTaylor.setText(_translate("MainWindow", "裁剪"))
        self.actionTaylor.setToolTip(_translate("MainWindow", "将图片裁剪为一个自由四边形"))
        self.actionBinarizaton.setText(_translate("MainWindow", "二值化"))
        self.actionBinarizaton.setToolTip(_translate("MainWindow", "二值化"))
        self.actionLeveling.setText(_translate("MainWindow", "平整"))
        self.actionLeveling.setToolTip(_translate("MainWindow", "页面弯曲变形矫正"))
        self.actionSuperman.setText(_translate("MainWindow", "一键处理"))
        self.actionSuperman.setToolTip(_translate("MainWindow", "一件智能处理，文字识别"))
        self.actionQuit.setText(_translate("MainWindow", "退出程序"))
        self.actionHelp.setText(_translate("MainWindow", "帮助"))
        self.actionAbout.setText(_translate("MainWindow", "关于"))
        self.actionCharacter.setText(_translate("MainWindow", "文字识别"))
        self.actionCharacter.setToolTip(_translate("MainWindow", "识别文字并在弹窗显示"))

    def loadImg(self):
        img0 = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        blue = cv2.split(img0)[0]
        self.image = QtGui.QImage(img0, img0.shape[1], img0.shape[0], blue.shape[1] * 3, QtGui.QImage.Format_RGB888)
        img2 = QPixmap(self.image).scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(img2)

    def showImg(self):
        img0 = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        blue = cv2.split(img0)[0]
        img1 = QtGui.QImage(img0, img0.shape[1], img0.shape[0], blue.shape[1] * 3, QtGui.QImage.Format_RGB888)
        img2 = QPixmap(img1).scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(img2)

    # 槽函数
    # 打开图片
    def funOpen(self):
        # print('open')
        self.path = QFileDialog.getOpenFileName(self, '打开文件', self.cwd,
                                                "(*.jpg);;(*.png);;(*.bmp);;(*.tif);;All Files (*)")
        if len(self.path[0]) != 0:
            self.image = cv2.imread(self.path[0])
            self.showImg()
            # self.label.setPixmap(QPixmap(self.path[0]))
            # 工具栏使能
            self.actionSave.setEnabled(True)
            self.actionTexture1.setEnabled(True)
            self.actionTexture2.setEnabled(True)
            self.actionTexture3.setEnabled(True)
            self.actionClockwiseRotation.setEnabled(True)
            self.actionAntiClockwiseRotation.setEnabled(True)
            self.actionFreeRotation.setEnabled(True)
            self.actionFlipHorizontal.setEnabled(True)
            self.actionFlipVertical.setEnabled(True)
            self.actionMagnify.setEnabled(True)
            self.actionShrink.setEnabled(True)
            self.actionTaylor.setEnabled(True)
            self.actionBinarizaton.setEnabled(True)
            self.actionLeveling.setEnabled(True)
            self.actionSuperman.setEnabled(True)
            self.actionCharacter.setEnabled(True)
        else:
            self.statusBar.showMessage("未选择图片", 5000)

    # 保存图片
    def funSave(self):
        # print('save')
        if self.image is None:
            self.statusBar.showMessage("没有图片可保存", 5000)
        else:
            save_path = QFileDialog.getSaveFileName(self, '打开文件', self.cwd,
                                                    "(*.jpg);;(*.png);;(*.bmp);;(*.tif);;All Files (*)")
            if len(save_path[0]) != 0:
                cv2.imwrite(save_path[0], self.image)
                self.statusBar.showMessage("保存成功", 5000)
            else:
                self.statusBar.showMessage("未选择路径", 5000)

    def funTexture1(self):
        # print('tex1')
        self.image = DYC.wenlipinghua(self.image)
        self.showImg()
        self.statusBar.showMessage("纹理平滑_迭代最小二乘法成功！", 5000)

    def funTexture2(self):
        # print('tex2')
        self.image = DYC.L0Smoothing(self.image)
        self.showImg()
        self.statusBar.showMessage("纹理平滑_L0平滑成功！", 5000)

    def funTexture3(self):
        # print('tex3')
        self.image = DYC.wenlizengqiang(self.image)
        self.showImg()
        self.statusBar.showMessage("纹理增强成功！", 5000)

    def funClockwise(self):
        # print('clockwise')
        self.image = DYC.rotate_clockwise(self.image)
        self.showImg()
        self.statusBar.showMessage("顺时针旋转成功！", 5000)

    def funAntiClockwise(self):
        # print('anticlockwise')
        self.image = DYC.rotate_anticlockwise(self.image)
        self.showImg()
        self.statusBar.showMessage("逆时针旋转成功！", 5000)

    def funFreeRotation(self):
        # print('rotate')
        self.showImg()
        # TODO

    def funFlipHorizontal(self):
        # print('flipH')
        self.image = DYC.horizontal_flip(self.image)
        self.showImg()
        self.statusBar.showMessage("水平翻转成功！", 5000)

    def funFlipVertical(self):
        # print('flipV')
        self.image = DYC.vertical_flip(self.image)
        self.showImg()
        self.statusBar.showMessage("垂直翻转成功！", 5000)

    def funMagnify(self):
        # print('magnify')
        self.image = DYC.changescale(self.image, 1.1)
        self.showImg()
        self.statusBar.showMessage("放大成功！", 5000)

    def funShrink(self):
        # print('shrink')
        self.image = DYC.changescale(self.image, 0.9)
        self.showImg()
        self.statusBar.showMessage("缩小成功！", 5000)

    def funTaylor(self):
        # print('taylor')
        height = self.label.height()
        width = self.label.width()
        self.dialog = cutter.cutDialog(self.image, width, height)
        self.dialog.setWindowTitle("图片裁剪器")
        self.dialog.signalCut.connect(self.cut)  # 信号连接槽
        self.dialog.show()
        self.dialog.showImg()

    def cut(self, coor):
        # print('cut')
        # print(coor)
        self.image = DYC.wrap(self.image, coor)
        self.showImg()
        self.statusBar.showMessage("裁剪成功！", 5000)

    def funBinarizaton(self):
        # print('bin')
        # print(self.image.dtype)
        # if self.image.channels() == 1:
        #     self.statusBar.showMessage("灰度图无需二值化！", 5000)
        #     return
        self.image = DYC.erzhihua(self.image)
        self.showImg()
        self.statusBar.showMessage("二值化成功！", 5000)
        self.actionMagnify.setEnabled(False)
        self.actionShrink.setEnabled(False)
        self.actionBinarizaton.setEnabled(False)
        self.actionLeveling.setEnabled(False)
        self.actionSuperman.setEnabled(False)
        self.actionTexture1.setEnabled(False)
        self.actionTexture2.setEnabled(False)
        self.actionTexture3.setEnabled(False)

    def funLeveling(self):
        # print('level')
        # print(self.image.dtype)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = DYC.jiaozheng(self.image)
        self.showImg()
        self.statusBar.showMessage("矫正成功！", 5000)
        self.actionMagnify.setEnabled(False)
        self.actionShrink.setEnabled(False)
        self.actionBinarizaton.setEnabled(False)
        self.actionLeveling.setEnabled(False)
        self.actionSuperman.setEnabled(False)
        self.actionTexture1.setEnabled(False)
        self.actionTexture2.setEnabled(False)
        self.actionTexture3.setEnabled(False)

    def funSuperman(self):
        # print('superman')
        self.image = DYC.wenlipinghua(self.image)
        self.image = DYC.erzhihua(self.image)
        self.image = DYC.jiaozheng(self.image)
        self.showImg()
        self.statusBar.showMessage("处理成功！", 5000)
        self.actionMagnify.setEnabled(False)
        self.actionShrink.setEnabled(False)
        self.actionBinarizaton.setEnabled(False)
        self.actionLeveling.setEnabled(False)
        self.actionSuperman.setEnabled(False)
        self.actionTexture1.setEnabled(False)
        self.actionTexture2.setEnabled(False)
        self.actionTexture3.setEnabled(False)

    def funQuit(self):
        # print('quit')
        self.close()

    def funHelp(self):
        self.plainTextEdit.setPlainText("提示信息：\n\n"
                                        "1、放大缩小操作是针对图像分辨率而不是在窗口中让图片看起来更大\n\n"
                                        "2、推荐处理流程：纹理1->纹理2->二值化\n\n3、读取、保存路径不要出现中文!!!\n\n"
                                        "4、平整图像会自动二值化，因此若需平整+二值化只需直接平整即可\n\n")

    def funAbout(self):
        # print('about')
        self.aboutAutohr()

    def aboutAutohr(self):
        self.plainTextEdit.setPlainText("作者：\n董屹晨（核心算法）\n张严严（文档）\n吕行（GUI界面，部分算法）")
        QMessageBox.about(self, "关于拍照图像处理大师",
                          "版本：2.0.1\n\n"
                          "作者：\n董屹晨（核心算法）\n张严严（文档）\n吕行（GUI界面，部分算法）")

    def funCharacter(self):
        character = DYC.ocr(self.image)
        self.plainTextEdit.setPlainText(character)
        self.statusBar.showMessage("处理成功！", 5000)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FirstWin()
    window.show()
    sys.exit(app.exec_())
