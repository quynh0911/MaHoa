# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\demo.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_JPEG(object):
    def setupUi(self, JPEG):
        JPEG.setObjectName("JPEG")
        JPEG.resize(1341, 797)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        JPEG.setFont(font)
        JPEG.setStyleSheet("background-color: rgb(127, 127, 127);\n"
"setWordWrap: True")
        JPEG.setInputMethodHints(QtCore.Qt.ImhNone)
        JPEG.setDocumentMode(False)
        self.centralwidget = QtWidgets.QWidget(JPEG)
        self.centralwidget.setObjectName("centralwidget")
        self.pre_frame = QtWidgets.QLabel(self.centralwidget)
        self.pre_frame.setGeometry(QtCore.QRect(70, 100, 560, 391))
        self.pre_frame.setMouseTracking(False)
        self.pre_frame.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.pre_frame.setText("")
        self.pre_frame.setObjectName("pre_frame")
        self.aft_frame = QtWidgets.QLabel(self.centralwidget)
        self.aft_frame.setGeometry(QtCore.QRect(715, 100, 560, 391))
        self.aft_frame.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.aft_frame.setText("")
        self.aft_frame.setObjectName("aft_frame")
        self.pre_img = QtWidgets.QLabel(self.centralwidget)
        self.pre_img.setEnabled(True)
        self.pre_img.setGeometry(QtCore.QRect(140, 30, 411, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.pre_img.setFont(font)
        self.pre_img.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.pre_img.setAcceptDrops(False)
        self.pre_img.setToolTip("")
        self.pre_img.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pre_img.setStyleSheet("background-color: rgb(255, 255, 127);")
        self.pre_img.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.pre_img.setFrameShadow(QtWidgets.QFrame.Plain)
        self.pre_img.setScaledContents(False)
        self.pre_img.setAlignment(QtCore.Qt.AlignCenter)
        self.pre_img.setObjectName("pre_img")
        self.aft_img = QtWidgets.QLabel(self.centralwidget)
        self.aft_img.setGeometry(QtCore.QRect(785, 30, 411, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.aft_img.setFont(font)
        self.aft_img.setStyleSheet("background-color: rgb(255, 255, 127);")
        self.aft_img.setAlignment(QtCore.Qt.AlignCenter)
        self.aft_img.setObjectName("aft_img")
        self.Decompress = QtWidgets.QPushButton(self.centralwidget)
        self.Decompress.setGeometry(QtCore.QRect(1120, 520, 181, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Decompress.setFont(font)
        self.Decompress.setStyleSheet("background-color: rgb(0, 0, 127);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 10px;\n"
"")
        self.Decompress.setObjectName("Decompress")
        self.chooseImage = QtWidgets.QPushButton(self.centralwidget)
        self.chooseImage.setGeometry(QtCore.QRect(40, 520, 181, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.chooseImage.setFont(font)
        self.chooseImage.setStyleSheet("background-color: rgb(0, 0, 127);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 10px;\n"
"")
        self.chooseImage.setObjectName("chooseImage")
        self.Reset = QtWidgets.QPushButton(self.centralwidget)
        self.Reset.setGeometry(QtCore.QRect(600, 520, 181, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Reset.setFont(font)
        self.Reset.setStyleSheet("background-color: rgb(0, 0, 127);\n"
"color: rgb(255, 255, 255);\n"
"border-radius: 10px;\n"
"")
        self.Reset.setObjectName("Reset")
        self.sizePre = QtWidgets.QLabel(self.centralwidget)
        self.sizePre.setGeometry(QtCore.QRect(730, 700, 131, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.sizePre.setFont(font)
        self.sizePre.setStyleSheet("background-color: rgb(255, 255, 127);\n"
"border-radius: 10px;\n"
"")
        self.sizePre.setAlignment(QtCore.Qt.AlignCenter)
        self.sizePre.setObjectName("sizePre")
        self.disSizePre = QtWidgets.QLabel(self.centralwidget)
        self.disSizePre.setGeometry(QtCore.QRect(930, 700, 211, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.disSizePre.setFont(font)
        self.disSizePre.setStyleSheet("border-radius: 5px;\n"
"color: rgb(0, 0, 0);\n"
"background-color: rgb(255, 255, 255);")
        self.disSizePre.setText("")
        self.disSizePre.setAlignment(QtCore.Qt.AlignCenter)
        self.disSizePre.setObjectName("disSizePre")
        self.disSizeCompress = QtWidgets.QLabel(self.centralwidget)
        self.disSizeCompress.setGeometry(QtCore.QRect(300, 700, 211, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.disSizeCompress.setFont(font)
        self.disSizeCompress.setStyleSheet("border-radius: 5px;\n"
"color: rgb(0, 0, 0);\n"
"background-color: rgb(255, 255, 255);")
        self.disSizeCompress.setText("")
        self.disSizeCompress.setAlignment(QtCore.Qt.AlignCenter)
        self.disSizeCompress.setObjectName("disSizeCompress")
        self.disRatio = QtWidgets.QLabel(self.centralwidget)
        self.disRatio.setGeometry(QtCore.QRect(930, 650, 211, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.disRatio.setFont(font)
        self.disRatio.setStyleSheet("border-radius: 5px;\n"
"color: rgb(0, 0, 0);\n"
"background-color: rgb(255, 255, 255);")
        self.disRatio.setText("")
        self.disRatio.setAlignment(QtCore.Qt.AlignCenter)
        self.disRatio.setObjectName("disRatio")
        self.sizeCompress = QtWidgets.QLabel(self.centralwidget)
        self.sizeCompress.setGeometry(QtCore.QRect(100, 700, 131, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.sizeCompress.setFont(font)
        self.sizeCompress.setStyleSheet("background-color: rgb(255, 255, 127);\n"
"border-radius: 10px;\n"
"")
        self.sizeCompress.setAlignment(QtCore.Qt.AlignCenter)
        self.sizeCompress.setObjectName("sizeCompress")
        self.Ratio = QtWidgets.QLabel(self.centralwidget)
        self.Ratio.setGeometry(QtCore.QRect(730, 650, 131, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.Ratio.setFont(font)
        self.Ratio.setStyleSheet("background-color: rgb(255, 255, 127);\n"
"border-radius: 10px;\n"
"")
        self.Ratio.setAlignment(QtCore.Qt.AlignCenter)
        self.Ratio.setObjectName("Ratio")
        self.labelMSE = QtWidgets.QLabel(self.centralwidget)
        self.labelMSE.setGeometry(QtCore.QRect(100, 650, 131, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.labelMSE.setFont(font)
        self.labelMSE.setStyleSheet("background-color: rgb(255, 255, 127);\n"
"border-radius: 10px;\n"
"")
        self.labelMSE.setAlignment(QtCore.Qt.AlignCenter)
        self.labelMSE.setObjectName("labelMSE")
        self.disMSE = QtWidgets.QLabel(self.centralwidget)
        self.disMSE.setGeometry(QtCore.QRect(300, 650, 211, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.disMSE.setFont(font)
        self.disMSE.setStyleSheet("border-radius: 5px;\n"
"color: rgb(0, 0, 0);\n"
"background-color: rgb(255, 255, 255);")
        self.disMSE.setText("")
        self.disMSE.setAlignment(QtCore.Qt.AlignCenter)
        self.disMSE.setObjectName("disMSE")
        JPEG.setCentralWidget(self.centralwidget)

        self.retranslateUi(JPEG)
        QtCore.QMetaObject.connectSlotsByName(JPEG)

    def retranslateUi(self, JPEG):
        _translate = QtCore.QCoreApplication.translate
        JPEG.setWindowTitle(_translate("JPEG", "IMAGE COMPRESSION"))
        self.pre_img.setText(_translate("JPEG", "Previous Image "))
        self.aft_img.setText(_translate("JPEG", "After Image"))
        self.Decompress.setText(_translate("JPEG", "Decompress"))
        self.chooseImage.setText(_translate("JPEG", "Choose Image"))
        self.Reset.setText(_translate("JPEG", "RESET"))
        self.sizePre.setText(_translate("JPEG", "Image original"))
        self.sizeCompress.setText(_translate("JPEG", "Compression image"))
        self.Ratio.setText(_translate("JPEG", "Compression ratio"))
        self.labelMSE.setText(_translate("JPEG", "MSE"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    JPEG = QtWidgets.QMainWindow()
    ui = Ui_JPEG()
    ui.setupUi(JPEG)
    JPEG.show()
    sys.exit(app.exec_())
