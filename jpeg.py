from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
import sys
import cv2
import numpy as np
from math import ceil
from collections import Counter


def zigzag(input_matrix):
    block_size = 8
    z = np.empty([block_size*block_size])
    index = -1
    bound = 0
    for i in range(0, 2 * block_size - 1):
        if i < block_size:
            bound = 0
        else:
            bound = i - block_size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                z[index] = input_matrix[j, i-j]
            else:
                z[index] = input_matrix[i-j, j]
    return z


def zigzag_reverse(input_matrix):
    block_size = 8
    output_matrix = np.empty([block_size, block_size])
    index = -1
    bound = 0
    for i in range(0, 2 * block_size - 1):
        if i < block_size:
            bound = 0
        else:
            bound = i - block_size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                output_matrix[j, i - j] = input_matrix[index]
            else:
                output_matrix[i - j, j] = input_matrix[index]
    return output_matrix


def trim(array: np.ndarray) -> np.ndarray:
    """
    in case the trim_zeros function returns an empty array, add a zero to the array to use as the DC component
    :param numpy.ndarray array: array to be trimmed
    :return numpy.ndarray:
    """
    trimmed = np.trim_zeros(array, 'b')
    if len(trimmed) == 0:
        trimmed = np.zeros(1)
    return trimmed


def run_length_encoding(array: np.ndarray) -> list:
    """
    finds the intermediary stream representing the zigzags
    format for DC components is <size><amplitude>
    format for AC components is <run_length, size> <Amplitude of non-zero>
    :param numpy.ndarray array: zigzag vectors in array
    :returns: run length encoded values as an array of tuples
    """
    encoded = list()
    run_length = 0
    eob = ("EOB",)

    for i in range(len(array)):
        for j in range(len(array[i])):
            trimmed = trim(array[i])
            if j == len(trimmed):
                encoded.append(eob)  # EOB
                break
            if i == 0 and j == 0:  # for the first DC component
                encoded.append((int(trimmed[j]).bit_length(), trimmed[j]))
            elif j == 0:  # to compute the difference between DC components
                diff = int(array[i][j] - array[i - 1][j])
                if diff != 0:
                    encoded.append((diff.bit_length(), diff))
                else:
                    encoded.append((1, diff))
                run_length = 0
            elif trimmed[j] == 0:  # increment run_length by one in case of a zero
                run_length += 1
            else:  # intermediary steam representation of the AC components
                encoded.append((run_length, int(trimmed[j]).bit_length(), trimmed[j]))
                run_length = 0
            # send EOB
        if not (encoded[len(encoded) - 1] == eob):
            encoded.append(eob)
    return encoded


def get_freq_dict(array: list) -> dict:
    """
    returns a dict where the keys are the values of the array, and the values are their frequencies
    :param numpy.ndarray array: intermediary stream as array
    :return: frequency table
    """
    #
    data = Counter(array)
    result = {k: d / len(array) for k, d in data.items()}
    return result


def find_huffman(p: dict) -> dict:
    """
    returns a Huffman code for an ensemble with distribution p
    :param dict p: frequency table
    :returns: huffman code for each symbol
    """
    # Base case of only two symbols, assign 0 or 1 arbitrarily; frequency does not matter
    if len(p) == 2:
        return dict(zip(p.keys(), ['0', '1']))

    # Create a new distribution by merging lowest probable pair
    p_prime = p.copy()
    a1, a2 = lowest_prob_pair(p)
    p1, p2 = p_prime.pop(a1), p_prime.pop(a2)
    p_prime[a1 + a2] = p1 + p2

    # Recurse and construct code on new distribution
    c = find_huffman(p_prime)
    ca1a2 = c.pop(a1 + a2)
    c[a1], c[a2] = ca1a2 + '0', ca1a2 + '1'

    return c


def lowest_prob_pair(p):
    # Return pair of symbols from distribution p with lowest probabilities
    sorted_p = sorted(p.items(), key=lambda x: x[1])
    return sorted_p[0][0], sorted_p[1][0]
# define quantization tables


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        loadUi("demo.ui", self)
        self.setWindowIcon(QtGui.QIcon("python-icon.png"))
        self.pre_img = None
        self.origin_img = None
        self.compress_img = None
        self.dctBlocks = None
        self.qDctBlocks = None
        self.numberqtz = 0
        self.image = None
        self.frame = None
        self.xLen = 0
        self.yLen = 0
        self.quantizationRatio = 8

        # set input button
        # self.setQratio.setText(str(self.quantizationRatio))
        self.chooseImage.clicked.connect(self.open_img)
        # self.DCT.clicked.connect(self.computeDCT)
        # self.quantization.clicked.connect(self.computeqDCT)
        self.Decompress.clicked.connect(self.jpeg_compression)
        self.Reset.clicked.connect(self.computeReset)
        # self.buttonOK.clicked.connect(self.setRatio)

    @pyqtSlot()
    def loadImage(self, fname):
        self.image = cv2.imread(fname)
        self.xLen = self.image.shape[1] // 8
        self.yLen = self.image.shape[0] // 8
        self.origin_img = cv2.resize(self.image, (560, 391))
        self.tmp = self.image
        self.displayImage()

    def jpeg_compression(self):
        QTY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],  # luminance quantization table
                        [12, 12, 14, 19, 26, 48, 60, 55],
                        [14, 13, 16, 24, 40, 57, 69, 56],
                        [14, 17, 22, 29, 51, 87, 80, 62],
                        [18, 22, 37, 56, 68, 109, 103, 77],
                        [24, 35, 55, 64, 81, 104, 113, 92],
                        [49, 64, 78, 87, 103, 121, 120, 101],
                        [72, 92, 95, 98, 112, 100, 103, 99]])

        QTC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],  # chrominance quantization table
                        [18, 21, 26, 66, 99, 99, 99, 99],
                        [24, 26, 56, 99, 99, 99, 99, 99],
                        [47, 66, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99]])
        # define window size
        windowSize = len(QTY)

        # read image
        # imgOriginal = cv2.imread('phongcanh.jpg', cv2.IMREAD_COLOR)
        imgOriginal = self.image
        imgOriginal = imgOriginal[:, :, ::-1]
        # convert BGR to YCrCb
        img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2YCR_CB)
        width = len(img[0])
        height = len(img)
        y = np.zeros((height, width), np.float32) + img[:, :, 0]
        cr = np.zeros((height, width), np.float32) + img[:, :, 1]
        cb = np.zeros((height, width), np.float32) + img[:, :, 2]
        # size of the image in bits before compression
        totalNumberOfBitsWithoutCompression = len(y) * len(y[0]) * 8 + len(cb) * len(cb[0]) * 8 + len(cr) * len(
            cr[0]) * 8
        # channel values should be normalized, hence subtract 128
        y = y - 128
        cr = cr - 128
        cb = cb - 128

        # 4: 2: 2 subsampling is used # another subsampling scheme can be used
        # thus chrominance channels should be sub-sampled
        # define subsampling factors in both horizontal and vertical directions
        SSH, SSV = 2, 2
        # filter the chrominance channels using a 2x2 averaging filter # another type of filter can be used
        # crf = cv2.boxFilter(cr, ddepth=-1, ksize=(2, 2))
        # cbf = cv2.boxFilter(cb, ddepth=-1, ksize=(2, 2))

        # crSub = crf[::SSV, ::SSH]
        # cbSub = cbf[::SSV, ::SSH]

        crSub = cr[::SSV, ::SSH]
        cbSub = cb[::SSV, ::SSH]

        # check if padding is needed,
        # if yes define empty arrays to pad each channel DCT with zeros if necessary
        yWidth, yLength = ceil(len(y[0]) / windowSize) * windowSize, ceil(len(y) / windowSize) * windowSize
        if (len(y[0]) % windowSize == 0) and (len(y) % windowSize == 0):
            yPadded = y.copy()
        else:
            yPadded = np.zeros((yLength, yWidth))
            for i in range(len(y)):
                for j in range(len(y[0])):
                    yPadded[i, j] += y[i, j]

        # chrominance channels have the same dimensions, meaning both can be padded in one loop
        cWidth, cLength = ceil(len(cbSub[0]) / windowSize) * windowSize, ceil(len(cbSub) / windowSize) * windowSize
        if (len(cbSub[0]) % windowSize == 0) and (len(cbSub) % windowSize == 0):
            crPadded = crSub.copy()
            cbPadded = cbSub.copy()
        # since chrominance channels have the same dimensions, one loop is enough
        else:
            crPadded = np.zeros((cLength, cWidth))
            cbPadded = np.zeros((cLength, cWidth))
            for i in range(len(crSub)):
                for j in range(len(crSub[0])):
                    crPadded[i, j] += crSub[i, j]
                    cbPadded[i, j] += cbSub[i, j]
        # get DCT of each channel
        # define three empty matrices
        yDct, crDct, cbDct = np.zeros((yLength, yWidth)), np.zeros((cLength, cWidth)), np.zeros((cLength, cWidth))

        # number of iteration on x axis and y axis to calculate the luminance cosine transform values
        hBlocksForY = int(len(yDct[0]) / windowSize)  # number of blocks in the horizontal direction for luminance
        vBlocksForY = int(len(yDct) / windowSize)  # number of blocks in the vertical direction for luminance
        # number of iteration on x axis and y axis to calculate the chrominance channels cosine transforms values
        hBlocksForC = int(len(crDct[0]) / windowSize)  # number of blocks in the horizontal direction for chrominance
        vBlocksForC = int(len(crDct) / windowSize)  # number of blocks in the vertical direction for chrominance

        # define 3 empty matrices to store the quantized values
        yq, crq, cbq = np.zeros((yLength, yWidth)), np.zeros((cLength, cWidth)), np.zeros((cLength, cWidth))
        # and another 3 for the zigzags
        yZigzag = np.zeros(((vBlocksForY * hBlocksForY), windowSize * windowSize))
        crZigzag = np.zeros(((vBlocksForC * hBlocksForC), windowSize * windowSize))
        cbZigzag = np.zeros(((vBlocksForC * hBlocksForC), windowSize * windowSize))

        for i in range(vBlocksForY):
            for j in range(hBlocksForY):
                yDct[i * windowSize: i * windowSize + windowSize,
                j * windowSize: j * windowSize + windowSize] = cv2.dct(
                    yPadded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
                yq[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = np.floor(
                    yDct[i * windowSize: i * windowSize + windowSize,
                    j * windowSize: j * windowSize + windowSize] / QTY + 0.5)
                yZigzag[hBlocksForY * i + j] = zigzag(
                    yq[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])

        # either crq or cbq can be used to compute the number of blocks
        for i in range(vBlocksForC):
            for j in range(hBlocksForC):
                crDct[i * windowSize: i * windowSize + windowSize,
                j * windowSize: j * windowSize + windowSize] = cv2.dct(
                    crPadded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
                crq[i * windowSize: i * windowSize + windowSize,
                j * windowSize: j * windowSize + windowSize] = np.floor(
                    crDct[i * windowSize: i * windowSize + windowSize,
                    j * windowSize: j * windowSize + windowSize] / QTC + 0.5)
                crZigzag[hBlocksForC * i + j] = zigzag(
                    crq[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
                cbDct[i * windowSize: i * windowSize + windowSize,
                j * windowSize: j * windowSize + windowSize] = cv2.dct(
                    cbPadded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
                cbq[i * windowSize: i * windowSize + windowSize,
                j * windowSize: j * windowSize + windowSize] = np.floor(
                    cbDct[i * windowSize: i * windowSize + windowSize,
                    j * windowSize: j * windowSize + windowSize] / QTC + 0.5)
                cbZigzag[hBlocksForC * i + j] = zigzag(
                    cbq[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])

        yZigzag = yZigzag.astype(np.int8)
        crZigzag = crZigzag.astype(np.int8)
        cbZigzag = cbZigzag.astype(np.int8)

        # find the run length encoding for each channel
        # then get the frequency of each component in order to form a Huffman dictionary
        yEncoded = run_length_encoding(yZigzag)
        yFrequencyTable = get_freq_dict(yEncoded)
        yHuffman = find_huffman(yFrequencyTable)

        crEncoded = run_length_encoding(crZigzag)
        crFrequencyTable = get_freq_dict(crEncoded)
        crHuffman = find_huffman(crFrequencyTable)

        cbEncoded = run_length_encoding(cbZigzag)
        cbFrequencyTable = get_freq_dict(cbEncoded)
        cbHuffman = find_huffman(cbFrequencyTable)

        # calculate the number of bits to transmit for each channel
        # and write them to an output file
        file = open("CompressedImage.asfh", "w")
        yBitsToTransmit = str()
        for value in yEncoded:
            yBitsToTransmit += yHuffman[value]

        crBitsToTransmit = str()
        for value in crEncoded:
            crBitsToTransmit += crHuffman[value]

        cbBitsToTransmit = str()
        for value in cbEncoded:
            cbBitsToTransmit += cbHuffman[value]

        if file.writable():
            file.write(yBitsToTransmit + "\n" + crBitsToTransmit + "\n" + cbBitsToTransmit)
        file.close()

        totalNumberOfBitsAfterCompression = len(yBitsToTransmit) + len(crBitsToTransmit) + len(cbBitsToTransmit)
        print(
            "Compression Ratio is " + str(
                np.round(totalNumberOfBitsWithoutCompression / totalNumberOfBitsAfterCompression, 1)))

        _yq, _crq, _cbq = np.zeros((yLength, yWidth)), np.zeros((cLength, cWidth)), np.zeros((cLength, cWidth))
        _yDct, _crDct, _cbDct = np.zeros((yLength, yWidth)), np.zeros((cLength, cWidth)), np.zeros((cLength, cWidth))
        _yPadded, _crPadded, _cbPadded = np.zeros((yLength, yWidth)), np.zeros((cLength, cWidth)), np.zeros(
            (cLength, cWidth))

        for i in range(vBlocksForY):
            for j in range(hBlocksForY):
                _yq[i * windowSize: i * windowSize + windowSize,
                j * windowSize: j * windowSize + windowSize] = zigzag_reverse(
                    yZigzag[hBlocksForY * i + j])
                _yDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = (
                        QTY * _yq[i * windowSize: i * windowSize + windowSize,
                              j * windowSize: j * windowSize + windowSize])
                _yPadded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = (
                cv2.idct(
                    _yDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize]))

        for i in range(vBlocksForC):
            for j in range(hBlocksForC):
                _crq[i * windowSize: i * windowSize + windowSize,
                j * windowSize: j * windowSize + windowSize] = zigzag_reverse(
                    crZigzag[hBlocksForC * i + j])
                _crDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = (
                        QTC * _crq[i * windowSize: i * windowSize + windowSize,
                              j * windowSize: j * windowSize + windowSize])
                _crPadded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = (
                cv2.idct(
                    _crDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize]))
                _cbq[i * windowSize: i * windowSize + windowSize,
                j * windowSize: j * windowSize + windowSize] = zigzag_reverse(
                    cbZigzag[hBlocksForC * i + j])
                _cbDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = (
                        QTC * _cbq[i * windowSize: i * windowSize + windowSize,
                              j * windowSize: j * windowSize + windowSize])
                _cbPadded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = (
                cv2.idct(
                    _cbDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize]))
        _yPaddedHeight = int(_yPadded.shape[0] / 2)
        _yPaddedWidth = int(_yPadded.shape[1] / 2)
        _cr = np.zeros((_yPaddedHeight * 2, _yPaddedWidth * 2))
        _cb = np.zeros((_yPaddedHeight * 2, _yPaddedWidth * 2))

        _y = _yPadded
        for i in range(_yPaddedHeight):
            for j in range(_yPaddedWidth):
                _cr[2 * i, 2 * j] = _crPadded[i, j]
                _cr[2 * i, 2 * j + 1] = _crPadded[i, j]
                _cr[2 * i + 1, 2 * j] = _crPadded[i, j]
                _cr[2 * i + 1, 2 * j + 1] = _crPadded[i, j]
                _cb[2 * i, 2 * j] = _cbPadded[i, j]
                _cb[2 * i, 2 * j + 1] = _cbPadded[i, j]
                _cb[2 * i + 1, 2 * j] = _cbPadded[i, j]
                _cb[2 * i + 1, 2 * j + 1] = _cbPadded[i, j]

        y = y + 128
        cr = cr + 128
        cb = cb + 128
        _y = _y + 128
        _cr = _cr + 128
        _cb = _cb + 128
        _y = _y.astype(np.uint8)
        _cr = _cr.astype(np.uint8)
        _cb = _cb.astype(np.uint8)
        _y = _y[:height, :width]
        _cr = _cr[:height, :width]
        _cb = _cb[:height, :width]
        _ycbcr = cv2.merge([_y.astype(np.uint8), _cr.astype(np.uint8), _cb.astype(np.uint8)])
        _img = cv2.cvtColor(_ycbcr, cv2.COLOR_YCR_CB2BGR)
        cv2.imwrite("img_aft.jpg", _img)
        pre_img = cv2.resize(_img, (560, 391))
        cv2.imwrite("pre_img.jpg", pre_img)
        self.pre_img = cv2.imread('pre_img.jpg')
        self.displayPreImage(2)
        difference_array = np.subtract(_img, imgOriginal)
        squared_array = np.square(difference_array)
        mse = squared_array.mean()
        print(mse)
        self.disRatio.setText(str(
                np.round(totalNumberOfBitsWithoutCompression / totalNumberOfBitsAfterCompression, 1)))
        self.disMSE.setText(str(mse))


    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.origin_img.shape) == 3:
            if (self.origin_img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.origin_img, self.origin_img.shape[1], self.origin_img.shape[0], self.origin_img.strides[0],
                     qformat)
        # image.shape[0] là số pixel theo chiều Y
        # image.shape[1] là số pixel theo chiều X
        # image.shape[2] lưu số channel biểu thị mỗi pixel
        img = img.rgbSwapped()  # chuyển đổi hiệu quả một ảnh RGB thành một ảnh BGR.
        if window == 1:
            self.pre_frame.setPixmap(QPixmap.fromImage(img))
            self.pre_frame.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)  # căn chỉnh vị trí xuất hiện của hình trên lable
        if window == 2:
            self.aft_frame.setPixmap(QPixmap.fromImage(img))
            self.aft_frame.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def displayPreImage(self, window=1):
        qformat = QImage.Format_Indexed8
        #
        if len(self.pre_img.shape) == 3:
            if (self.pre_img.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.pre_img, self.pre_img.shape[1], self.pre_img.shape[0], self.pre_img.strides[0], qformat)
        # image.shape[0] là số pixel theo chiều Y
        # image.shape[1] là số pixel theo chiều X
        # image.shape[2] lưu số channel biểu thị mỗi pixel
        # img = img.rgbSwapped()  # chuyển đổi hiệu quả một ảnh RGB thành một ảnh BGR.
        # img = self.pre_img
        if window == 1:
            self.pre_frame.setPixmap(QPixmap.fromImage(img))
            self.pre_frame.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)  # căn chỉnh vị trí xuất hiện của hình trên lable
        if window == 2:
            self.aft_frame.setPixmap(QPixmap.fromImage(img))
            self.aft_frame.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def open_img(self):
        self.fname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'This PC', "Image Files (*)")

        if self.fname:
            self.loadImage(self.fname)
        else:
            print("Invalid Image")

    def setRatio(self):
        self.quantizationRatio = int(self.setQratio.text())

    def computeReset(self):
        self.pre_frame.clear()
        self.aft_frame.clear()
        self.disSizePre.setText("")
        self.disSizeCompress.setText("")
        self.disRatio.setText("")
        self.disMSE.setText("")



app = QApplication(sys.argv)
win = UI()
win.show()
sys.exit(app.exec())


