from PyQt6 import QtWidgets, QtCore
import sys, cv2, big_data
from PyQt6.QtGui import *


class Algorithms_GUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("Drone Position Algorithms")
        self.setWindowTitle("Drone Position Algorithms")
        self.resize(820, 500)
        self.ui()

    def openvcImag_to_QPixmap(self, opencvImg):
        img = cv2.cvtColor(opencvImg, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytesPerline = channel * width
        qimg = QImage(img, width, height, bytesPerline, QImage.Format.Format_RGB888)
        canvas = QPixmap(300, 300).fromImage(qimg)
        return canvas

    def show_img(self):
        if self.img_combobox.currentIndex() == 0:
            img = self.test.createSpectrogram(self.test.matrix, 1)
        elif self.img_combobox.currentIndex() == 1:
            img = self.test.createSpectrogram(self.test.feature_matrix, 10)
        self.label.setPixmap(self.openvcImag_to_QPixmap(img))

    def start(self):
        self.test = big_data.Feature_value_judgment()
        self.test.inputFile(self.filePath)
        self.test.inputStarttime(int(self.start_time_input.text()))
        self.test.inputEndtime(int(self.end_time_input.text()))
        self.test.inputDroneSpeed(int(self.drone_speed_input.text()))
        self.test.inputQuantity(int(self.droneQuantityInput.text()))
        self.test.calculate()
        self.img_combobox.setDisabled(False)
        self.show_img()
        self.img_combobox.currentIndexChanged.connect(self.show_img)

    def open(self):
        self.filePath, self.filterType = QtWidgets.QFileDialog.getOpenFileName()
        self.file_name_label.setText(self.filePath)

    def img_change(self):
        print("1")

    def ui(self):
        self.start_time_label = QtWidgets.QLabel("起始時間:")
        self.start_time_input = QtWidgets.QLineEdit(self)
        self.start_time_input.setText(str(0))
        self.end_time_label = QtWidgets.QLabel("結束時間:")
        self.end_time_input = QtWidgets.QLineEdit(self)
        self.end_time_input.setText(str(23))
        self.drone_speed_label = QtWidgets.QLabel("無人機速率(km/h):")
        self.drone_speed_input = QtWidgets.QLineEdit(self)
        self.drone_speed_input.setText(str(60))
        self.start_calculat_button = QtWidgets.QPushButton("開始計算")
        self.start_calculat_button.clicked.connect(self.start)
        self.open_file_button = QtWidgets.QPushButton(self)
        self.open_file_button.setText("開啟檔案")
        self.open_file_button.clicked.connect(self.open)
        self.file_name_label = QtWidgets.QLabel()
        self.file_name_label.setWordWrap(True)
        self.img_combobox = QtWidgets.QComboBox(self)
        self.img_combobox.addItems(["分布圖", "特徵圖", "計算過程分布圖", "計算過程特徵圖", "部屬位置"])
        self.img_combobox.setDisabled(True)
        self.droneQuantityLabel = QtWidgets.QLabel("無人機數量:")
        self.droneQuantityInput = QtWidgets.QLineEdit(self)
        self.droneQuantityInput.setText(str(10))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setDisabled(True)
        self.slider.valueChanged.connect(self.img_change)

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(310, 10, 490, 480)

        self.box = QtWidgets.QWidget(self)
        self.box.setGeometry(10, 10, 290, 480)

        self.layout = QtWidgets.QFormLayout(self.box)
        self.layout.addRow(self.open_file_button)
        self.layout.addRow(self.file_name_label)
        self.layout.addRow(self.start_time_label, self.start_time_input)
        self.layout.addRow(self.end_time_label, self.end_time_input)
        self.layout.addRow(self.drone_speed_label, self.drone_speed_input)
        self.layout.addRow(self.droneQuantityLabel, self.droneQuantityInput)
        self.layout.addRow(self.start_calculat_button)
        self.layout.addRow(self.img_combobox)
        self.layout.addRow(self.slider)


def main():
    app = QtWidgets.QApplication(sys.argv)
    Form = Algorithms_GUI()
    Form.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
