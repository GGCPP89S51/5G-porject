from PyQt6 import QtWidgets, QtCore #,QtWebEngineWidgets
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
            self.label.setVisible(True)
            self.slider.setDisabled(False)
            self.slider.setRange(0,self.test.outNumberDrones())
            self.slider.setTickInterval(1)
            img = self.test.createSpectrogram(self.test.outputMatrixChanges(self.slider.value()), 1)
            self.label.setPixmap(self.openvcImag_to_QPixmap(img))
        elif self.img_combobox.currentIndex() == 1:
            self.label.setVisible(True)
            self.slider.setDisabled(False)
            self.slider.setRange(0,self.test.outNumberDrones())
            self.slider.setTickInterval(1)
            img = self.test.createSpectrogram(self.test.outputFeatrueMatrixChanges(self.slider.value()), 10)
            self.label.setPixmap(self.openvcImag_to_QPixmap(img))
        elif self.img_combobox.currentIndex==2:
            self.web.setVisible(True)
            self.label.setVisible(False)
            with open('my_map.html',"r",encoding="utf-8") as file:
                html_content=file.read()
            self.web.setHtml(html_content)
        

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
        self.probabilityLabel.setText("車禍覆蓋率:")
        self.probabilityOutput.setText(str(self.test.outputProbability())+"%")
        dronePosition=self.test.outEndPoint()
        for i in dronePosition:
            position=""
            for s in i:
                position=position+str(s)+" "
            self.dronePositionListwidget.addItem(position)

    def open(self):
        self.filePath, self.filterType = QtWidgets.QFileDialog.getOpenFileName()
        self.file_name_label.setText(self.filePath)

    def img_change(self):
        self.show_img()

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
        self.img_combobox.addItems(["分布圖", "特徵圖", "部屬位置"])
        self.img_combobox.setDisabled(True)
        self.droneQuantityLabel = QtWidgets.QLabel("無人機數量:")
        self.droneQuantityInput = QtWidgets.QLineEdit(self)
        self.droneQuantityInput.setText(str(10))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setDisabled(True)
        self.slider.valueChanged.connect(self.img_change)
        self.probabilityLabel=QtWidgets.QLabel()
        self.probabilityOutput=QtWidgets.QLabel()
        self.dronePositionLabel=QtWidgets.QLabel("無人機部屬位置:")
        self.dronePositionListwidget=QtWidgets.QListWidget(self)

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(310, 10, 490, 480)
        self.web=QtWidgets.QTextBrowser(self)
        self.web.setGeometry(310, 10, 490, 480)
        self.web.setVisible(False)
        #self.web=QtWebEngineWidgets.QWebEngineView(self)
        #self.web.setGeometry(320, 10, 490, 480)
        #self.web.load(QtCore.QUrl("https://google.com"))

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
        self.layout.addRow(self.probabilityLabel, self.probabilityOutput)
        self.layout.addRow(self.dronePositionLabel)
        self.layout.addRow(self.dronePositionListwidget)


def main():
    app = QtWidgets.QApplication(sys.argv)
    Form = Algorithms_GUI()
    Form.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
