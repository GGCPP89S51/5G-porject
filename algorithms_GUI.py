from PyQt6 import QtWidgets, QtCore, QtWebEngineWidgets
import sys, cv2, big_data
from PyQt6.QtGui import *
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest


key = "AIzaSyDwJ3GEiiLnMB-t-Mx7LzejCYXLW4pNYRo"


class Algorithms_GUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("Drone Position Algorithms")
        self.setWindowTitle("Drone Position Algorithms")
        self.resize(920, 600)
        self.ui()

    def openvcImag_to_QPixmap(self, opencvImg):
        img = cv2.cvtColor(opencvImg, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytesPerline = channel * width
        qimg = QImage(img, width, height, bytesPerline, QImage.Format.Format_RGB888)
        canvas = QPixmap(590, 580).fromImage(qimg)
        return canvas

    def show_img(self):
        if self.img_combobox.currentIndex() == 0:
            img = QImage("AccidentsListImg.png")
            self.label.setPixmap(QPixmap.fromImage(img))
        elif self.img_combobox.currentIndex() == 1:
            self.slider.setDisabled(False)
            self.slider.setRange(0, self.test.outNumberDrones())
            self.slider.setTickInterval(1)
            img = self.test.outputMatrixChanges(self.slider.value())
            self.label.setPixmap(self.openvcImag_to_QPixmap(img))
        elif self.img_combobox.currentIndex() == 2:
            self.slider.setDisabled(False)
            self.slider.setRange(0, self.test.outNumberDrones())
            self.slider.setTickInterval(1)
            img = self.test.outputFeatrueMatrixChanges(self.slider.value())
            self.label.setPixmap(self.openvcImag_to_QPixmap(img))
        elif self.img_combobox.currentIndex() == 3:
            self.label.setPixmap(
                self.openvcImag_to_QPixmap(self.test.outputAreaMatrixImg())
            )
        elif self.img_combobox.currentIndex() == 4:
            url = QtCore.QUrl(self.test.outputImgWebUrl(key))
            self.load_map_image(url)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def load_map_image(self, url):
        manager = QNetworkAccessManager(self)
        request = QNetworkRequest(url)
        reply = manager.get(request)
        reply.finished.connect(self.onDownloadFinished)

    def onDownloadFinished(self):
        reply = self.sender()
        data = reply.readAll()
        pixmap = QPixmap()
        pixmap.loadFromData(data)
        self.label.setPixmap(pixmap)
        reply.deleteLater()

    def start(self):
        self.clear()
        self.test = big_data.Feature_value_judgment()
        self.test.inputFile(self.filePath)
        self.test.inputStarttime(int(self.start_time_input.text()))
        self.test.inputEndtime(int(self.end_time_input.text()))
        self.test.inputDroneSpeed(int(self.drone_speed_input.text()))
        self.test.inputQuantity(int(self.droneQuantityInput.text()))
        self.test.inputFeaturesLowest(int(self.lowestRiskInput.text()))
        self.test.inputCityArea(int(self.cityAreaInput.text()))
        self.test.calculate()
        self.img_combobox.setDisabled(False)
        self.show_img()
        self.droneDispatchQuantityOutput.setText(str(self.test.outNumberDrones()))
        self.droneCoverageAreaOutput.setText(str(self.test.outCoverageArea()))
        self.droneProportionAreaCityOutput.setText(
            "%.2f" % self.test.outputProportionAreaCity() + "%"
        )
        self.probabilityOutput.setText(str(self.test.outputProbability()) + "%")
        dronePosition = self.test.outEndPoint()
        for index, i in enumerate(dronePosition):
            item = (
                str(index + 1)
                + " 緯度:"
                + str(i[1])
                + ", 經度:"
                + str(i[0])
                + ", 危險值:"
                + str(int(i[2]))
            )
            self.dronePositionListwidget.addItem(item)

    def clear(self):
        self.dronePositionListwidget.clear()

    def show_drone_position(self):
        url = QtCore.QUrl(
            self.test.outputImgWebUrl(
                key, self.dronePositionListwidget.currentIndex().row()
            )
        )
        self.load_map_image(url)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

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
        self.drone_speed_input.setText(str(45))
        self.start_calculat_button = QtWidgets.QPushButton("開始計算")
        self.start_calculat_button.clicked.connect(self.start)
        self.open_file_button = QtWidgets.QPushButton(self)
        self.open_file_button.setText("開啟檔案")
        self.open_file_button.clicked.connect(self.open)
        self.file_name_label = QtWidgets.QLabel()
        self.file_name_label.setWordWrap(True)
        self.img_combobox = QtWidgets.QComboBox(self)
        self.img_combobox.addItems(
            ["車禍時間分布圖", "車禍位置分布圖", "車禍位置特徵圖", "無人機覆蓋範圍", "無人機部屬位置地圖"]
        )
        self.img_combobox.currentIndexChanged.connect(self.show_img)
        self.img_combobox.setDisabled(True)
        self.droneQuantityLabel = QtWidgets.QLabel("無人機數量:")
        self.droneQuantityInput = QtWidgets.QLineEdit(self)
        self.droneQuantityInput.setText(str(100))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setDisabled(True)
        self.slider.valueChanged.connect(self.img_change)
        self.probabilityLabel = QtWidgets.QLabel("車禍覆蓋率:")
        self.probabilityOutput = QtWidgets.QLabel()
        self.dronePositionLabel = QtWidgets.QLabel("無人機部屬位置:")
        self.dronePositionListwidget = QtWidgets.QListWidget(self)
        self.dronePositionListwidget.clicked.connect(self.show_drone_position)
        self.lowestRiskLabel = QtWidgets.QLabel("最低風險值:")
        self.lowestRiskInput = QtWidgets.QLineEdit(self)
        self.lowestRiskInput.setText(str(60))
        self.cityAreaLabel = QtWidgets.QLabel("城市面積(平方公里):")
        self.cityAreaInput = QtWidgets.QLineEdit(self)
        self.cityAreaInput.setText(str(2192))
        self.droneDispatchQuantityLabel = QtWidgets.QLabel("無人機部屬數量:")
        self.droneDispatchQuantityOutput = QtWidgets.QLabel()
        self.droneCoverageAreaLabel = QtWidgets.QLabel("無人機覆蓋面積(平方公里):")
        self.droneCoverageAreaOutput = QtWidgets.QLabel()
        self.droneProportionAreaCityLabel = QtWidgets.QLabel("無人機覆蓋面積占比:")
        self.droneProportionAreaCityOutput = QtWidgets.QLabel()

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(310, 10, 590, 580)

        self.box = QtWidgets.QWidget(self)
        self.box.setGeometry(10, 10, 290, 580)

        self.layout = QtWidgets.QFormLayout(self.box)
        self.layout.addRow(self.open_file_button)
        self.layout.addRow(self.file_name_label)
        self.layout.addRow(self.start_time_label, self.start_time_input)
        self.layout.addRow(self.end_time_label, self.end_time_input)
        self.layout.addRow(self.drone_speed_label, self.drone_speed_input)
        self.layout.addRow(self.droneQuantityLabel, self.droneQuantityInput)
        self.layout.addRow(self.lowestRiskLabel, self.lowestRiskInput)
        self.layout.addRow(self.cityAreaLabel, self.cityAreaInput)
        self.layout.addRow(self.start_calculat_button)
        self.layout.addRow(self.img_combobox)
        self.layout.addRow(self.slider)
        self.layout.addRow(
            self.droneDispatchQuantityLabel, self.droneDispatchQuantityOutput
        )
        self.layout.addRow(self.droneCoverageAreaLabel, self.droneCoverageAreaOutput)
        self.layout.addRow(
            self.droneProportionAreaCityLabel, self.droneProportionAreaCityOutput
        )
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
