from PyQt6 import QtWidgets, QtCore
import sys, cv2, big_data, hot_point
from PyQt6.QtGui import *
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest
import requests
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

key = "AIzaSyDwJ3GEiiLnMB-t-Mx7LzejCYXLW4pNYRo"


class Algorithms_GUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("Drone Position Algorithms")
        self.setWindowTitle("Drone Position Algorithms")
        self.resize(900, 600)
        self.init_ui()

    def init_ui(self):
        self.tabs = QtWidgets.QTabWidget(self)
        self.tabs.setGeometry(10, 10, 880, 580)
        self.feature_algorithm = self.create_feature_algorithm()
        self.kdtree_algorithm = self.create_kdtree_algorithm()
        self.tabs.addTab(self.feature_algorithm, "feature algorithm")
        self.tabs.addTab(self.kdtree_algorithm, "kdtree algorithm")

    def create_feature_algorithm(self):
        tab = QtWidgets.QWidget()
        self.start_time_label = QtWidgets.QLabel("起始時間:")
        self.start_time_input = QtWidgets.QLineEdit()
        self.start_time_input.setText(str(0))
        self.end_time_label = QtWidgets.QLabel("結束時間:")
        self.end_time_input = QtWidgets.QLineEdit()
        self.end_time_input.setText(str(23))
        self.drone_speed_label = QtWidgets.QLabel("無人機速率(km/h):")
        self.drone_speed_input = QtWidgets.QLineEdit()
        self.drone_speed_input.setText(str(45))
        self.start_calculat_button = QtWidgets.QPushButton("開始計算")
        self.start_calculat_button.clicked.connect(self.start)
        self.open_file_button = QtWidgets.QPushButton()
        self.open_file_button.setText("開啟檔案")
        self.open_file_button.clicked.connect(self.open)
        self.file_name_label = QtWidgets.QLabel()
        self.file_name_label.setWordWrap(True)
        self.img_combobox = QtWidgets.QComboBox()
        self.img_combobox.addItems(
            ["車禍時間分布圖", "車禍位置分布圖", "車禍位置特徵圖", "無人機覆蓋範圍", "無人機部屬位置地圖"]
        )
        self.img_combobox.currentIndexChanged.connect(self.show_img)
        self.img_combobox.setDisabled(True)
        self.droneQuantityLabel = QtWidgets.QLabel("無人機數量:")
        self.droneQuantityInput = QtWidgets.QLineEdit()
        self.droneQuantityInput.setText(str(100))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setDisabled(True)
        self.slider.valueChanged.connect(self.img_change)
        self.probabilityLabel = QtWidgets.QLabel("車禍覆蓋率:")
        self.probabilityOutput = QtWidgets.QLabel()
        self.dronePositionLabel = QtWidgets.QLabel("無人機部屬位置:")
        self.dronePositionListwidget = QtWidgets.QListWidget()
        self.dronePositionListwidget.clicked.connect(self.show_drone_position)
        self.lowestRiskLabel = QtWidgets.QLabel("最低風險值:")
        self.lowestRiskInput = QtWidgets.QLineEdit()
        self.lowestRiskInput.setText(str(60))
        self.cityAreaLabel = QtWidgets.QLabel("城市面積(平方公里):")
        self.cityAreaInput = QtWidgets.QLineEdit()
        self.cityAreaInput.setText(str(2192))
        self.droneDispatchQuantityLabel = QtWidgets.QLabel("無人機部屬數量:")
        self.droneDispatchQuantityOutput = QtWidgets.QLabel()
        self.droneCoverageAreaLabel = QtWidgets.QLabel("無人機覆蓋面積(平方公里):")
        self.droneCoverageAreaOutput = QtWidgets.QLabel()
        self.droneProportionAreaCityLabel = QtWidgets.QLabel("無人機覆蓋面積占比:")
        self.droneProportionAreaCityOutput = QtWidgets.QLabel()

        self.label = QtWidgets.QLabel(tab)
        self.label.setGeometry(310, 10, 560, 535)
        self.label.setVisible(True)

        self.graphicview = QtWidgets.QGraphicsView(tab)
        self.graphicview.setGeometry(310, 10, 560, 535)
        self.graphicscene = QtWidgets.QGraphicsScene()
        self.graphicscene.setSceneRect(0, 0, 540, 515)
        self.graphicview.setScene(self.graphicscene)

        self.box = QtWidgets.QWidget(tab)
        self.box.setGeometry(10, 10, 290, 535)

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
        self.kd_filePath = None
        return tab

    def create_kdtree_algorithm(self):
        tab = QtWidgets.QWidget()
        self.kd_box = QtWidgets.QWidget(tab)
        self.kd_box.setGeometry(10, 10, 290, 535)
        self.kd_layout = QtWidgets.QFormLayout(self.kd_box)
        self.kd_file_name_label = QtWidgets.QLabel()
        self.kd_file_name_label.setWordWrap(True)
        self.kd_open_file_button = QtWidgets.QPushButton("開啟檔案")
        self.kd_open_file_button.clicked.connect(self.kd_open)
        self.kd_drone_speed_label = QtWidgets.QLabel("無人機速率(km/h):")
        self.kd_drone_speed_input = QtWidgets.QLineEdit("45")
        self.kd_start_calculat_button = QtWidgets.QPushButton("開始計算")
        self.kd_start_calculat_button.clicked.connect(self.kd_start)
        self.kd_droneQuantityLabel = QtWidgets.QLabel("無人機數量:")
        self.kd_droneQuantityInput = QtWidgets.QLineEdit("100")
        self.kd_combobox = QtWidgets.QComboBox()
        self.kd_combobox.addItems(
            ["夜間時段:23-4", "上班通勤時段:5-8", "工作時間時段:9-15", "下班通勤時段:16-18", "空閒時段:19-22"]
        )
        self.kd_combobox.currentIndexChanged.connect(self.kd_show)
        self.kd_dronePositionLabel = QtWidgets.QLabel("無人機部屬位置:")
        self.kd_dronePositionListwidget = QtWidgets.QListWidget()
        self.kd_dronePositionListwidget.clicked.connect(self.kd_show_drone_position)

        self.kd_layout.addRow(self.kd_file_name_label)
        self.kd_layout.addRow(self.kd_open_file_button)
        self.kd_layout.addRow(self.kd_drone_speed_label, self.kd_drone_speed_input)
        self.kd_layout.addRow(self.kd_start_calculat_button)
        self.kd_layout.addRow(self.kd_droneQuantityLabel, self.kd_droneQuantityInput)
        self.kd_layout.addRow(self.kd_combobox)
        self.kd_layout.addRow(self.kd_dronePositionLabel)
        self.kd_layout.addRow(self.kd_dronePositionListwidget)

        self.kd_label = QtWidgets.QLabel(tab)
        self.kd_label.setGeometry(310, 10, 560, 535)
        self.kd_filePath = None

        return tab

    def kd_open(self):
        self.kd_filePath, self.kd_filterType = QtWidgets.QFileDialog.getOpenFileName()
        self.kd_file_name_label.setText(self.kd_filePath)

    def kd_start(self):
        if self.kd_filePath == None:
            self.mbox = QtWidgets.QMessageBox(self)
            self.mbox.information(self, "warning", "Please select a file")
        else:
            self.kd = hot_point.Drone_deployment()
            self.kd.computingHotspots(
                self.kd_filePath, int(self.kd_drone_speed_input.text())
            )
            self.kd_show()

    def kd_show(self):
        num = int(self.kd_droneQuantityInput.text())
        if self.kd_combobox.currentIndex() == 0:
            self.hot_point = self.kd.night_time_analysis(num)
        elif self.kd_combobox.currentIndex() == 1:
            self.hot_point = self.kd.commuting_work_time_analysis(num)
        elif self.kd_combobox.currentIndex() == 2:
            self.hot_point = self.kd.work_time_analysis(num)
        elif self.kd_combobox.currentIndex() == 3:
            self.hot_point = self.kd.commuting_off_work_time_analysis(num)
        elif self.kd_combobox.currentIndex() == 4:
            self.hot_point = self.kd.Leisure_time_analysis(num)
        self.refresh_kd_listwidget(self.hot_point)

    def refresh_kd_listwidget(self, hot_point):
        self.kd_dronePositionListwidget.clear()
        self.kd_dronePositionListwidget.addItem("全部")
        for i in hot_point:
            string = (
                " 緯度:" + str(i[1]) + ", 經度:" + str(i[0]) + ", 危險值:" + str(int(i[2]))
            )
            self.kd_dronePositionListwidget.addItem(string)

    def kd_show_drone_position(self):
        url = self.outputImgWebUrl(
            key, self.kd_dronePositionListwidget.currentIndex().row() - 1
        )
        self.kd_load_map_image(QtCore.QUrl(url))
        self.kd_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def outputImgWebUrl(self, key, num=None):
        url = "https://maps.googleapis.com/maps/api/staticmap?"
        if num == -1:
            url = url + "center=23.16,120.35" + "&" + "zoom=10" + "&" + "size=470x470"
            for i in self.hot_point:
                url = url + "&" + "markers="
                url = url + "size:tiny"
                url = url + "|" + "color:red"
                url = url + "|" + str(i[1]) + "," + str(i[0])
        else:
            center = (
                "center="
                + str(self.hot_point[num][1])
                + ","
                + str(self.hot_point[num][0])
            )
            url = (
                url
                + center
                + "&"
                + "zoom=15"
                + "&"
                + "size=470x470"
                + "&"
                + "markers="
                + "size:tiny"
                + "|"
                + "color:red"
                + "|"
                + str(self.hot_point[num][1])
                + ","
                + str(self.hot_point[num][0])
            )
        url = url + "&" + "key=" + key
        return url

    def kd_load_map_image(self, url):
        manager = QNetworkAccessManager(self)
        request = QNetworkRequest(url)
        reply = manager.get(request)
        reply.finished.connect(self.kd_onDownloadFinished)

    def kd_onDownloadFinished(self):
        reply = self.sender()
        data = reply.readAll()
        pixmap = QPixmap()
        pixmap.loadFromData(data)
        self.kd_label.setPixmap(pixmap)
        reply.deleteLater()

    def openvcImag_to_QPixmap(self, opencvImg):
        img = cv2.cvtColor(opencvImg, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytesPerline = channel * width
        qimg = QImage(img, width, height, bytesPerline, QImage.Format.Format_RGB888)
        canvas = QPixmap(590, 580).fromImage(qimg)
        return canvas

    def show_img(self):
        if self.img_combobox.currentIndex() == 0:
            self.label.setVisible(False)
            self.graphicview.setVisible(True)
            canvas = FigureCanvas(self.test.creatAccidentsListImg())
            self.graphicscene.addWidget(canvas)
            # self.label.setPixmap(QPixmap.fromImage(img))
        elif self.img_combobox.currentIndex() == 1:
            self.label.setVisible(True)
            self.graphicview.setVisible(False)
            self.slider.setDisabled(False)
            self.slider.setRange(0, self.test.outNumberDrones())
            self.slider.setTickInterval(1)
            img = self.test.outputMatrixChanges(self.slider.value())
            self.label.setPixmap(self.openvcImag_to_QPixmap(img))
        elif self.img_combobox.currentIndex() == 2:
            self.label.setVisible(True)
            self.graphicview.setVisible(False)
            self.slider.setDisabled(False)
            self.slider.setRange(0, self.test.outNumberDrones())
            self.slider.setTickInterval(1)
            img = self.test.outputFeatrueMatrixChanges(self.slider.value())
            self.label.setPixmap(self.openvcImag_to_QPixmap(img))
        elif self.img_combobox.currentIndex() == 3:
            self.label.setVisible(True)
            self.graphicview.setVisible(False)
            self.label.setPixmap(
                self.openvcImag_to_QPixmap(self.test.outputAreaMatrixImg())
            )
        elif self.img_combobox.currentIndex() == 4:
            self.label.setVisible(True)
            self.graphicview.setVisible(False)
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
        if self.filePath == None:
            self.mbox = QtWidgets.QMessageBox(self)
            self.mbox.information(self, "warning", "Please select a file")
        else:
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


def main():
    app = QtWidgets.QApplication(sys.argv)
    Form = Algorithms_GUI()
    Form.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
