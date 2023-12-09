import pandas as pd
import numpy as np
import math
import cv2
import folium
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic
from folium import plugins
import matplotlib.pyplot as plt


# 特徵值判斷演算法
class Feature_value_judgment:
    # 建立地圖陣列
    def __init__(self):
        self.radius = 10  # 單位100m
        self.train_df, self.test_df = None, None
        self.matrix_changes = []
        self.featrue_matrix_changes = []
        self.num = 0
        self.counter = 0
        self.end_point = []
        self.Probability = 0
        self.start_time = 0
        self.end_time = 24
        self.matrix = None
        self.quantity = 100
        self.Features_lowest = 60
        self.area_matrix = None
        self.total_sum = 0
        self.Area = 0
        self.city_area = 0
        self.accidents_list = [0] * 24

    # 輸入檔案
    def inputFile(self, file):
        df = pd.read_csv(file, encoding="utf-8")
        df = df[
            df["發生時間"].apply(
                lambda x: self.__judgmentTime(
                    x, start_time=self.start_time, end_time=self.end_time
                )
            )
        ]
        self.train_df, self.test_df = train_test_split(
            df, test_size=0.15, random_state=42
        )

        """
        self.train_df.to_csv('train_data.csv', index=False)
        self.test_df.to_csv('test_data.csv', index=False)
        """
        # 地圖的(最東)(最北)(最西)(最南)
        self.boundary = self.__findBoundary(self.train_df)

    # 輸入無人機時速
    def inputDroneSpeed(self, speed):
        self.radius = int(speed / 6)

    # 輸入開始時間
    def inputStarttime(self, start_time):
        self.start_time = start_time

    # 輸入結束時間
    def inputEndtime(self, end_time):
        self.end_time = end_time

    # 輸入無人機數量
    def inputQuantity(self, quantity):
        self.quantity = quantity

    # 輸入最小風險值
    def inputFeaturesLowest(self, Features_lowest):
        self.Features_lowest = Features_lowest

    # 輸入城市總面積
    def inputCityArea(self, city_area):
        self.city_area = city_area

    # 建立地圖矩陣
    def __createMatrix(self):
        self.matrix = self.__createMapMatrixTimeRange(
            self.train_df, self.boundary, self.start_time, self.end_time
        )
        self.matrix_changes.append(self.createSpectrogram(self.matrix, 1))
        """
        self.initial_location = [22.9969 , 120.213]
        self.mymap = folium.Map(location=self.initial_location, zoom_start=15)
        """
        self.area_matrix = self.__createMapMatrix(self.boundary)
        self.area_matrix = np.pad(
            self.area_matrix, pad_width=self.radius, mode="constant", constant_values=0
        )

    # 將時間分割並判斷
    def __judgmentTime(self, i, start_time=None, end_time=None):
        self.accidents_list[i // 10000] += 1
        if start_time == None or end_time == None:
            return True

        # 將時間值轉換為小時、分鐘和秒
        hour, minute, second = i // 10000, math.floor(i / 10000 % 1 * 100), i % 100

        # 確保開始時間小於結束時間
        if start_time > end_time:
            return (hour >= start_time and hour < 24) or (
                hour <= end_time and hour >= 0
            )

        # 判斷時間是否在範圍內
        return start_time <= hour <= end_time

    # 判斷要運算的地圖範圍 (最東)(最北)(最西)(最南)
    def __findBoundary(self, df):
        max_lon = 0
        max_lat = 0
        min_lon = float("inf")
        min_lat = float("inf")

        for lon, lat in zip(df["GPS經度"], df["GPS緯度"]):
            if max_lon < lon:
                max_lon = lon
            if max_lat < lat:
                max_lat = lat
            if min_lon > lon:
                min_lon = lon
            if min_lat > lat:
                min_lat = lat
        boundary = [max_lon, max_lat, min_lon, min_lat]
        return boundary

    # 將經緯度轉換成建立矩陣大小的整數並建立矩陣每個矩陣為地圖上100m*100m的範圍空間
    def __createMapMatrix(self, boundary):
        boundarys = list(boundary)
        boundarys[0] = math.ceil(boundarys[0] * 1000) / 1000
        boundarys[1] = math.ceil(boundarys[1] * 1000) / 1000
        boundarys[2] = math.floor(boundarys[2] * 1000) / 1000
        boundarys[3] = math.floor(boundarys[3] * 1000) / 1000
        Long = math.floor((boundarys[1] - boundarys[3]) * 1000)
        wight = math.floor((boundarys[0] - boundarys[2]) * 1000)
        matrix = np.zeros((wight + 1, Long + 1), dtype=int)
        return matrix

    # 將符合時間範圍內的資料填入建立好的矩陣(位於矩陣範圍內的點每有一點矩陣內的值即1)
    def __punctuation(self, df, matrix, boundary, start_time=None, end_time=None):
        boundarys = list(boundary)
        min_wight = math.floor(boundarys[2] * 1000) / 1000
        min_long = math.floor(boundarys[3] * 1000) / 1000

        for lon, lat, time in zip(df["GPS經度"], df["GPS緯度"], df["發生時間"]):
            if self.__judgmentTime(time, start_time, end_time):
                long = round(lon - min_wight, 3) * 1000
                wight = round(lat - min_long, 3) * 1000
                matrix[int(long)][int(wight)] += 1

    # 矩陣可視化
    def createSpectrogram(self, hazard_distribution_array, Size):
        size = np.shape(hazard_distribution_array)
        img = np.zeros((size[0], size[1], 3), np.uint8)
        color = [
            [255, 255, 255],
            [176, 211, 93],
            [190, 215, 72],
            [209, 223, 66],
            [242, 235, 56],
            [254, 242, 2],
            [255, 217, 0],
            [252, 177, 24],
            [250, 157, 28],
            [247, 139, 31],
            [243, 121, 31],
            [243, 112, 43],
            [243, 101, 48],
            [242, 90, 49],
            [240, 65, 48],
            [238, 26, 47],
            [197, 37, 65],
            [187, 3, 75],
            [198, 49, 107],
            [179, 44, 120],
            [165, 63, 151],
            [122, 60, 145],
        ]
        for x in range(size[0]):
            for y in range(size[1]):
                if hazard_distribution_array[x][y] / Size < 20:
                    n = hazard_distribution_array[x][y] // Size
                    n = int(n)
                else:
                    n = 20
                img[x][y][0] = color[n][2]
                img[x][y][1] = color[n][1]
                img[x][y][2] = color[n][0]
        # img=cv2.resize(img,(size[1]*2, size[0]*2))
        # cv2.imshow("2",cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        # print(img.shape)
        # cv2.imshow("spectrogram", img)
        # cv2.imshow("2", img[0:535,511:1022])
        # cv2.waitKey(0)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    # 建立時間範圍內的地圖矩陣
    def __createMapMatrixTimeRange(self, df, boundary, start_time=None, end_time=None):
        matrix = self.__createMapMatrix(boundary)
        self.__punctuation(df, matrix, boundary, start_time, end_time)
        print(matrix)
        self.createSpectrogram(matrix, 1)
        return matrix

    # 判斷圓形之特徵矩陣
    def __eigenvalueMatrix(self, radius):
        array_size = radius * 2 + 1
        original_array = np.zeros((array_size, array_size), dtype=int)
        center_x, center_y = array_size // 2, array_size // 2

        for i in range(array_size):
            for j in range(array_size):
                if np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2) <= radius:
                    original_array[i, j] = 1
        return original_array

    # 計算特徵值
    def __convolution(self, matrix_1, matrix_2, radius):
        ans = 0
        for i in range(radius * 2 + 1):
            for j in range(radius * 2 + 1):
                ans += matrix_1[i][j] * matrix_2[i][j]
        return ans

    # 計算特徵值矩陣
    def __createFeatureMatrix(
        self, matrix, padding_matrx, drone_coverage_matrix, radius
    ):
        feature_matrix = np.zeros((matrix.shape))
        size = np.shape(matrix)
        for i in range(size[0]):
            for j in range(size[1]):
                feature_matrix[i][j] = self.__convolution(
                    drone_coverage_matrix,
                    padding_matrx[i : i + radius * 2 + 1, j : j + radius * 2 + 1],
                    radius,
                )
            print(str(i // (size[0] / 100)) + "%")
        return feature_matrix

    # 搜尋特徵值矩陣最大的點
    def __searchMaxPoint(self, matrix):
        max = [0, 0, 0]
        size = np.shape(matrix)
        for x in range(size[0]):
            for y in range(size[1]):
                if matrix[x][y] > max[2]:
                    max[0] = x
                    max[1] = y
                    max[2] = matrix[x][y]
        return max

    def __matrixAreaZero(self, matrix, x, y, zero_matrix, radius):
        size = np.shape(matrix)
        small_long = x - radius

        if small_long < 0:
            small_long = 0

        long_long = x + radius + 1

        if long_long > size[0]:
            long_long = size[0]

        small_tail = y - radius

        if small_tail < 0:
            small_tail = 0

        long_tail = y + radius + 1

        if long_tail > size[1]:
            long_tail = size[1]

        for i in range(small_long, long_long):
            for j in range(small_tail, long_tail):
                matrix[i][j] = (
                    matrix[i][j] * zero_matrix[i - small_long][j - small_tail]
                )
                if zero_matrix[i - small_long][j - small_tail] == 0:
                    self.area_matrix[i][j] = 1
                elif self.area_matrix[i][j] != 1:
                    self.area_matrix[i][j] = 0

        self.matrix_changes.append(self.createSpectrogram(matrix, 1))

    # 矩陣重新計算
    def __featrueMatrixAreaRefresh(self, matrix, featrue_matrix, x, y, radius):
        size = np.shape(matrix)
        small_long = max(x - radius * 2, 0)
        long_long = min(x + radius * 2 + 1, size[0])
        small_tail = max(y - radius * 2, 0)
        long_tail = min(y + radius * 2 + 1, size[1])

        identity_matrix = self.__eigenvalueMatrix(radius)
        padding_matrix = np.pad(
            matrix, pad_width=radius, mode="constant", constant_values=0
        )
        print(matrix.shape, featrue_matrix.shape)
        for i in range(small_long, long_long):
            for j in range(small_tail, long_tail):
                # featrue_matrix[i][j] = new_matrix[i - small_long][j - small_tail]
                featrue_matrix[i][j] = self.__convolution(
                    identity_matrix,
                    padding_matrix[i : i + radius * 2 + 1, j : j + radius * 2 + 1],
                    radius,
                )
        self.featrue_matrix_changes.append(self.createSpectrogram(featrue_matrix, 10))

    # 部屬點計算
    def __point(self, matrix, feature_matrix, radius):
        quantity = self.quantity
        drone_location = []
        identity_matrix = self.__eigenvalueMatrix(radius)
        # print(identity_matrix)
        zero_matrix = np.where(identity_matrix == 0, 1, 0)
        # print(zero_matrix)

        for i in range(int(quantity)):
            max_point = self.__searchMaxPoint(feature_matrix)
            if max_point[2] < self.Features_lowest:
                break
            self.num += 1
            drone_location.append(max_point)
            print(max_point)
            self.__matrixAreaZero(
                matrix, max_point[0], max_point[1], zero_matrix, radius
            )
            # self.create_spectrogram(matrix, 0.1)
            self.__featrueMatrixAreaRefresh(
                matrix,
                feature_matrix,
                drone_location[i][0],
                drone_location[i][1],
                self.radius,
            )
            # self.create_spectrogram(feature_matrix, 10)

        return drone_location

    # 找尋部屬點
    def __deploymentPoint(self):
        # 特徵值矩陣
        eigenvalue_matrix = self.__eigenvalueMatrix(self.radius)
        # self.create_spectrogram(identity_matrix, 1)

        padding_matrx = np.pad(
            self.matrix, pad_width=self.radius, mode="constant", constant_values=0
        )
        # self.create_spectrogram(padding_matrx, 1)

        feature_matrix = self.__createFeatureMatrix(
            self.matrix, padding_matrx, eigenvalue_matrix, self.radius
        )
        self.featrue_matrix_changes.append(self.createSpectrogram(feature_matrix, 10))

        """
        csv_file = "matrix.csv"
        df = pd.read_csv(csv_file)
        feature_matrix = df.to_numpy()
        """
        print(self.matrix.shape, feature_matrix.shape)

        # self.create_spectrogram(feature_matrix, 10)
        np.savetxt("matrix.csv", feature_matrix, delimiter=",", fmt="%d")
        deployment_point = self.__point(self.matrix, feature_matrix, self.radius)
        print(deployment_point)
        print(np.shape(deployment_point))
        return deployment_point

    def __accuracyCalculation(self):
        self.__createMatrix()
        point = self.__deploymentPoint()
        end_point = [[0, 0, 0.0] for _ in range(len(point))]
        test_point = [
            [lon, lat] for lon, lat in zip(self.test_df["GPS經度"], self.test_df["GPS緯度"])
        ]
        counter = len(test_point)

        for i in range(len(point)):
            end_point[i][0] = round(point[i][0] / 1000 + self.boundary[2], 3)
            end_point[i][1] = round(point[i][1] / 1000 + self.boundary[3], 3)
            end_point[i][2] = point[i][2]

        for i in range(len(test_point)):
            for j in range(len(end_point)):
                coord1 = (end_point[j][1], end_point[j][0])
                coord2 = (test_point[i][1], test_point[i][0])
                distance_km = geodesic(coord1, coord2).kilometers
                if distance_km <= 1:
                    counter -= 1
                    break

        np.savetxt("area_matrix.csv", self.area_matrix, delimiter=",", fmt="%d")
        Probability = counter / len(test_point)
        Probability = round((1 - Probability) * 100, 5)
        self.end_point = end_point
        self.Probability = Probability
        print("", end_point)
        print(Probability, "%")
        self.calculateArea()
        self.__creatAccidentsListImg()

    def __creatAccidentsListImg(self):
        x = list(range(24))
        fig = plt.figure(figsize=(5.5, 5.5))
        plt.bar(x, self.accidents_list)
        plt.xlabel("time")
        plt.ylabel("Number of car accidents")
        plt.title("Distribution of car accidents in different time periods")
        plt.savefig("AccidentsListImg.png")

    def calculateArea(self):
        for row in self.area_matrix:
            for element in row:
                self.total_sum += element

        self.Area = self.total_sum * 0.1 * 0.1
        print(self.Area)

    # 計算
    def calculate(self):
        self.__accuracyCalculation()

    # 輸出無人機數量
    def outNumberDrones(self):
        return self.num

    # 輸出地圖矩陣
    def outputMatrixChanges(self, i):
        return self.matrix_changes[i]

    # 輸出特徵值矩陣
    def outputFeatrueMatrixChanges(self, i):
        return self.featrue_matrix_changes[i]

    # 輸出部屬點
    def outEndPoint(self):
        return self.end_point

    # 輸出覆蓋率
    def outputProbability(self):
        return self.Probability

    # 輸出覆蓋面積
    def outCoverageArea(self):
        return self.Area

    # 輸出覆蓋面積在城市占比
    def outputProportionAreaCity(self):
        area = self.Area / self.city_area * 100
        return area

    # 輸出Googlemap圖片url
    def outputImgWebUrl(self, key, num=None):
        center = [23.16, 120.35]
        zoom = 10
        size = [470, 470]
        maker = "markers=size:tiny|Ccolor:red|23.229,120.348"
        url = "https://maps.googleapis.com/maps/api/staticmap?"
        if num == None:
            url = url + "center=23.16,120.35" + "&" + "zoom=10" + "&" + "size=470x470"
            for i in self.end_point:
                url = url + "&" + "markers="
                url = url + "size:tiny"
                url = url + "|" + "color:red"
                url = url + "|" + str(i[1]) + "," + str(i[0])
        else:
            for index, i in enumerate(self.end_point):
                if index == num:
                    url = (
                        url
                        + "center="
                        + str(i[1])
                        + ","
                        + str(i[0])
                        + "&"
                        + "zoom=14"
                        + "&"
                        + "size=470x470"
                    )
                    url = (
                        url
                        + "&markers=size:mid|color:red|"
                        + str(i[1])
                        + ","
                        + str(i[0])
                    )
        url = url + "&" + "key=" + key
        return url

    def outputAreaMatrixImg(self):
        return self.createSpectrogram(self.area_matrix, 0.1)


def main():
    file_path = r"臺南市112年上半年道路交通事故原因傷亡統計.csv"
    test = Feature_value_judgment()

    test.inputFile(file_path)
    test.inputStarttime(23)
    test.inputEndtime(5)
    test.inputDroneSpeed(60)
    test.inputQuantity(100)
    test.inputFeaturesLowest(60)
    test.inputCityArea(2192)
    test.calculate()
    print(test.outNumberDrones())
    test.outputMatrixChanges(1)
    test.outputFeatrueMatrixChanges(1)
    test.outputProportionAreaCity()
    print(test.outEndPoint())
    print(test.outputProbability())
    print(test.accidents_list)
    # print(test.outputImgWebUrl("AIzaSyDwJ3GEiiLnMB-t-Mx7LzejCYXLW4pNYRo"))


if __name__ == "__main__":
    main()
