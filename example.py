import random
from shapely.geometry import Point
from shapely.ops import cascaded_union
import geopy.distance

# 生成隨機的經緯度座標點
lon1 = random.uniform(-180, 180)
lat1 = random.uniform(-90, 90)
lon2 = random.uniform(-180, 180)
lat2 = random.uniform(-90, 90)

# 顯示隨機生成的座標點
print("隨機生成的座標點1：({}, {})".format(lon1, lat1))
print("隨機生成的座標點2：({}, {})".format(lon2, lat2))

# 計算兩座標之間的距離（單位：公尺）
distance = geopy.distance.distance((lat1, lon1), (lat2, lon2)).m

# 定義圓的半徑（一公里）
radius = 1000

# 使用Shapely庫建立兩個圓的Polygon對象
circle1 = Point(lon1, lat1).buffer(radius)
circle2 = Point(lon2, lat2).buffer(radius)

# 計算兩個圓的交集
intersection = circle1.intersection(circle2)

# 計算交集面積
intersection_area = intersection.area

# 計算兩個圓的聯集面積
union_area = cascaded_union([circle1, circle2]).area

# 打印結果
print("兩圓相交的聯集面積：", union_area)
print("兩圓相交的交集面積：", intersection_area)