import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
import pandas as pd
import math

class Kmeans:
    def __init__(self) -> None:
        self.k = 2
        self.file = None
        self.kmeans = None
        self.start_time=0
        self.end_time = 0
        self.radius = 0
        self.kdtree = 0
    
    def inputFile(self, file):

        df = pd.read_csv(file, encoding="utf-8")
        '''
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
        '''
    # 輸入無人機時速
    def inputDroneSpeed(self, speed):
        self.radius = int(speed / 60)
        
    # 輸入開始時間
    def inputStarttime(self, start_time):
        self.start_time = start_time

    # 輸入結束時間
    def inputEndtime(self, end_time):
        self.end_time = end_time

    # 輸入城市總面積
    def inputCityArea(self, city_area):
        self.city_area = city_area
    '''
    def __judgmentTime(self, i, start_time=None, end_time=None):
        self.accidents_list[i // 10000] += 1
        if start_time == None or end_time == None:
            return True

        # 將時間值轉換為小時、分鐘和秒
        hour, minute, second = (
            i // 10000,
            math.floor(i / 10000 % 1 * 100),
            i % 100,
        )
    
        # 確保開始時間小於結束時間
        if start_time > end_time:
            return (hour >= start_time and hour < 24) or (
                hour <= end_time and hour >= 0
            )

        # 判斷時間是否在範圍內
        return start_time <= hour <= end_time
    '''
    def build_kd_tree(self,Centroid):
        
        # 將 EndPoint 轉換為 NumPy 陣列，並建立 KD-Tree
        sliced_list = Centroid
        # 將串列轉換為 NumPy 陣列
        points = np.array(sliced_list)
        self.kdtree = cKDTree(points)
    
    def __judgeDistance(self,Centroid,Cluster):

        if self.kdtree is None:
            self.build_kd_tree(Cluster)

        for i in range(0,len(Centroid)):
            distances, indices = self.kdtree.query(Centroid[i], k=1, distance_upper_bound=None)
            if distances > self.radius* 0.01 :
                return True
            
        return False
    

    def k_means(self,data, k, max_iterations=100, tolerance=0.001):
    # 步骤1：初始化聚类中心
        centroids = np.array(self.randomly_initialize_centroids(data, k))
    
        for iteration in range(max_iterations):
            # 步骤2：分配数据点到最近的聚类中心
            clusters = self.assign_to_clusters(data, centroids)
        
            # 步骤3：更新聚类中心
            new_centroids = self.update_centroids(clusters)
        
            # 检查聚类中心是否变化不大
            if self.centroids_converged(centroids, new_centroids, tolerance):
                break
        
            centroids = new_centroids
    
        return clusters, centroids

    # 辅助函数：初始化聚类中心
    def randomly_initialize_centroids(self,data, k):
        indices = np.random.choice(len(data), k, replace=False)
        return [data[i] for i in indices]

    # 辅助函数：分配数据点到最近的聚类中心
    def assign_to_clusters(self,data, centroids):
        clusters = [[] for _ in range(len(centroids))]
    
        for point in data:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            closest_centroid_index = np.argmin(distances)
            clusters[closest_centroid_index].append(point)
    
        return clusters

    # 辅助函数：更新聚类中心
    def update_centroids(self,clusters):
        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]
        return new_centroids

    # 辅助函数：检查聚类中心是否变化不大
    def centroids_converged(self,old_centroids, new_centroids, tolerance):
        return all(np.linalg.norm(old - new) < tolerance for old, new in zip(old_centroids, new_centroids))


    def calculate(self):
        clusters, centroids = self.k_means(self.file,self.k,10000,self.radius* 0.01)
        while(self.__judgeDistance(centroids,clusters)):
            self.k += 1
            clusters, centroids = self.k_means(self.file,self.k,10000,self.radius* 0.01)

        return clusters, centroids

def main():
    test = Kmeans()
    file_path = r"臺南市112年上半年道路交通事故原因傷亡統計.csv"
    test.inputFile(file_path)
    test.inputCityArea(2192)
    test.inputEndtime(24)
    test.inputStarttime(0)
    test.inputDroneSpeed(45)
    clusters, centroids = test.calculate()
    print(clusters, centroids)    

if __name__=='main':
    main()


    