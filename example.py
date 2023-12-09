import big_data as bd
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
from sklearn.model_selection import train_test_split

class Drone_deployment(bd.Feature_value_judgment):
    def __init__(self):
        super().__init__()
        self.EndPoint = []
        self.file = None
        self.kdtree = None
        self.serch_radius = 0

    def computingHotspots(self,file,DroneSpeed):

        self.inputDroneSpeed(DroneSpeed)
        self.inputQuantity(1000)
        self.inputFeaturesLowest(60)
        self.inputStarttime (0) 
        self.inputEndtime(24)
        self.serch_radius = self.radius / 10
        
        self.file = file
        df = pd.read_csv(self.file, encoding="utf-8")
        df = df[df['發生時間'].apply(lambda x: self._Feature_value_judgment__judgmentTime(x,start_time = self.start_time, end_time = self.end_time))]
        self.boundary = self._Feature_value_judgment__findBoundary(df)
        self.train_df, self.test_df = train_test_split(
            df, test_size=0.15, random_state=42
        )
        self.train_df = df
        self.calculate()

        self.EndPoint = self.outEndPoint()
        for i in range(0,len(self.EndPoint)) :        
            self.EndPoint[i][2] = 0

    def build_kd_tree(self):
        
        # 將 EndPoint 轉換為 NumPy 陣列，並建立 KD-Tree
        sliced_list = [(sublist[1],sublist[0]) for sublist in self.EndPoint]
        # 將串列轉換為 NumPy 陣列
        points = np.array(sliced_list)
        print(len(self.EndPoint))
        print(len(points))
        self.kdtree = cKDTree(points)

    def night_time_analysis(self):
        EndPoint = self.EndPoint
        self.inputFeaturesLowest(30)
        if self.kdtree is None:
            self.build_kd_tree()
        self.start_time = 23
        self.end_time = 4    
        df = pd.read_csv(self.file, encoding="utf-8")
        df = df[df['發生時間'].apply(lambda x: self._Feature_value_judgment__judgmentTime(x,start_time = self.start_time, end_time = self.end_time))]

        # 要查找的點
        query_points = np.array([(lat, lon) for lon, lat in zip(df["GPS經度"], df["GPS緯度"])])
        print(len(query_points))
        query_points = cKDTree(query_points)
        # 使用 KD-Tree 查找在1公里內的點
        indices = self.kdtree.query_ball_tree(query_points, r=self.serch_radius*0.01)
        counter = 0
        for i in range( 0 , len(indices)):
            for j in range(0,len(indices[i])) :
                EndPoint[i][2] += 1
                counter += 1
        print(EndPoint)
                
    def commuting_work_time_analysis(self):
        EndPoint = self.EndPoint
        self.inputFeaturesLowest(30)
        if self.kdtree is None:
            self.build_kd_tree()
        self.start_time = 5
        self.end_time = 8    
        df = pd.read_csv(self.file, encoding="utf-8")
        df = df[df['發生時間'].apply(lambda x: self._Feature_value_judgment__judgmentTime(x,start_time = self.start_time, end_time = self.end_time))]

        # 要查找的點
        query_points = np.array([(lat, lon) for lon, lat in zip(df["GPS經度"], df["GPS緯度"])])
        print(len(query_points))
        query_points = cKDTree(query_points)
        # 使用 KD-Tree 查找在1公里內的點
        indices = self.kdtree.query_ball_tree(query_points, r=self.serch_radius*0.01)
        for i in range( 0 , len(indices)):
            for j in range(0,len(indices[i])) :
                EndPoint[i][2] += 1
                
    def work_time_analysis(self):
        EndPoint = self.EndPoint
        self.inputFeaturesLowest(30)
        if self.kdtree is None:
            self.build_kd_tree()
        self.start_time = 9
        self.end_time = 15    
        df = pd.read_csv(self.file, encoding="utf-8")
        df = df[df['發生時間'].apply(lambda x: self._Feature_value_judgment__judgmentTime(x,start_time = self.start_time, end_time = self.end_time))]

        # 要查找的點
        query_points = np.array([(lat, lon) for lon, lat in zip(df["GPS經度"], df["GPS緯度"])])
        print(len(query_points))
        query_points = cKDTree(query_points)
        # 使用 KD-Tree 查找在1公里內的點
        indices = self.kdtree.query_ball_tree(query_points, r=self.serch_radius*0.01)
        for i in range( 0 , len(indices)):
            for j in range(0,len(indices[i])) :
                EndPoint[i][2] += 1
                
    def commuting_off_work_time_analysis(self):
        EndPoint = self.EndPoint
        self.inputFeaturesLowest(30)
        if self.kdtree is None:
            self.build_kd_tree()
        self.start_time = 16
        self.end_time = 18    
        df = pd.read_csv(self.file, encoding="utf-8")
        df = df[df['發生時間'].apply(lambda x: self._Feature_value_judgment__judgmentTime(x,start_time = self.start_time, end_time = self.end_time))]

        # 要查找的點
        query_points = np.array([(lat, lon) for lon, lat in zip(df["GPS經度"], df["GPS緯度"])])
        print(len(query_points))
        query_points = cKDTree(query_points)
        # 使用 KD-Tree 查找在1公里內的點
        indices = self.kdtree.query_ball_tree(query_points, r=self.serch_radius*0.01)
        for i in range( 0 , len(indices)):
            for j in range(0,len(indices[i])) :
                EndPoint[i][2] += 1
                
    def Leisure_time_analysis(self):
        EndPoint = self.EndPoint
        self.inputFeaturesLowest(30)
        if self.kdtree is None:
            self.build_kd_tree()
        self.start_time = 19
        self.end_time = 22    
        df = pd.read_csv(self.file, encoding="utf-8")
        df = df[df['發生時間'].apply(lambda x: self._Feature_value_judgment__judgmentTime(x,start_time = self.start_time, end_time = self.end_time))]

        # 要查找的點
        query_points = np.array([(lat, lon) for lon, lat in zip(df["GPS經度"], df["GPS緯度"])])
        print(len(query_points))
        query_points = cKDTree(query_points)
        # 使用 KD-Tree 查找在1公里內的點
        indices = self.kdtree.query_ball_tree(query_points, r=self.serch_radius*0.01)
        for i in range( 0 , len(indices)):
            for j in range(0,len(indices[i])) :
                EndPoint[i][2] += 1
                

def main():
    file_path = r"臺南市112年上半年道路交通事故原因傷亡統計.csv"
    test = Drone_deployment()
    test.computingHotspots(file_path,45)
    test.night_time_analysis()
if __name__=="__main__":
    main()

        

   
    


