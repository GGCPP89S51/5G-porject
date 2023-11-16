import pandas as pd
import numpy as np
import math
import cv2
import folium

def Find_the_boundary(df):
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
    boundary = [max_lon , max_lat , min_lon ,min_lat]
    return  boundary

def Create_map_matrix(boundary) :
    boundarys = list(boundary)
    boundarys[0] = math.ceil(boundarys[0] * 1000) / 1000 
    boundarys[1] = math.ceil(boundarys[1] * 1000) / 1000 
    boundarys[2] = math.floor(boundarys[2] * 1000) / 1000 
    boundarys[3] = math.floor(boundarys[3] * 1000) / 1000 
    long  = math.floor((boundarys[1] - boundarys[3])*1000)
    wight = math.floor((boundarys[0] - boundarys[2])*1000)
    matrix = np.zeros((wight+1, long+1), dtype=int)
    return matrix

def punctuation(df,matrix,boundary):
    boundarys = list(boundary)
    min_wight =math.floor(boundarys[2] * 1000) / 1000
    min_long =math.floor(boundarys[3] * 1000) / 1000
    for lon, lat in zip(df['GPS經度'], df['GPS緯度']):
        long = round(lon - min_wight,3)*1000
        wight = round(lat - min_long,3)*1000
        matrix[int(long)][int(wight)] += 1
    return 0

def create_spectrogram(hazard_distribution_array):
    size = np.shape(hazard_distribution_array)
    img = np.zeros((size[0], size[1], 3), np.uint8)
    color = [
        [255,255,255],
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
            if hazard_distribution_array[x][y]/10 < 20:
                n = hazard_distribution_array[x][y]//10
            else:
                n = 20
            img[x][y][0] = color[n][2]
            img[x][y][1] = color[n][1]
            img[x][y][2] = color[n][0]
    #img=cv2.resize(img,(size[1]*2, size[0]*2))
    #cv2.imshow("2",cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    print(img.shape)
    cv2.imshow("spectrogram", img)
    #cv2.imshow("2", img[0:535,511:1022])
    cv2.waitKey(0)

def creat_featrue_matrix(matrix,feature_matrix):
    size=np.shape(matrix)
    for x in range(size[0]):
        for y in range(size[1]):
            feature_matrix[x][y]=feature_matrix_point(matrix,x,y)

def feature_matrix_point(matrix,x,y):
    num=0
    size=np.shape(matrix)
    if(x-10<0):
        x_1=0
    else:
        x_1=x-10
    if(x+10>size[0]):
        x_2=size[0]
    else:
        x_2=x+10
    if(y-10<0):
        y_1=0
    else:
        y_1=y-10
    if(y+10>size[1]):
        y_2=size[1]
    else:
        y_2=y+10

    for i in range(x_1,x_2):
        for j in range(y_1,y_2):
            num+=matrix[i][j]
    return num

def search_max_point(matrix):
    max=[0,0,0]
    size=np.shape(matrix)
    for x in range(size[0]):
        for y in range(size[1]):
            if(matrix[x][y]>max[2]):
                max[0]=x
                max[1]=y
                max[2]=matrix[x][y]
    return max
            
def matrix_area_zero(matrix,x,y):
    size=np.shape(matrix)
    if(x-10<0):
        x_1=0
    else:
        x_1=x-10
    if(x+10>size[0]):
        x_2=size[0]
    else:
        x_2=x+10
    if(y-10<0):
        y_1=0
    else:
        y_1=y-10
    if(y+10>size[1]):
        y_2=size[1]
    else:
        y_2=y+10

    for i in range(x_1,x_2):
        for j in range(y_1,y_2):
            matrix[i][j]=0

def featrue_matrix_area_refresh(matrix,featrue_matrix,x,y):
    size=np.shape(matrix)
    if(x-20<0):
        x_1=0
    else:
        x_1=x-20
    if(x+20>size[0]):
        x_2=size[0]
    else:
        x_2=x+20
    if(y-20<0):
        y_1=0
    else:
        y_1=y-20
    if(y+20>size[1]):
        y_2=size[1]
    else:
        y_2=y+20

    for i in range(x_1,x_2):
        for j in range(y_1,y_2):
            featrue_matrix[i][j]=feature_matrix_point(matrix,i,j)

def main():
    file_path = (
        r"C:\Users\STUST\Downloads\20a0110c-525e-4138-ae1a-d352c09beca5.csv"
    )
    df = pd.read_csv(file_path, encoding="utf-8")
    boundary=Find_the_boundary(df)
    print(boundary)
    matrix = Create_map_matrix(boundary)
    punctuation(df,matrix,boundary)
    print(matrix)
    #create_spectrogram(matrix)
    feature_matrix=Create_map_matrix(boundary)
    creat_featrue_matrix(matrix,feature_matrix)
    create_spectrogram(feature_matrix)
    quantity=input("請輸入無人機數量:")
    drone_location=[]
    mymap = folium.Map(location=[22.9969, 120.213], zoom_start=12)
    j = 0
    for i in range(int(quantity)):
        circle_group = folium.FeatureGroup(name= j )
        max_point = search_max_point(feature_matrix)  
        if max_point[2] < 60 :
            break
        folium.Marker([round((max_point[1]/1000)+boundary[3],3),round((max_point[0]/1000)+boundary[2],3)], popup= j ).add_to(mymap)
        folium.Circle(location = [round((max_point[1]/1000)+boundary[3],3),round((max_point[0]/1000)+boundary[2],3)], radius=1000, color='blue').add_to(circle_group)
        circle_group.add_to(mymap)
        drone_location.append(max_point)
        matrix_area_zero(matrix,drone_location[i][0],drone_location[i][1])
        featrue_matrix_area_refresh(matrix,feature_matrix,drone_location[i][0],drone_location[i][1])
        create_spectrogram(feature_matrix)
        j += 1
    folium.LayerControl().add_to(mymap)
    print(drone_location)
    #for i in range(quantity):
    print(j)
    mymap.save("mymap.html")

if __name__=="__main__":
    main()