import pandas as pd
import numpy as np
import math
import cv2

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
            if hazard_distribution_array[x][y] / 1 < 20:
                n = hazard_distribution_array[x][y] // 1
            elif hazard_distribution_array[x][y]== 0:
                img[x][y][0] = 255
                img[x][y][1] = 255
                img[x][y][2] = 255
            else:
                n = 20
            img[x][y][0] = color[n][2]
            img[x][y][1] = color[n][1]
            img[x][y][2] = color[n][0]
    cv2.imshow("spectrogram", img)
    cv2.waitKey(0)

file_path = (
    r"C:\Users\s0901\Downloads\20a0110c-525e-4138-ae1a-d352c09beca5.csv"
)
df = pd.read_csv(file_path, encoding="utf-8")
boundary=Find_the_boundary(df)
print(boundary)
matrix = Create_map_matrix(boundary)
punctuation(df,matrix,boundary)
print(matrix)
create_spectrogram(matrix)