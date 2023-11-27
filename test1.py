import numpy as np



def main():
    size = [10,10]
    long_long = 5
    small_long = 5
    long_tail = 5
    small_tail = 5
    radius = 2
    # 分布圖
    matrix = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 2, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 4, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
            [0, 1, 0, 0, 1, 0, 0, 1, 2, 2],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        ]
    )
    padding_matrx = np.zeros((max(long_long +radius,size[0])  - min(small_long-radius,0), max(long_tail +radius,size[1]) - min(small_tail-radius,0)), dtype=int)

    print(small_long,small_tail,long_long,long_tail,radius,size)

    for i in range(max(small_long-radius,0),min(long_long+radius,size[0])):
        for j in range(max(small_tail-radius,0),min(long_tail+radius,size[1])):
            padding_matrx[max(i - small_long-radius,0)][max(i - small_tail-radius,0)] = matrix[i][j]
            print(max(i - small_long-radius,0),max(i - small_tail-radius,0))

    print(padding_matrx)

if __name__==main():
    main()