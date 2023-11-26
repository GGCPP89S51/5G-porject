import numpy as np

def convolution(matrix_1, matrix_2,radius):
        ans = 0
        for i in range(radius*2+1):
            for j in range(radius*2+1):
                ans += matrix_1[i][j] * matrix_2[i][j]
        return ans

def create_feature_matrix(matrix, padding_matrx, drone_coverage_matrix,radius):
    feature_matrix = np.zeros((matrix.shape))
    for i in range(10):
        for j in range(10):
            feature_matrix[i][j] = convolution(
                drone_coverage_matrix, padding_matrx[i : i + radius*2+1, j : j + radius*2+1],radius
            )
    return feature_matrix

def Identity_matrix(radius):
        array_size = radius * 2 + 1
        original_array = np.zeros((array_size, array_size), dtype=int)
        center_x, center_y = array_size // 2, array_size // 2

        for i in range(array_size):
            for j in range(array_size):
                if np.sqrt((i - center_x)**2 + (j - center_y)**2) <= radius:
                    original_array[i, j] = 1
        return original_array
    
def matrix_area_zero(matrix,x,y,zero_matrix,radius):
        small_long = x - radius
        long_long = x + radius +1
        small_tail = y - radius
        long_tail = y + radius +1
        for i in range(small_long,long_long):
            for j in range(small_tail,long_tail):
                matrix[i][j] = matrix[i][j] * zero_matrix[i - small_long][j - small_tail]

def main():
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
    
    identity_matrx = Identity_matrix(2)
    print(identity_matrx)
    padding_matrx = np.pad(matrix, pad_width=3, mode='constant', constant_values=0)
    print(padding_matrx)    
    feature_matrix = create_feature_matrix(matrix, padding_matrx, identity_matrx,2)
    print(feature_matrix)
    zero_matrix = np.where(identity_matrx == 0, 1, 0)
    print(zero_matrix)
    matrix_area_zero(feature_matrix,2,2,zero_matrix,2)
    print(feature_matrix)
if __name__==main():
    main()