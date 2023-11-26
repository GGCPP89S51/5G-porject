import numpy as np


# 將矩陣各點相乘後相加
def convolution(matrix_1, matrix_2):
    ans = 0
    for i in range(5):
        for j in range(5):
            ans += matrix_1[i][j] * matrix_2[i][j]
    return ans


# 找出矩陣最大的點並回傳位置及數值
def matrix_max_point(matrix):
    size = np.shape(matrix)
    max = [0, 0, 0]
    for i in range(size[0]):
        for j in range(size[1]):
            if max[2] < matrix[i][j]:
                max = [i, j, matrix[i][j]]
    return max


# 生成無人機覆蓋範圍矩陣
def create_drone_coverage_matrix(distance):
    size = distance * 2 + 1
    matrix = np.ones((size, size))
    distance_square = distance * distance
    for i in range(size):
        for j in range(size):
            if (
                (i - distance) * (i - distance) + (j - distance) * (j - distance)
            ) > distance_square:
                matrix[i][j] = 0
    return matrix


# 生成填充矩陣
def create_padding_matrix(matrix, padding):
    size = np.shape(matrix)
    padding_matrix = np.zeros((size[0] + padding * 2, size[1] + padding * 2))
    for i in range(padding, padding + size[0]):
        for j in range(padding, padding + size[1]):
            padding_matrix[i][j] = matrix[i - padding][j - padding]
    return padding_matrix


# 生成特徵矩陣
def create_feature_matrix(matrix, padding_matrx, drone_coverage_matrix):
    feature_matrix = np.zeros((matrix.shape))
    for i in range(10):
        for j in range(10):
            feature_matrix[i][j] = convolution(
                drone_coverage_matrix, padding_matrx[i : i + 5, j : j + 5]
            )
    return feature_matrix


# 更新填充矩陣
def refresh_padding_matrix(padding_matrix):
    return padding_matrix


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

    # 捲積核
    kernel = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ]
    )

    # 生成填充矩陣
    matrix_padding = np.zeros((14, 14))
    for i in range(2, 11):
        for j in range(2, 11):
            matrix_padding[i][j] = matrix[i - 2][j - 2]

    # 生成空白的特徵圖
    feature_matrix = np.zeros((10, 10))

    # 計算特徵圖
    for i in range(10):
        for j in range(10):
            feature_matrix[i][j] = convolution(
                kernel, matrix_padding[i : i + 5, j : j + 5]
            )

    # 找出佈署的位置並存入串列
    max_points = []

    max_points.append(matrix_max_point(feature_matrix))

    # print(matrix)
    # print(kernel)
    # print(matrix_padding)
    print(feature_matrix)
    print(max_points)

    matrix_2 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 1, 2, 2],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        ]
    )

    # 生成填充矩陣
    matrix_padding_2 = np.zeros((14, 14))
    for i in range(2, 11):
        for j in range(2, 11):
            matrix_padding_2[i][j] = matrix_2[i - 2][j - 2]

    # 生成空白的特徵圖
    feature_matrix_2 = np.zeros((10, 10))

    # 計算特徵圖
    for i in range(10):
        for j in range(10):
            feature_matrix_2[i][j] = convolution(
                kernel, matrix_padding_2[i : i + 5, j : j + 5]
            )

    print(feature_matrix_2)


def test():
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
    padding_matrx = create_padding_matrix(matrix, 2)
    drone_coverage_matrix = create_drone_coverage_matrix(2)
    feature_matrix = create_feature_matrix(matrix, padding_matrx, drone_coverage_matrix)
    print(feature_matrix)


if __name__ == "__main__":
    test()
