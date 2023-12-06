import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 创建一个10x10的随机矩阵
np.random.seed(42)
data = np.random.rand(10, 10)

# 将其中几个点标记为整数，其余为0
data[2, 3] = data[5, 7] = 5.0
data[8, 1] = 8.0

# 打印原始矩阵
print("Original Matrix:")
print(data)

# 将矩阵转换为一维数组
data_flat = data.flatten().reshape(-1, 1)

# 使用 K-MinMeans 进行聚类，设置最小的 K 值为2
k_min = 6
kmeans = KMeans(n_clusters=k_min, random_state=42)
kmeans.fit(data_flat)

# 获取每个数据点所属的簇
labels = kmeans.labels_

# 将标签重塑回矩阵形状
labels_matrix = labels.reshape(data.shape)
print(labels_matrix)
# 绘制原始矩阵
plt.subplot(1, 2, 1)
plt.imshow(data, cmap='viridis', interpolation='nearest')
plt.title('Original Matrix')

# 绘制聚类结果
plt.subplot(1, 2, 2)
plt.imshow(labels_matrix, cmap='viridis', interpolation='nearest')
plt.title(f'K-MinMeans Clustering (K={k_min})')

plt.show()

