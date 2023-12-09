from scipy.spatial import cKDTree
import numpy as np

# 假設 coords 是一個包含經度和緯度的二維 NumPy 陣列，並已經建立了 KD-Tree
coords = np.array([
    [22.995, 120.204],
    [22.986, 120.226],
    # ... 其他座標 ...
])
kdtree = cKDTree(coords)

# 要查找的點的座標
query_points = np.array([
    [23.0, 120.2],
    [23.1, 120.3],
    # ... 其他查詢點 ...
])

# 使用 KD-Tree 查找在1公里內的點
indices = kdtree.query_ball_tree(cKDTree(query_points), r=1.0)

# 打印結果
for i, query_point_indices in enumerate(indices):
    print(f"Indices for Query Point {i + 1}: {query_point_indices}")