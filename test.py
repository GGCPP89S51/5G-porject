import numpy as np

def k_means(data, k, max_iterations=100, tolerance=0.001):
    # 步骤1：初始化聚类中心
    centroids = np.array(randomly_initialize_centroids(data, k))
    
    for iteration in range(max_iterations):
        # 步骤2：分配数据点到最近的聚类中心
        clusters = assign_to_clusters(data, centroids)
        
        # 步骤3：更新聚类中心
        new_centroids = update_centroids(clusters)
        
        # 检查聚类中心是否变化不大
        if centroids_converged(centroids, new_centroids, tolerance):
            break
        
        centroids = new_centroids
    
    return clusters, centroids

# 辅助函数：初始化聚类中心
def randomly_initialize_centroids(data, k):
    indices = np.random.choice(len(data), k, replace=False)
    return [data[i] for i in indices]

# 辅助函数：分配数据点到最近的聚类中心
def assign_to_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    
    for point in data:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        closest_centroid_index = np.argmin(distances)
        clusters[closest_centroid_index].append(point)
    
    return clusters

# 辅助函数：更新聚类中心
def update_centroids(clusters):
    new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]
    return new_centroids

# 辅助函数：检查聚类中心是否变化不大
def centroids_converged(old_centroids, new_centroids, tolerance):
    return all(np.linalg.norm(old - new) < tolerance for old, new in zip(old_centroids, new_centroids))

if __name__ == "__main__":
    # 生成示例数据（替换成你的实际数据）
    data = np.random.rand(100, 2)  # 100个二维数据点
    
    # 设置K值和其他参数
    k = 3
    max_iterations = 100
    tolerance = 0.001
    
    # 调用K均值聚类函数
    clusters, centroids = k_means(data, k, max_iterations, tolerance)
    
    # 打印结果或进行其他后续处理
    print("Final Clusters:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")

    print("\nFinal Centroids:")
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i + 1}: {centroid}")