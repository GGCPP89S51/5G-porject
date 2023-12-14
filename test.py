import numpy as np
import random

# 生成随机的需求点
num_demands = 10
demand_points = np.random.rand(num_demands, 2) * 100  # 二维空间中的需求点

# 蜂群算法函数
def bee_algorithm_facility_location(max_iter, num_bees, patch_size):
    # 初始化蜜蜂群体
    bees = [{'locations': np.random.rand(2) * 100} for _ in range(num_bees)]

    for iteration in range(max_iter):
        # 评估蜜蜂适应度
        fitness_values = [np.min(np.linalg.norm(bee['locations'] - demand_points, axis=1)) for bee in bees]

        # 选择蜜蜂
        elite_bee = bees[np.argmin(fitness_values)]

        # 信息传递和局部搜索
        new_locations = elite_bee['locations'] + np.random.uniform(-patch_size, patch_size, 2)

        # 更新解
        if np.min(np.linalg.norm(new_locations - demand_points, axis=1)) < np.min(np.linalg.norm(elite_bee['locations'] - demand_points, axis=1)):
            elite_bee['locations'] = new_locations

    # 返回最优解
    best_bee = min(bees, key=lambda bee: np.min(np.linalg.norm(bee['locations'] - demand_points, axis=1)))
    return best_bee

if __name__ == "__main__":
    max_iterations = 100
    num_bees = 20
    patch_size = 5.0

    best_solution = bee_algorithm_facility_location(max_iter=max_iterations, num_bees=num_bees, patch_size=patch_size)

    print("最优设施位置：", best_solution['locations'])
    print("目标函数值：", np.min(np.linalg.norm(best_solution['locations'] - demand_points, axis=1)))
