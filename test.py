import random

def objective_function(x, y):
    return x**2 + y**2

def bee_algorithm(max_iter, num_bees, elite_sites, patch_size):
    bees = [{'x': random.uniform(-5, 5), 'y': random.uniform(-5, 5)} for _ in range(num_bees)]

    for iteration in range(max_iter):
        fitness_values = [objective_function(bee['x'], bee['y']) for bee in bees]
        elite_bees = sorted(range(num_bees), key=lambda k: fitness_values[k])[:elite_sites]

        for elite_bee in elite_bees:
            new_x = bees[elite_bee]['x'] + random.uniform(-patch_size, patch_size)
            new_y = bees[elite_bee]['y'] + random.uniform(-patch_size, patch_size)

            if objective_function(new_x, new_y) < objective_function(bees[elite_bee]['x'], bees[elite_bee]['y']):
                bees[elite_bee]['x'] = new_x
                bees[elite_bee]['y'] = new_y

    best_bee = min(bees, key=lambda bee: objective_function(bee['x'], bee['y']))
    return best_bee

if __name__ == "__main__":
    max_iterations = 100
    num_bees = 20
    elite_sites = 5
    patch_size = 0.5

    best_solution = bee_algorithm(max_iter=max_iterations, num_bees=num_bees, elite_sites=elite_sites, patch_size=patch_size)

    print("最优解：", best_solution)
    print("目标函数值：", objective_function(best_solution['x'], best_solution['y']))