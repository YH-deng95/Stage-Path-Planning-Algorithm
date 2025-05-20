import numpy as np
import random
from collections import defaultdict

# 参数设置
n = 3  # 每层网格大小
layers = 3  # 三维网格的层数
total_nodes = n * n * layers  # 总节点数
population_size = 100  # 初始种群大小
generations = 1000  # 遗传算法迭代代数
crossover_rate = 0.8  # 交叉概率
mutation_rate = 0.1  # 变异概率
queue_penalty_weight = 1  # 排队惩罚权重因子
elite_size = 10  # 精英保留的个体数

# 功能节点设置
adc_nodes = [6, 7, 12, 15, 0, 1, 10, 11, 5, 8, 16, 17]  # 采集节点
ddr_nodes = [19, 21, 23]  # 存储节点
transmission_node = 22  # 传输节点

# 定义采集节点与存储节点的对应关系
adc_to_ddr_mapping = {
    19: [6, 7, 12, 15],
    21: [0, 1, 10, 11],
    23: [5, 8, 16, 17]
}


# 创建三维网格邻接矩阵
def create_3d_network(n, layers):
    total_nodes = n * n * layers
    network = np.zeros((total_nodes, total_nodes), dtype=int)

    def node_index(x, y, z):
        return x * n + y + z * n * n

    for x in range(n):
        for y in range(n):
            for z in range(layers):
                node = node_index(x, y, z)
                if x > 0:
                    network[node, node_index(x - 1, y, z)] = 1  # 水平链路
                if x < n - 1:
                    network[node, node_index(x + 1, y, z)] = 1  # 水平链路
                if y > 0:
                    network[node, node_index(x, y - 1, z)] = 1  # 水平链路
                if y < n - 1:
                    network[node, node_index(x, y + 1, z)] = 1  # 水平链路
                if z > 0:
                    network[node, node_index(x, y, z - 1)] = 1  # 垂直链路
                if z < layers - 1:
                    network[node, node_index(x, y, z + 1)] = 1  # 垂直链路
    return network


# 定义个体类
class Individual:
    def __init__(self, path, fitness):
        self.path = path
        self.fitness = fitness


# 生成个体路径
def generate_individual(start, end, network, forbidden_edges=None):
    path = [start]
    current = start
    visited = set(path)
    while current != end:
        neighbors = [i for i in range(total_nodes) if network[current, i] == 1 and i not in visited]
        if forbidden_edges:
            neighbors = [n for n in neighbors if (current, n) not in forbidden_edges]
        if not neighbors:
            return None
        current = random.choice(neighbors)
        path.append(current)
        visited.add(current)
    path.extend([-1] * (total_nodes - len(path)))
    return path


# 计算适应度
def calculate_fitness(path):
    if path is None or len(path) < 2:
        return float('inf')
    fitness = 0
    for node in path:
        if node == -1:
            break
        fitness += 4  # 假定每个节点的基础延时为4
    return fitness


# 去重函数
def remove_duplicates(path):
    seen = set()
    new_path = []
    for node in path:
        if node in seen or node == -1:
            break
        seen.add(node)
        new_path.append(node)
    new_path.extend([-1] * (len(path) - len(new_path)))
    return new_path


# 交叉操作
def crossover(parent1, parent2, network, forbidden_edges=None):
    path1 = parent1.path
    path2 = parent2.path

    # 获取父代路径的有效基因部分（去除无效基因 -1 部分）
    valid_path1 = [node for node in path1 if node != -1]
    valid_path2 = [node for node in path2 if node != -1]

    # 确保路径长度足够进行交叉
    if len(valid_path1) <= 2 or len(valid_path2) <= 2:
        # 如果路径长度不足以进行交叉，直接返回父代作为子代
        return parent1, parent2

    # 提取所有重复基因（包括源节点、目标节点）
    common_nodes = list(set(valid_path1) & set(valid_path2))

    # 排除源节点和目标节点（目标节点为有效基因片段中的最后一个）
    common_nodes = [node for node in common_nodes
                    if node != valid_path1[0]  # 源节点
                    and node != valid_path1[-1]  # 目标节点
                    and node != valid_path2[-1]]  # 确保排除另一个父代的目标节点

    if len(common_nodes) == 1:
        # 单点交叉
        crossover_point = common_nodes[0]
        idx1 = path1.index(crossover_point)
        idx2 = path2.index(crossover_point)
        child1_path = path1[:idx1] + path2[idx2:]
        child2_path = path2[:idx2] + path1[idx1:]
    elif len(common_nodes) > 1:
        # 两点交叉
        point1, point2 = random.sample(common_nodes, 2)
        idx1_1, idx1_2 = path1.index(point1), path1.index(point2)
        idx2_1, idx2_2 = path2.index(point1), path2.index(point2)
        child1_path = path1[:idx1_1] + path2[idx2_1:idx2_2] + path1[idx1_2:]
        child2_path = path2[:idx2_1] + path1[idx1_1:idx1_2] + path2[idx2_2:]
    else:
        # 无公共节点，基于邻接关系拼接路径
        n1 = random.randint(1, len(path1) - 2)
        child1_path = path1[:n1]
        for i in range(1, len(path2)):
            if path2[i] != -1 and network[path1[n1], path2[i]] == 1:
                child1_path += path2[i:]
                break
        child1_path = remove_duplicates(child1_path)
        child2_path = parent2.path

    return Individual(child1_path, float('inf')), Individual(child2_path, float('inf'))


# 变异操作
def mutate(individual, network, forbidden_edges=None):
    path = individual.path[:]

    # 检查路径长度是否足够进行变异操作
    if len(path) <= 2:  # 如果路径长度只有源节点和目的节点，不进行变异
        return individual

    mutation_point = random.randint(1, len(path) - 2)  # 随机选择变异点

    # 获取该变异点的邻接节点
    neighbors = [i for i in range(total_nodes) if
                 network[path[mutation_point - 1], i] == 1 and i != path[mutation_point]]
    if forbidden_edges:
        neighbors = [n for n in neighbors if (path[mutation_point - 1], n) not in forbidden_edges]

    if neighbors:
        path[mutation_point] = random.choice(neighbors)  # 执行变异，替换节点

    if not is_valid_path(path, network):
        return None  # 如果变异后的路径无效，则返回None

    return Individual(path, float('inf'))  # 返回变异后的个体

def calculate_adaptive_rates(individual_fitness, avg_fitness, max_fitness, min_fitness):
    """
    参数:
        individual_fitness: 当前个体的适应度
        avg_fitness: 种群平均适应度
        max_fitness: 种群最大适应度
        min_fitness: 种群最小适应度

    返回:
        crossover_rate: 自适应交叉概率 [0.6, 0.9]
        mutation_rate: 自适应变异概率 [0.01, 0.1]
    """
    # 定义概率区间
    CROSSOVER_MIN = 0.6
    CROSSOVER_MAX = 0.9
    MUTATION_MIN = 0.01
    MUTATION_MAX = 0.1

    # 防止除零
    fitness_range = max(1e-6, max_fitness - min_fitness)
    normalized_fitness = (individual_fitness - min_fitness) / fitness_range

    # 对于适应度高于平均的个体
    if individual_fitness < avg_fitness:
        # 好个体：保持高交叉率，降低变异率
        crossover_rate = CROSSOVER_MAX
        # 变异率随适应度增加线性减小
        mutation_rate = MUTATION_MAX - (MUTATION_MAX - MUTATION_MIN) * normalized_fitness
    else:
        # 差个体：适当降低交叉率，提高变异率
        crossover_rate = CROSSOVER_MIN + (CROSSOVER_MAX - CROSSOVER_MIN) * 0.5  # 中等交叉率
        # 变异率随适应度增加线性增大
        mutation_rate = MUTATION_MIN + (MUTATION_MAX - MUTATION_MIN) * normalized_fitness

    # 确保概率在指定区间内
    crossover_rate = min(max(crossover_rate, CROSSOVER_MIN), CROSSOVER_MAX)
    mutation_rate = min(max(mutation_rate, MUTATION_MIN), MUTATION_MAX)

    return crossover_rate, mutation_rate

def is_valid_path(path, network):
    for i in range(len(path) - 1):
        if path[i] == -1:
            break
        if network[path[i], path[i + 1]] == 0:
            return False
    return True


# 计算负载均衡度
def calculate_load_balance(global_best_paths, total_nodes):
    # 计算每个节点的负载
    node_loads = defaultdict(int)
    for adc, best_individual in global_best_paths.items():
        path = best_individual.path
        for node in path:
            if node != -1:
                node_loads[node] += 1

    # 计算节点负载的平均值
    load_values = list(node_loads.values())
    avg_load = np.mean(load_values)

    # 计算负载的标准差
    std_load = np.std(load_values)

    # 负载均衡度 = 标准差 / 平均负载
    load_balance = std_load / avg_load if avg_load != 0 else 0
    return load_balance


# 计算所有路径的总时延
def calculate_total_latency(global_best_paths, network):
    total_latency = 0
    node_usage = defaultdict(int)

    for adc, best_individual in global_best_paths.items():
        path = best_individual.path
        path_latency = 0

        # 计算路径的基础时延和链路时延
        for i in range(len(path) - 1):
            if path[i] == -1:
                break
            # 计算每个节点的基础时延
            path_latency += 4  # 每个路由节点基础时延为4

            # 计算链路时延：水平链路的时延为1，垂直链路的时延为0.2
            if i < len(path) - 1 and network[path[i], path[i + 1]] == 1:
                if (path[i] // n) == (path[i + 1] // n):  # 水平链路
                    path_latency += 1
                else:  # 垂直链路
                    path_latency += 0.2

            # 记录节点的使用次数，用于计算排队时延
            node_usage[path[i]] += 1

        # 计算排队时延
        for node, usage in node_usage.items():
            if usage > 1:
                path_latency += (usage - 1) * 1  # 排队时延：重复出现的次数 * 1

        total_latency += path_latency

    return total_latency


# 遗传算法求解每个采集节点路径
def genetic_algorithm_sequential(network, adc_to_ddr, forbidden_edges):
    global_best_paths = {}

    for ddr, adcs in adc_to_ddr.items():
        for adc in adcs:
            population = []
            for _ in range(population_size):
                path = generate_individual(adc, ddr, network, forbidden_edges)
                if path is not None:
                    fitness = calculate_fitness(path)
                    population.append(Individual(path, fitness))

            # 进行遗传算法迭代
            for generation in range(generations):
                population.sort(key=lambda ind: ind.fitness)
                if population[0].fitness == float('inf'):
                    continue
                # 计算当前种群适应度统计信息
                fitness_values = [ind.fitness for ind in population]
                best_fitness = min(fitness_values)
                avg_fitness = np.mean(fitness_values)
                max_fitness = max(fitness_values)
                min_fitness = min(fitness_values)
                # 精英保留
                new_population = population[:elite_size]
                while len(new_population) < population_size:
                    p1, p2 = random.sample(population[:20], 2)
                    # 计算自适应概率
                    crossover_rate, mutation_rate = calculate_adaptive_rates(
                        min(p1.fitness, p2.fitness),  # 取两个父代中较好的适应度
                        avg_fitness,
                        max_fitness,
                        min_fitness
                    )
                    if random.random() < crossover_rate:
                        child1, child2 = crossover(p1, p2, network, forbidden_edges)
                        if child1 and child2:
                            new_population.extend([child1, child2])
                    if random.random() < mutation_rate:
                        mutated_individual = mutate(random.choice(new_population), network, forbidden_edges)
                        if mutated_individual:
                            new_population.append(mutated_individual)
                population = new_population[:population_size]
                best_individual = min(new_population, key=lambda ind: ind.fitness)
                global_best_paths[adc] = best_individual

    return global_best_paths


# 主程序入口
def main():
    network = create_3d_network(n, layers)
    forbidden_edges = set()

    # 第一阶段：存储节点到传输节点路径设为专用路径
    print("第一阶段：计算存储节点到传输节点的路径并设为专用路径")
    for ddr in ddr_nodes:
        # 存储节点到传输节点的路径为 [ddr, transmission_node, -1, ..., -1]
        path = [ddr, transmission_node] + [-1] * (total_nodes - 2)
        forbidden_edges.update(zip(path, path[1:]))
        print(f"存储节点 {ddr} 到传输节点 {transmission_node} 的专用路径: {path}")

    # 第二阶段：依次计算采集节点到存储节点的路径
    print("\n第二阶段：依次计算采集节点到存储节点的路径")
    global_best_paths = genetic_algorithm_sequential(network, adc_to_ddr_mapping, forbidden_edges)

    # 计算并输出负载均衡度
    load_balance = calculate_load_balance(global_best_paths, total_nodes)
    print(f"负载均衡度: {load_balance:.4f}")

    # 计算并输出总时延
    total_latency = calculate_total_latency(global_best_paths, network)
    print(f"所有采集节点到存储节点的总时延: {total_latency:.4f}")

    # 输出每个采集节点的最优路径及适应度
    for adc, best_individual in global_best_paths.items():
        print(f"采集节点 {adc} 的最优路径: {best_individual.path}, 适应度: {best_individual.fitness}")


if __name__ == "__main__":
    main()
