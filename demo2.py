import numpy as np
from collections import defaultdict

# 三维mesh网络的参数设置
n = 4  # 每层网格的大小
layers = 3  # 三维网格的层数
total_nodes = n * n * layers  # 总节点数

# 输入的采集节点到存储节点的路径信息
paths = [
    [1, 17, 33],
    [2, 1, 17, 33],
    [3, 2, 1, 17, 33],
    [5, 21, 37, 33],
    [17, 33],
    [18, 22, 21, 17, 33],
    [19, 3, 7, 23, 22, 21, 17, 33],
    [22, 21, 17, 33],
    [0, 16, 20, 36],
    [4, 20, 36],
    [8, 4, 20, 36],
    [9, 25, 24, 40, 36],
    [16, 20, 36],
    [20, 36],
    [24, 40, 36],
    [21, 37, 36],
    [6, 22, 38],
    [7, 23, 22, 38],
    [10, 6, 22, 38],
    [11, 15, 31, 47, 43, 39, 38],
    [23, 22, 38],
    [25, 41, 42, 38],
    [26, 10, 6, 22, 38],
    [27, 43, 42, 38],
    [12, 28, 29, 45, 41],
    [13, 29, 25, 41],
    [14, 10, 9, 25, 41],
    [15, 14, 13, 29, 45, 41],
    [28, 24, 25, 41],
    [29, 45, 41],
    [30, 26, 25, 41],
    [31, 30, 26, 25, 41]
]

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


# 计算时延
def calculate_latency(paths, network):
    node_usage = defaultdict(int)  # 记录每个节点被多少路径经过
    total_latency = 0  # 总时延

    # 遍历所有路径，计算节点的基础时延、链路时延和排队时延
    for path in paths:
        path_latency = 0  # 每条路径的时延

        for i in range(len(path) - 1):
            node = path[i]
            next_node = path[i + 1]

            # 节点基础时延
            path_latency += 4  # 每个路由节点基础时延为4

            # 计算链路时延（水平链路和垂直链路的时延不同）
            if network[node, next_node] == 1:  # 如果两个节点之间有链路
                if (node // n) == (next_node // n):  # 水平链路
                    path_latency += 1
                else:  # 垂直链路
                    path_latency += 0.2

            # 记录节点使用次数（用于计算排队时延）
            node_usage[node] += 1

        # 计算路径经过的节点的排队时延
        for node, usage in node_usage.items():
            if usage > 1:  # 排队时延：多次经过的节点
                path_latency += (usage - 1) * 1  # 每多一个路径经过该节点，加1的排队时延

        total_latency += path_latency

    return total_latency


# 主程序入口
def main():
    # 创建三维网格网络
    network = create_3d_network(n, layers)

    # 计算并输出总时延
    total_latency = calculate_latency(paths, network)
    print(f"总传输时延: {total_latency:.4f}")


if __name__ == "__main__":
    main()
