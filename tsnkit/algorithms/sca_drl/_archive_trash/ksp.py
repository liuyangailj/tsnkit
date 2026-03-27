import networkx as nx
from itertools import islice

class KSPCalculator:
    def __init__(self, G):
        self.G = G

    def compute_k_shortest_paths(self, src, dst, k=5):
        """
        计算源宿之间的 K 条最短路径 (Yen's Algorithm)
        返回: list of paths, e.g., [[0, 1, 3], [0, 2, 3]]
        """
        try:
            # networkx 的 shortest_simple_paths 也就是 Yen's 算法的变体
            # weight=None 表示按跳数 (hop) 计算，也可以设为 'propagation_delay'
            paths_gen = nx.shortest_simple_paths(self.G, source=src, target=dst, weight=None)
            
            # 取前 k 条
            k_paths = list(islice(paths_gen, k))
            return k_paths
        except nx.NetworkXNoPath:
            return []