"""
提供 Fig.10 CEV 拓扑的构建函数，供 topo_analysis 工具脚本使用。
拓扑定义与 data_generater.py 保持一致：15 SW + 31 ES，共 46 节点。
节点编号：SW0~SW14 → 0~14，ES1~ES31 → 15~45
SW3(node 3) 和 SW7(node 7) 无挂载 ES
"""

import networkx as nx

# SW-SW 骨干链路（0-based）
_SW_EDGES = [
    (0, 2), (0, 12), (1, 2),  (1, 12),
    (2, 3), (2, 4),  (2, 13), (2, 14),
    (3, 4), (3, 11), (3, 12),
    (4, 5), (4, 7),  (4, 8),
    (5, 6), (6, 7),
    (7, 9), (7, 11), (8, 11),
    (9, 10),(10, 11),(11, 12),
    (12, 13),(12, 14),
]

# ES 挂载表：SW node → [ES node, ...]
_ES_MOUNT = {
    0:  [15, 16],
    1:  [17, 18, 19],
    2:  [20],
    4:  [21, 22, 23],
    5:  [24, 25],
    6:  [26, 27],
    8:  [28, 29],
    9:  [30, 31],
    10: [32, 33],
    11: [34, 35, 36],
    12: [37, 38],
    13: [39, 40],
    14: [41, 42, 43, 44, 45],
}

# 反向映射：ES node → 其所属 SW node
_ES_TO_SW = {es: sw for sw, es_list in _ES_MOUNT.items() for es in es_list}


def build_topology():
    """
    构建 Fig.10 CEV 无向图。

    Returns
    -------
    g : nx.Graph
        完整拓扑图（SW 节点 0~14 + ES 节点 15~45）。
    es_list : list[int]
        所有 ES 节点编号列表（已排序）。
    es_to_sw : dict[int, int]
        ES node → 所属 SW node 的映射（也用作 ES 节点集合的查找键）。
    """
    G = nx.Graph()
    G.add_edges_from(_SW_EDGES)

    es_list = []
    for sw_id, es_ids in _ES_MOUNT.items():
        for es_id in es_ids:
            G.add_edge(es_id, sw_id)
            es_list.append(es_id)

    return G, sorted(es_list), _ES_TO_SW
