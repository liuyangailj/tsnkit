"""
统计所有 ES 对的最短路径负载情况。
约束：src 和 dst 不能挂载在同一个 SW 上（同 SW 流量不经过骨干网）。

输出：
  - 前 5 条重叠最多的 SW-SW 链路
  - 前 5 个被经过最多的 SW 节点
"""

from collections import Counter
from itertools import combinations

import networkx as nx

from traffic_generator import build_topology


def collect_path_stats():
    g, es_list, es_to_sw = build_topology()

    edge_load: Counter = Counter()       # SW-SW 链路负载
    node_load: Counter = Counter()       # SW 节点中转负载（不计端点 SW）
    sw_endpoint_load: Counter = Counter()  # SW 作为端点的负载（含挂载 ES 的 SW）

    total_pairs = 0
    skipped_same_sw = 0

    # 对所有无序 ES 对统计（避免重复计算 a->b 和 b->a）
    for src, dst in combinations(es_list, 2):
        if es_to_sw[src] == es_to_sw[dst]:
            skipped_same_sw += 1
            continue
        total_pairs += 1

        path = nx.shortest_path(g, src, dst)
        # path[0] = src ES, path[-1] = dst ES
        # path[1] = src 挂载的 SW, path[-2] = dst 挂载的 SW
        # 中间的 path[2:-2] 是中转 SW（如果有）

        # SW-SW 链路：相邻的两个都是 SW 节点
        for u, v in zip(path[:-1], path[1:]):
            if u not in es_to_sw and v not in es_to_sw:
                edge_load[tuple(sorted((u, v)))] += 1

        # SW 节点负载：所有出现在路径里的 SW
        sw_in_path = [n for n in path if n not in es_to_sw]
        # 中间中转 SW（不含两端的接入 SW）
        for sw in sw_in_path[1:-1]:
            node_load[sw] += 1
        # 端点 SW（接入 SW），单独统计便于对比
        if len(sw_in_path) >= 1:
            sw_endpoint_load[sw_in_path[0]] += 1
            if len(sw_in_path) >= 2:
                sw_endpoint_load[sw_in_path[-1]] += 1

    return {
        "total_pairs": total_pairs,
        "skipped_same_sw": skipped_same_sw,
        "edge_load": edge_load,
        "node_load_transit": node_load,
        "node_load_endpoint": sw_endpoint_load,
    }


def print_report():
    stats = collect_path_stats()
    total = stats["total_pairs"]
    skipped = stats["skipped_same_sw"]

    print(f"统计范围：")
    print(f"  有效 ES 对（跨 SW）: {total}")
    print(f"  跳过（同 SW）:      {skipped}")
    print(f"  总计:               {total + skipped}")
    print()

    print(f"重叠最多的前 5 条 SW-SW 链路:")
    print(f"  {'排名':<4} {'链路':<12} {'被经过次数':<10} {'占比':<8}")
    for i, (edge, load) in enumerate(stats["edge_load"].most_common(5), 1):
        pct = load / total * 100
        print(f"  {i:<4} SW{edge[0]:>2}-SW{edge[1]:<4}  {load:<10} {pct:>5.1f}%")
    print()

    print(f"被经过最多的前 5 个 SW 节点（仅作为中转，不含端点）:")
    print(f"  {'排名':<4} {'SW':<6} {'中转次数':<10} {'占比':<8}")
    for i, (sw, load) in enumerate(stats["node_load_transit"].most_common(5), 1):
        pct = load / total * 100
        print(f"  {i:<4} SW{sw:<4}  {load:<10} {pct:>5.1f}%")
    print()

    print(f"被经过最多的前 5 个 SW 节点（含作为端点接入 SW）:")
    print(f"  {'排名':<4} {'SW':<6} {'总次数':<10} {'(中转':<8} {'+端点)':<10}")
    combined = Counter()
    for sw, n in stats["node_load_transit"].items():
        combined[sw] += n
    for sw, n in stats["node_load_endpoint"].items():
        combined[sw] += n
    for i, (sw, load) in enumerate(combined.most_common(5), 1):
        transit = stats["node_load_transit"].get(sw, 0)
        endpoint = stats["node_load_endpoint"].get(sw, 0)
        print(f"  {i:<4} SW{sw:<4}  {load:<10} {transit:<8} {endpoint:<10}")


if __name__ == "__main__":
    print_report()