"""最小独立验证：DRL env DFS[:5] vs Phase1 KSP[:5]，只依赖 networkx + csv。

直接读 0_topo.csv，对 CSV 中第一行流 (39→45) 逐条打印两种方法的路径节点序列，
不依赖任何 tsnkit 类，排除封装层引入 bug 的可能。

运行：
  cd tsnkit/algorithms/sca_drl
  python eval/proof_paths.py
  python eval/proof_paths.py --src 18 --dst 16   # 任意指定 src/dst
"""

import csv
import ast
import argparse
from itertools import islice

import networkx as nx

TOPO_CSV = "data/data_storm/fig10/0_topo.csv"
K = 5


def load_digraph(topo_path):
    """只读 link 列构建 DiGraph，不需要 tsnkit。"""
    G = nx.DiGraph()
    with open(topo_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u, v = ast.literal_eval(row["link"])
            G.add_edge(u, v)
    return G


def dfs_paths(G, src, dst, k):
    """方法 A：all_simple_paths DFS 顺序前 k 条（等同 net.get_all_path()[:k]）。"""
    return list(islice(nx.all_simple_paths(G, src, dst), k))


def ksp_paths(G, src, dst, k):
    """方法 B：shortest_simple_paths Yen 算法跳数升序前 k 条（等同 Phase1 KSP）。"""
    try:
        return list(islice(nx.shortest_simple_paths(G, src, dst), k))
    except nx.NetworkXNoPath:
        return []


def show(G, src, dst):
    pa_list = dfs_paths(G, src, dst, K)
    pb_list = ksp_paths(G, src, dst, K)
    set_a = {tuple(p) for p in pa_list}
    set_b = {tuple(p) for p in pb_list}
    shared = set_a & set_b

    print(f"\n{'='*80}")
    print(f"Flow  {src} -> {dst}  |  overlap = {len(shared)}/{K}")
    print(f"{'─'*80}")
    print(f"  # | Method A  (DRL env: all_simple_paths DFS[:5])        hops")
    print(f"    | Method B  (Phase1:  shortest_simple_paths Yen[:5])   hops")
    print(f"{'─'*80}")
    for i in range(K):
        pa = pa_list[i] if i < len(pa_list) else None
        pb = pb_list[i] if i < len(pb_list) else None
        tag_a = "<<shared>>" if pa and tuple(pa) in shared else ""
        tag_b = "<<shared>>" if pb and tuple(pb) in shared else ""
        ha = len(pa) - 1 if pa else "-"
        hb = len(pb) - 1 if pb else "-"
        same = " <-- IDENTICAL" if pa == pb else ""
        print(f" A{i+1} | hops={ha:>2}  {pa}  {tag_a}{same}")
        print(f" B{i+1} | hops={hb:>2}  {pb}  {tag_b}")
        print()

    # 全量路径数（此拓扑上简单路径总数）
    all_simple = list(nx.all_simple_paths(G, src, dst))
    all_hops = sorted(len(p) - 1 for p in all_simple)
    print(f"Total simple paths for {src}->{dst}: {len(all_simple)}")
    print(f"Hop distribution: {dict(zip(*[list(x) for x in __import__('numpy').unique(all_hops, return_counts=True)]))}")
    print(f"\nA selects paths with hops: {sorted(len(p)-1 for p in pa_list)}")
    print(f"B selects paths with hops: {sorted(len(p)-1 for p in pb_list)}")
    print(f"A - B (unique to A, not in B): {sorted(len(p)-1 for p in set_a - set_b)}")
    print(f"B - A (unique to B, not in A): {sorted(len(p)-1 for p in set_b - set_a)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=int, default=39)
    parser.add_argument("--dst", type=int, default=45)
    parser.add_argument("--topo", default=TOPO_CSV)
    args = parser.parse_args()

    G = load_digraph(args.topo)
    print(f"Topology: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    show(G, args.src, args.dst)

    # 额外验证几对（CSV里 overlap=0 的典型案例）
    for src, dst in [(29, 24), (42, 33), (37, 30), (28, 41)]:
        show(G, src, dst)
