"""比较 DRL env 与 Phase 1 KSP 两种路径算法的候选路径集合重合度。

方法 A（DRL env）: net.get_all_path()[:K]  — nx.all_simple_paths 前 K 条
方法 B（Phase 1）: nx.shortest_simple_paths islice K  — 严格按跳数升序 K 条

路径标识：tuple(节点整数序列)，有序，两种方法统一转换后做 set 交集。

输入：benchmark_v3 每档取第 1 个文件，共 12 档。
输出：终端汇总表 + results/path_comparison.csv（逐流明细）。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import glob
import csv
import argparse
from itertools import islice
from collections import defaultdict

import numpy as np
import networkx as nx

from tsnkit import core as utils_tsnkit
from sca_drl.common.utils import load_config, resolve_path

K = 5  # 候选路径数，与 Phase 1 和 DRL env 一致


def k_shortest_paths(graph, source, target, k):
    """严格按跳数升序返回前 k 条简单路径（与 phase1/dataset.py 一致）。"""
    try:
        return list(islice(nx.shortest_simple_paths(graph, source, target), k))
    except nx.NetworkXNoPath:
        return []


def path_to_key(path_obj):
    """Path 对象 → 可哈希的节点元组。"""
    return tuple(int(n) for n in path_obj.nodes)


def compare_flow(net, stream, k=K):
    """对单条流计算两种方法的路径集合并返回 overlap 统计。"""
    src, dst = stream.src, stream.dst

    # 方法 A：DRL env 实际使用（all_simple_paths 前 K 条）
    all_paths = net.get_all_path(src, dst)
    set_a = set(path_to_key(p) for p in all_paths[:k])

    # 方法 B：Phase 1 KSP（shortest_simple_paths 前 K 条）
    ksp_node_lists = k_shortest_paths(net._net_nx, int(src), int(dst), k)
    set_b = set(tuple(nodes) for nodes in ksp_node_lists)

    overlap  = len(set_a & set_b)
    size_a   = len(set_a)
    size_b   = len(set_b)

    return {
        "src":          int(src),
        "dst":          int(dst),
        "size_a":       size_a,
        "size_b":       size_b,
        "overlap":      overlap,
        "exact_match":  int(overlap == k and size_a == k and size_b == k),
    }


def run(benchmark_subdir, output_dir):
    dc        = load_config("configs/data_config.yaml")
    data_dir  = resolve_path(dc["data_dir"])
    topo_path = os.path.join(data_dir, "0_topo.csv")
    net       = utils_tsnkit.load_network(topo_path)

    n_dirs = [d for d in sorted(glob.glob(os.path.join(data_dir, benchmark_subdir, "N*")))]
    if not n_dirs:
        print(f"❌ 未找到数据: {data_dir}/{benchmark_subdir}/N*/")
        return

    os.makedirs(output_dir, exist_ok=True)
    result_csv = os.path.join(output_dir, "path_comparison.csv")

    all_rows = []

    print(f"路径算法对比 — {benchmark_subdir}，每档取第 1 个文件，K={K}\n")
    print(f"  {'N':>4} | {'flows':>6} | {'exact(5/5)':>10} | {'4/5':>6} | {'3/5':>6} | {'<=2/5':>6} | {'mean overlap':>12}")
    print(f"  {'─'*70}")

    for n_dir in n_dirs:
        n = int(os.path.basename(n_dir)[1:])
        files = sorted(glob.glob(os.path.join(n_dir, "*_task.csv")))
        if not files:
            continue
        task_path = files[0]                       # 每档只取第 1 个文件
        task      = utils_tsnkit.load_stream(task_path)

        overlaps = []
        for stream in task.streams:
            row = compare_flow(net, stream, k=K)
            row["N"]         = n
            row["task_file"] = os.path.basename(task_path)
            all_rows.append(row)
            overlaps.append(row["overlap"])

        overlaps = np.array(overlaps)
        exact    = np.mean(overlaps == K) * 100
        ge4      = np.mean(overlaps == 4) * 100
        ge3      = np.mean(overlaps == 3) * 100
        le2      = np.mean(overlaps <= 2) * 100
        mean_ov  = overlaps.mean()

        print(f"  N={n:>3d} | {len(overlaps):>6} | {exact:>9.1f}% | {ge4:>5.1f}% | {ge3:>5.1f}% | {le2:>5.1f}% | {mean_ov:>10.3f}/5")

    # 写 CSV
    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["N", "task_file", "src", "dst",
                                                "size_a", "size_b", "overlap", "exact_match"])
        writer.writeheader()
        writer.writerows(all_rows)

    # 全局汇总
    all_overlaps = np.array([r["overlap"] for r in all_rows])
    print(f"\n  {'─'*70}")
    print(f"  全局合计: {len(all_overlaps)} 条流")
    print(f"  exact(5/5): {np.mean(all_overlaps == K)*100:.1f}%")
    print(f"  mean overlap: {all_overlaps.mean():.3f} / {K}")
    print(f"\n✅ 逐流明细已保存: {result_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRL vs KSP 路径集合重合度分析")
    parser.add_argument("--benchmark_subdir", default="benchmark_v3")
    parser.add_argument("--output", default="results/path_comparison")
    args = parser.parse_args()

    run(args.benchmark_subdir, resolve_path(args.output))
