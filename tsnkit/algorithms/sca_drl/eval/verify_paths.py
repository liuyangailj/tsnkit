"""定性验证：DRL env vs Phase 1 KSP 两种方法的实际路径节点序列与跳数对比。

用法：
  cd tsnkit/algorithms/sca_drl
  python eval/verify_paths.py
  python eval/verify_paths.py --n_show 5 --n_dir N40
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import glob
import argparse
from itertools import islice

import numpy as np
import networkx as nx

from tsnkit import core as utils_tsnkit
from sca_drl.common.utils import load_config, resolve_path

K = 5


def k_shortest_paths(graph, source, target, k):
    try:
        return list(islice(nx.shortest_simple_paths(graph, source, target), k))
    except nx.NetworkXNoPath:
        return []


def path_to_key(path_obj):
    return tuple(int(n) for n in path_obj.nodes)


def show_flow(net, src, dst, flow_idx):
    paths_a = [path_to_key(p) for p in net.get_all_path(src, dst)[:K]]
    paths_b = [tuple(p) for p in k_shortest_paths(net._net_nx, int(src), int(dst), K)]
    set_a, set_b = set(paths_a), set(paths_b)
    overlap = len(set_a & set_b)
    shared  = set_a & set_b
    hops_a  = [len(p) - 1 for p in paths_a]
    hops_b  = [len(p) - 1 for p in paths_b]

    print("", flush=True)
    print(f"Flow #{flow_idx}  {int(src)} -> {int(dst)}  overlap={overlap}/{K}"
          f"  A_avg_hops={np.mean(hops_a):.1f}  B_avg_hops={np.mean(hops_b):.1f}", flush=True)
    print(f"  {'#':<3} {'Method A  (DRL env, all_simple_paths[:5])':<48} hops"
          f"   {'Method B  (Phase1 KSP[:5])':<48} hops  same?", flush=True)
    print(f"  {'-'*120}", flush=True)

    n_rows = max(len(paths_a), len(paths_b))
    for i in range(n_rows):
        pa = paths_a[i] if i < len(paths_a) else None
        pb = paths_b[i] if i < len(paths_b) else None
        mark_a = "[shared]" if pa and pa in shared else "        "
        mark_b = "[shared]" if pb and pb in shared else "        "
        pa_str = str(list(pa)) if pa else "---"
        pb_str = str(list(pb)) if pb else "---"
        ha_str = str(len(pa) - 1) if pa else "-"
        hb_str = str(len(pb) - 1) if pb else "-"
        same   = "<-- SAME" if pa == pb else ""
        print(f"  [{i+1}] {mark_a} {pa_str:<48} {ha_str:>4}"
              f"   {mark_b} {pb_str:<48} {hb_str:>4}  {same}", flush=True)

    return overlap, hops_a, hops_b, shared


def run(benchmark_subdir, n_dir_name, n_show):
    dc        = load_config("configs/data_config.yaml")
    data_dir  = resolve_path(dc["data_dir"])
    topo_path = os.path.join(data_dir, "0_topo.csv")
    net = utils_tsnkit.load_network(topo_path)

    if n_dir_name:
        n_dir = os.path.join(data_dir, benchmark_subdir, n_dir_name)
    else:
        n_dirs = sorted(glob.glob(os.path.join(data_dir, benchmark_subdir, "N*")))
        if not n_dirs:
            print(f"No data found: {data_dir}/{benchmark_subdir}/N*/", flush=True)
            return
        n_dir = n_dirs[0]

    task_files = sorted(glob.glob(os.path.join(n_dir, "*_task.csv")))
    if not task_files:
        print(f"No task files: {n_dir}", flush=True)
        return

    task_path = task_files[0]
    task = utils_tsnkit.load_stream(task_path)
    n = int(os.path.basename(n_dir)[1:])

    print(f"File: {task_path}  (N={n}, {len(task.streams)} flows)", flush=True)
    print(f"Showing first {n_show} flows in detail\n", flush=True)

    all_hops_a, all_hops_b, all_overlaps = [], [], []
    all_shared_hops = []

    for idx, stream in enumerate(task.streams):
        src, dst = stream.src, stream.dst
        if idx < n_show:
            ov, ha, hb, shared = show_flow(net, src, dst, idx + 1)
            for p in shared:
                all_shared_hops.append(len(p) - 1)
        else:
            paths_a = [path_to_key(p) for p in net.get_all_path(src, dst)[:K]]
            paths_b = [tuple(p) for p in k_shortest_paths(net._net_nx, int(src), int(dst), K)]
            set_a, set_b = set(paths_a), set(paths_b)
            ov = len(set_a & set_b)
            ha = [len(p) - 1 for p in paths_a]
            hb = [len(p) - 1 for p in paths_b]
            for p in set_a & set_b:
                all_shared_hops.append(len(p) - 1)
        all_overlaps.append(ov)
        all_hops_a.extend(ha)
        all_hops_b.extend(hb)

    # ---- global stats ----
    print(f"\n{'='*90}", flush=True)
    print(f"Full-file stats  N={n}  flows={len(task.streams)}", flush=True)
    print(f"  mean overlap            : {np.mean(all_overlaps):.3f} / {K}", flush=True)
    print(f"  A avg hops (DRL env)    : {np.mean(all_hops_a):.3f}", flush=True)
    print(f"  B avg hops (Phase1 KSP) : {np.mean(all_hops_b):.3f}", flush=True)
    print(f"  A - B hop diff          : {np.mean(all_hops_a) - np.mean(all_hops_b):+.3f}", flush=True)
    if all_shared_hops:
        print(f"  shared paths avg hops   : {np.mean(all_shared_hops):.3f}"
              f"  (shared paths tend to be shortest?)", flush=True)

    print(f"\n  Hop distribution -- B (Phase1 KSP, should be shortest-first):", flush=True)
    vals, counts = np.unique(all_hops_b, return_counts=True)
    for h, c in zip(vals, counts):
        bar = "#" * (c * 40 // len(all_hops_b))
        print(f"    {h} hops: {bar:<40} {c:>5} ({c*100/len(all_hops_b):.1f}%)", flush=True)

    print(f"\n  Hop distribution -- A (DRL env, DFS order):", flush=True)
    vals, counts = np.unique(all_hops_a, return_counts=True)
    for h, c in zip(vals, counts):
        bar = "#" * (c * 40 // len(all_hops_a))
        print(f"    {h} hops: {bar:<40} {c:>5} ({c*100/len(all_hops_a):.1f}%)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_subdir", default="benchmark_v3")
    parser.add_argument("--n_dir",  default="", help="e.g. N40")
    parser.add_argument("--n_show", type=int, default=5)
    args = parser.parse_args()
    run(args.benchmark_subdir, args.n_dir, args.n_show)
