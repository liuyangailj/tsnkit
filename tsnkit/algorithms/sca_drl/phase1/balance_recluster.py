"""均衡谱聚类后处理脚本。

从已有的 *_k5_d32_emb.pt 直接读取 embedding（不重建冲突图，不重跑 GNN），
对指定 benchmark 子目录下的所有实例执行均衡化谱聚类，
输出 *_k5_d32_c4bal_group.csv，与原 c4_group.csv 并存，不覆盖任何已有文件。

均衡算法（两步法）
  1. 标准 SpectralClustering 得初始标签（与 batch_infer 亲和度矩阵完全一致）
  2. 贪心后处理：识别超量组 (size > ceil(N/k × (1+slack)))，
     把超量组内距自身中心最远的边界点，逐一移到最近的欠量组，
     直到所有组满足 size ≤ max_size 或无法继续移动为止。

用法
----
cd tsnkit/algorithms/sca_drl
python phase1/balance_recluster.py                              # 默认 benchmark_v4
python phase1/balance_recluster.py --benchmark_subdir benchmark_v4
python phase1/balance_recluster.py --slack 0.05 --n_clusters 4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import glob
import csv
import math
import argparse
import time

import torch
import numpy as np
from sklearn.cluster import SpectralClustering

from sca_drl.common.utils import load_config, resolve_path
from sca_drl.phase1.infer import compute_affinity_matrix


MODEL_TAG  = "k5_d32"   # 与 batch_infer 输出文件名一致


# ─────────────────────────────────────────────────────────────────────────────

def _bal_suffix(n_clusters: int) -> str:
    return f"c{n_clusters}bal"


def _out_path(task_path: str, n_clusters: int) -> str:
    base = os.path.basename(task_path).replace(".csv", "")
    d    = os.path.dirname(task_path)
    return os.path.join(d, f"{base}_{MODEL_TAG}_{_bal_suffix(n_clusters)}_group.csv")


def _emb_path(task_path: str) -> str:
    return task_path.replace(".csv", f"_{MODEL_TAG}_emb.pt")


def _already_done(task_path: str, n_clusters: int) -> bool:
    return os.path.exists(_out_path(task_path, n_clusters))


# ─────────────────────────────────────────────────────────────────────────────

def _greedy_balance(labels: np.ndarray, emb_norm: np.ndarray,
                    n_clusters: int, max_size: int) -> np.ndarray:
    """贪心后处理：每次移动一个点，直到所有组不超过 max_size。

    移动策略：从当前最大超量组中，选距自身中心最远的边界点，
    移入距该点最近且仍有空余的欠量组。
    """
    labels = labels.copy()
    D = emb_norm.shape[1]

    for _ in range(len(labels)):   # 最多移动 N 次，安全上限
        sizes = np.bincount(labels, minlength=n_clusters)
        if np.all(sizes <= max_size):
            break

        # 计算各组中心
        centers = np.zeros((n_clusters, D))
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                centers[c] = emb_norm[mask].mean(axis=0)

        # 选最大超量组
        over_sizes = sizes.copy()
        over_sizes[sizes <= max_size] = 0
        src = int(np.argmax(over_sizes))
        if sizes[src] <= max_size:
            break

        # 该组内距自身中心最远的点（最边界）
        pts = np.where(labels == src)[0]
        dist_own = np.linalg.norm(emb_norm[pts] - centers[src], axis=1)
        border_pt = int(pts[np.argmax(dist_own)])

        # 找最近的欠量组
        under_mask = sizes < max_size
        if not under_mask.any():
            break
        dists = np.where(
            under_mask,
            np.linalg.norm(centers - emb_norm[border_pt], axis=1),
            np.inf,
        )
        tgt = int(np.argmin(dists))
        labels[border_pt] = tgt

    return labels


def balanced_spectral_cluster(embeddings: np.ndarray,
                               n_clusters: int = 4,
                               slack: float = 0.05,
                               random_state: int = 42) -> np.ndarray:
    """标准谱聚类 + 贪心均衡后处理。

    Returns
    -------
    labels : np.ndarray shape [N], dtype int
    """
    W = compute_affinity_matrix(embeddings)
    W = (W + W.T) / 2   # 对称化（消除浮点误差）

    sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed",
                             random_state=random_state)
    labels = sc.fit_predict(W)

    N = len(labels)
    max_size = math.ceil(N / n_clusters * (1 + slack))

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / (norms + 1e-8)

    labels = _greedy_balance(labels, emb_norm, n_clusters, max_size)
    return labels


# ─────────────────────────────────────────────────────────────────────────────

def recluster_one(task_path: str, n_clusters: int, slack: float) -> dict:
    """对单个实例做均衡聚类，返回统计信息字典。"""
    emb_file = _emb_path(task_path)
    if not os.path.exists(emb_file):
        return {"status": "no_emb"}

    emb_dict = torch.load(emb_file, weights_only=False)
    # 按 stream_id 排序，保证顺序一致
    sids = sorted(emb_dict.keys(), key=lambda x: int(x))
    embeddings = np.stack([emb_dict[s].numpy() for s in sids])

    labels = balanced_spectral_cluster(embeddings, n_clusters=n_clusters, slack=slack)

    out_file = _out_path(task_path, n_clusters)
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stream_id", "group_id"])
        for sid, gid in zip(sids, labels):
            writer.writerow([sid, int(gid)])

    sizes = np.bincount(labels, minlength=n_clusters)
    return {
        "status": "done",
        "N": len(labels),
        "sizes": sizes.tolist(),
        "max_size": int(sizes.max()),
        "min_size": int(sizes.min()),
        "ratio": float(sizes.max() / sizes.min()) if sizes.min() > 0 else float("inf"),
    }


# ─────────────────────────────────────────────────────────────────────────────

def run(benchmark_subdir: str = "benchmark_v4",
        n_clusters: int = 4,
        slack: float = 0.05) -> None:

    dc       = load_config("configs/data_config.yaml")
    data_dir = resolve_path(dc["data_dir"])

    n_dirs = sorted(glob.glob(os.path.join(data_dir, benchmark_subdir, "N*")))
    if not n_dirs:
        print(f"❌ 未找到数据: {data_dir}/{benchmark_subdir}/N*/")
        return

    all_files = []
    for d in n_dirs:
        all_files.extend(sorted(glob.glob(os.path.join(d, "*_task.csv"))))

    pending = [f for f in all_files if not _already_done(f, n_clusters)]
    skipped = len(all_files) - len(pending)

    tag = _bal_suffix(n_clusters)
    print("=" * 60)
    print(f"  均衡谱聚类后处理  →  {tag}_group.csv")
    print(f"  benchmark: {benchmark_subdir}  |  n_clusters={n_clusters}  |  slack={slack*100:.0f}%")
    print(f"  max_size per group = ceil(N/{n_clusters} × {1+slack:.2f})")
    print(f"  总计: {len(all_files)} 套  |  跳过(已有): {skipped}  |  待处理: {len(pending)}")
    print("=" * 60)

    if not pending:
        print("  全部已完成，无需处理。")
        return

    t0 = time.time()
    done, no_emb = 0, 0
    ratio_list = []

    for i, task_path in enumerate(pending, 1):
        res = recluster_one(task_path, n_clusters, slack)
        elapsed = time.time() - t0
        eta     = elapsed / i * (len(pending) - i) if i < len(pending) else 0
        stem    = os.path.basename(task_path)

        if res["status"] == "no_emb":
            no_emb += 1
            sys.stdout.write(f"\r  [{i:>4}/{len(pending)}] {stem:<40} ⚠ emb.pt 缺失")
        else:
            done += 1
            ratio_list.append(res["ratio"])
            sizes_str = "/".join(str(s) for s in res["sizes"])
            sys.stdout.write(
                f"\r  [{i:>4}/{len(pending)}] {stem:<40}"
                f" sizes=[{sizes_str}] ratio={res['ratio']:.2f}x"
                f" | {elapsed:5.0f}s ETA {eta:5.0f}s"
            )
        sys.stdout.flush()

    print(f"\n{'─'*60}")
    print(f"  完成: {done}  跳过(已有): {skipped}  emb缺失: {no_emb}")
    if ratio_list:
        arr = np.array(ratio_list)
        print(f"  max/min 比率  均值: {arr.mean():.2f}x  最差: {arr.max():.2f}x  最好: {arr.min():.2f}x")
    print(f"  总耗时: {time.time()-t0:.1f}s")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="均衡谱聚类后处理（不重训、不重跑GNN）")
    parser.add_argument("--benchmark_subdir", default="benchmark_v4")
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--slack",      type=float, default=0.05,
                        help="均衡松弛比例，默认 0.05 即 max_size=ceil(N/k×1.05)")
    args = parser.parse_args()
    run(args.benchmark_subdir, args.n_clusters, args.slack)
