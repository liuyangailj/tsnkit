"""Phase 1 推理：加载 GAT → 计算亲和度矩阵 → 谱聚类分组。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import csv

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import SpectralClustering

from sca_drl.common.utils import (
    set_seed, load_config, resolve_path, get_device, ensure_dir
)
from sca_drl.phase1.dataset import TSNPhase1Dataset
from sca_drl.phase1.model import GNNPartitionModel

def compute_affinity_matrix(embeddings):
    """纯 GNN 亲和度矩阵: W_ij = (cos_sim + 1) / 2，值域 [0, 1]。

    Args:
        embeddings: numpy [N, D]，GNN 输出

    Returns:
        numpy [N, N]: 对称亲和度矩阵
    """
    emb = F.normalize(torch.tensor(embeddings, dtype=torch.float32), p=2, dim=1)
    W = ((emb @ emb.t() + 1) / 2).numpy()
    np.fill_diagonal(W, 1.0)
    return W

def main(config: dict):
    """从配置字典启动推理 + 谱聚类。"""
    data_cfg = config["data"]
    model_cfg = config["model"]
    infer_cfg = config["inference"]

    task_path = resolve_path(infer_cfg["task_file"])
    topo_path = resolve_path(data_cfg["topo_file"])

    # 构建冲突图（复用训练时相同的图结构）
    dataset = TSNPhase1Dataset(task_path, topo_path, k_paths=data_cfg["k_paths"])
    data = dataset[0]

    # 加载模型
    device = get_device()
    model = GNNPartitionModel(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
        heads=model_cfg["heads"],
    ).to(device)

    ckpt = resolve_path(infer_cfg["checkpoint_path"])
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"Model loaded: {ckpt}")

    # 前向推理
    data_dev = data.to(device)
    with torch.no_grad():
        embeddings = model(data_dev).cpu().numpy()

    # 谱聚类
    W = compute_affinity_matrix(embeddings)

    n_clusters = infer_cfg["n_clusters"]
    print(f"Spectral clustering (k={n_clusters})...")
    sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42)
    labels = sc.fit_predict(W)

    # 保存分组 CSV
    out_csv = resolve_path(infer_cfg["output"]["groups_csv"])
    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stream_id", "group_id"])
        for sid, gid in zip(data.stream_ids, labels):
            writer.writerow([sid, int(gid)])

    # 保存 embedding（供 Phase 2 使用）
    out_emb = resolve_path(infer_cfg["output"]["embeddings_pt"])
    emb_dict = {sid: embeddings[i] for i, sid in enumerate(data.stream_ids)}
    emb_dict_torch = {k: torch.tensor(v) for k, v in emb_dict.items()}
    torch.save(emb_dict_torch, out_emb)

    print(f"Groups  → {out_csv}")
    print(f"Embeddings → {out_emb}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1 Inference")
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    main(cfg)