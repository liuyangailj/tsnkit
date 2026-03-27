## 📂 项目文件结构 (Project Structure)
```text
📁 /
    📄 dataset.py
    📄 infer.py
    📄 init.py
    📄 model.py
    📄 train.py
```

# 💻 源代码上下文 (Source Code Context)

## File: `dataset.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase1\dataset.py`

```python
"""流冲突图数据集构建，基于 TSNKit 数据接口。"""

import os

import math
from itertools import islice

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data, Dataset

from tsnkit import core as utils

def k_shortest_paths(graph, source, target, k, weight=None):
    """基于 NetworkX 计算 K 条最短简单路径。

    Args:
        graph: NetworkX 图对象
        source: 源节点
        target: 目标节点
        k: 最大路径数
        weight: 边权属性名

    Returns:
        list[list]: 最多 k 条路径
    """
    try:
        return list(islice(
            nx.shortest_simple_paths(graph, source, target, weight=weight), k
        ))
    except nx.NetworkXNoPath:
        return []

class TSNPhase1Dataset(Dataset):
    """基于 TSNKit 构建流冲突图。

    节点 = 流，边 = K 最短路径链路重叠关系，边权 = 重叠概率 e_ij。

    Args:
        task_path: TSNKit task.csv 绝对路径
        topo_path: TSNKit topo.csv 绝对路径
        k_paths: 每条流计算的候选最短路径数
    """

    def __init__(self, task_path, topo_path, k_paths=3):
        self.task_path = task_path
        self.topo_path = topo_path
        self.k_paths = k_paths
        super().__init__(root=None, transform=None, pre_transform=None)
        self.data = self._build_conflict_graph()

    def _build_conflict_graph(self):
        """构建流冲突图，返回 PyG Data 对象。"""
        print(f"[Phase1] Loading data: {self.task_path}")

        # TSNKit 原生加载器
        tsn_net = utils.load_network(self.topo_path)
        tsn_tasks = utils.load_stream(self.task_path)

        graph_nx = tsn_net.net_nx
        streams = tsn_tasks.streams
        num_streams = len(streams)
        print(f"[Phase1] {num_streams} streams loaded, building conflict graph...")

        x_list = []
        stream_ids = []
        periods = []
        stream_path_links = []

        for s in streams:
            target = s.dst[0] if isinstance(s.dst, list) else s.dst
            sid = s.name if hasattr(s, 'name') else s._id
            stream_ids.append(sid)
            periods.append(s.period)

            # 节点特征: [size, log(period+1), log(deadline+1)]
            x_list.append([
                s.size,
                math.log(s.period + 1),
                math.log(s.deadline + 1),
            ])

            # K 最短路径 → 链路集合
            paths = k_shortest_paths(graph_nx, s.src, target, self.k_paths)
            path_sets = [set(zip(p[:-1], p[1:])) for p in paths if len(p) >= 2]
            stream_path_links.append(path_sets)

        # 特征张量 & 标准化
        x = torch.tensor(x_list, dtype=torch.float)
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)

        # 冲突边
        edge_index, edge_attr = self._compute_conflict_edges(
            num_streams, stream_path_links
        )

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.stream_ids = stream_ids
        data.periods = torch.tensor(periods, dtype=torch.float)
        return data

    @staticmethod
    def _compute_conflict_edges(num_streams, stream_path_links):
        """计算流对之间的链路重叠概率，构建冲突边。"""
        edges, attrs = [], []

        for i in range(num_streams):
            paths_i = stream_path_links[i]
            k_i = len(paths_i)
            if k_i == 0:
                continue
            for j in range(i + 1, num_streams):
                paths_j = stream_path_links[j]
                k_j = len(paths_j)
                if k_j == 0:
                    continue

                overlap = sum(
                    1 for pi in paths_i for pj in paths_j
                    if not pi.isdisjoint(pj)
                )
                e_ij = overlap / (k_i * k_j)

                if e_ij > 0:
                    edges.extend([[i, j], [j, i]])
                    attrs.extend([[e_ij], [e_ij]])

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        return edge_index, edge_attr

    def len(self):
        return 1

    def get(self, idx):
        return self.data
    
class MultiInstanceDataset(Dataset):    
    """多流集合实例数据集，每个样本是一张独立的冲突图。

    Args:
        task_files: task.csv 路径列表
        topo_path: 共享的 topo.csv 路径
        k_paths: K 最短路径数
    """

    def __init__(self, task_files, topo_path, k_paths=3):
        self.task_files = task_files
        self.topo_path = topo_path
        self.k_paths = k_paths
        super().__init__(root=None, transform=None, pre_transform=None)
        self._graphs = self._build_all_graphs()

    def _build_all_graphs(self):
        graphs = []
        for i, tf in enumerate(self.task_files):
            print(f"[{i+1}/{len(self.task_files)}] Building graph: {os.path.basename(tf)}")
            ds = TSNPhase1Dataset(tf, self.topo_path, self.k_paths)
            graphs.append(ds[0])
        return graphs

    def len(self):
        return len(self._graphs)

    def get(self, idx):
        return self._graphs[idx]
```

## File: `infer.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase1\infer.py`

```python
"""Phase 1 推理：加载 GAT → 计算亲和度矩阵 → 谱聚类分组。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import csv
import math

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import SpectralClustering

from sca_drl.common.utils import (
    set_seed, load_config, resolve_path, get_device, ensure_dir
)
from sca_drl.phase1.dataset import TSNPhase1Dataset
from sca_drl.phase1.model import GNNPartitionModel

def calculate_harmonic_prior(period_i, period_j):
    """谐波先验: gcd(T_i, T_j) / lcm(T_i, T_j)，值域 [0, 1]。"""
    pi, pj = int(period_i), int(period_j)
    gcd = math.gcd(pi, pj)
    lcm = (pi * pj) // gcd
    return gcd / lcm

def compute_affinity_matrix(embeddings, periods, alpha=0.6):
    """混合亲和度矩阵: W_ij = α·s_gnn + (1-α)·s_prior。

    Args:
        embeddings: numpy [N, D]，GNN 输出
        periods: list/tensor，各流周期
        alpha: GNN 相似度权重

    Returns:
        numpy [N, N]: 对称亲和度矩阵
    """
    n = len(periods)
    emb = F.normalize(torch.tensor(embeddings, dtype=torch.float32), p=2, dim=1)
    s_gnn = ((emb @ emb.t() + 1) / 2).numpy()

    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        W[i, i] = 1.0
        for j in range(i + 1, n):
            s_prior = calculate_harmonic_prior(periods[i], periods[j])
            W[i, j] = W[j, i] = alpha * s_gnn[i, j] + (1 - alpha) * s_prior

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
    periods = data.periods.tolist()
    W = compute_affinity_matrix(embeddings, periods, alpha=infer_cfg["alpha"])

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
```

## File: `init.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase1\init.py`

```python
"""Phase 1: 基于 GAT 的流相似度学习与谱聚类分组。"""

from sca_drl.phase1.dataset import TSNPhase1Dataset
from sca_drl.phase1.model import GNNPartitionModel
```

## File: `model.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase1\model.py`

```python
"""GNNPartitionModel: 两层 GAT 流 embedding 模型。"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNNPartitionModel(nn.Module):
    """两层 GAT，显式注入边权 e_ij 到注意力机制。

    Args:
        input_dim: 节点特征维度
        hidden_dim: 隐藏层维度
        output_dim: 输出 embedding 维度
        heads: 第一层注意力头数
    """

    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, edge_dim=1, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, edge_dim=1, concat=False)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x
```

## File: `train.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase1\train.py`

```python
"""Phase 1 GAT 自监督训练：多实例 batch 训练。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import glob
import time

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from sca_drl.common.utils import (
    set_seed, load_config, resolve_path, get_device, ensure_dir
)
from sca_drl.phase1.dataset import TSNPhase1Dataset, MultiInstanceDataset
from sca_drl.phase1.model import GNNPartitionModel

def compute_loss(embeddings, data, neg_sample_ratio=0.5):
    """计算单张图的自监督拓扑重构损失。

    Args:
        embeddings: 模型输出 [N, D]
        data: PyG Data（可能是 batch 中的一张图）
        neg_sample_ratio: 负采样比例

    Returns:
        torch.Tensor: 标量损失
    """
    device = embeddings.device
    num_nodes = embeddings.size(0)

    if data.edge_index.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    src, dst = data.edge_index
    e_ij = data.edge_attr.squeeze()

    # 正样本损失
    sim_pos = (F.cosine_similarity(embeddings[src], embeddings[dst]) + 1) / 2
    sim_pos = torch.clamp(sim_pos, 1e-7, 1 - 1e-7)
    loss_pos = F.binary_cross_entropy(sim_pos, e_ij)

    # 负采样损失
    num_neg = max(1, int(src.size(0) * neg_sample_ratio))
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=device)
    sim_neg = (F.cosine_similarity(embeddings[neg_src], embeddings[neg_dst]) + 1) / 2
    sim_neg = torch.clamp(sim_neg, 1e-7, 1 - 1e-7)
    loss_neg = F.binary_cross_entropy(sim_neg, torch.zeros_like(sim_neg))

    return loss_pos + loss_neg

def train_epoch(model, loader, optimizer, neg_sample_ratio, device):
    """一个 epoch：遍历 DataLoader 中所有 batch。

    Returns:
        float: epoch 平均损失
    """
    model.train()
    total_loss = 0.0
    num_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        embeddings = model(batch)

        # PyG batch 把多张图拼成一张大图，用 batch.batch 区分
        # 需要按图拆分计算损失
        loss = torch.tensor(0.0, device=device)
        graph_ids = batch.batch.unique()

        for gid in graph_ids:
            mask = batch.batch == gid
            node_emb = embeddings[mask]

            # 提取该子图的边
            node_indices = mask.nonzero(as_tuple=True)[0]
            node_min = node_indices.min()
            edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
            sub_edge_index = batch.edge_index[:, edge_mask] - node_min
            sub_edge_attr = batch.edge_attr[edge_mask]

            sub_data = type(batch)(
                edge_index=sub_edge_index,
                edge_attr=sub_edge_attr,
            )
            loss = loss + compute_loss(node_emb, sub_data, neg_sample_ratio)

        loss = loss / len(graph_ids)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(graph_ids)
        num_graphs += len(graph_ids)

    return total_loss / max(num_graphs, 1)

def resolve_task_files(data_cfg, key="task_pattern"):
    """从配置解析 task 文件列表，兼容单文件和多文件模式。"""
    if "task_file" in data_cfg and key == "task_pattern":
        return [resolve_path(data_cfg["task_file"])]

    task_dir = resolve_path(data_cfg["task_dir"])
    pattern = data_cfg.get("task_pattern", "*_task.csv")
    files = sorted(glob.glob(os.path.join(task_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {task_dir}")
    return files

def evaluate(model, loader, neg_sample_ratio, device):
    """验证集评估，不更新梯度。"""
    model.eval()
    total_loss = 0.0
    num_graphs = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embeddings = model(batch)
            graph_ids = batch.batch.unique()

            for gid in graph_ids:
                mask = batch.batch == gid
                node_emb = embeddings[mask]
                node_indices = mask.nonzero(as_tuple=True)[0]
                node_min = node_indices.min()
                edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
                sub_edge_index = batch.edge_index[:, edge_mask] - node_min
                sub_edge_attr = batch.edge_attr[edge_mask]

                sub_data = type(batch)(
                    edge_index=sub_edge_index,
                    edge_attr=sub_edge_attr,
                )
                total_loss += compute_loss(node_emb, sub_data, neg_sample_ratio).item()
                num_graphs += 1

    return total_loss / max(num_graphs, 1)

def main(config: dict):
    """多实例 batch 训练主流程(含验证)。"""
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    topo_path = resolve_path(data_cfg["topo_file"])
    k_paths=data_cfg["k_paths"]
    
    # 训练集
    train_files = resolve_task_files(data_cfg, key="task_pattern")
    train_dataset = MultiInstanceDataset(train_files, topo_path, k_paths)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_cfg.get("batch_size", 8), 
        shuffle=True,
        )
    
    # 验证集
    val_loader = None
    val_pattern = data_cfg.get("val_pattern")
    if val_pattern:
        val_dir = resolve_path(data_cfg["task_dir"])
        val_files = sorted(glob.glob(os.path.join(val_dir, val_pattern)))
        if val_files:
             val_dataset = MultiInstanceDataset(val_files, topo_path, k_paths)
             val_loader = DataLoader(val_dataset, batch_size=train_cfg.get("batch_size", 8))
             print(f"Validation set: {len(val_files)} graphs")

    # 模型 & 优化器
    device = get_device()
    model = GNNPartitionModel(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
        heads=model_cfg["heads"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    n_epochs = train_cfg["n_epochs"]
    neg_ratio = train_cfg.get("neg_sample_ratio", 0.5)
    best_val_loss = float("inf")

    print(f"Training: {n_epochs} epochs, {len(train_dataset)} graphs," 
          f"batch_size={train_cfg.get('batch_size', 8)}")
    print("-" * 60)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, neg_ratio, device)
        elapsed = time.time() - t0
        
        log = f"Epoch {epoch:03d}/{n_epochs} | Train Loss: {train_loss:.4f}"
        # print(f"Epoch {epoch:03d}/{n_epochs} | Loss: {train_loss:.4f} | {elapsed:.1f}s")
        
        if val_loader:
            val_loss = evaluate(model, val_loader, neg_ratio, device)
            log += f" | Val Loss: {val_loss:.4f}"
            # print(f"  Validation Loss: {val_loss:.4f}")
            
            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = resolve_path(
                    train_cfg["checkpoint_path"].replace(".pth", "_best.pth")
                )
                ensure_dir(os.path.dirname(best_path))
                torch.save(model.state_dict(), best_path)
                log += " ★"
                
        log += f" | {elapsed:.1f}s"
        print(log)
        
    # 保存最终模型
    ckpt_path = resolve_path(train_cfg["checkpoint_path"])
    ensure_dir(os.path.dirname(ckpt_path))
    torch.save(model.state_dict(), ckpt_path)
    print(f"Final model saved: {ckpt_path}")
    if val_loader:
        print(f"Best model (val_loss={best_val_loss:.4f}): {best_path}")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1 GAT Training")
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    main(cfg)
```
