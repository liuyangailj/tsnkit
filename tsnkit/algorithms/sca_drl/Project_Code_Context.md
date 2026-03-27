## 📂 项目文件结构 (Project Structure)
```text
📁 /
    📄 phase1_lbsp_training.py
    📄 __init__.py
        📁 configs/
        📁 data/
            📄 data_generater.py
            📁 data_storm/
            📁 processed/
        📁 doc/
        📁 models/
        📁 phase1/
            📄 dataset.py
            📄 infer.py
            📄 init.py
            📄 model.py
            📄 train.py
        📁 phase2/
            📄 agent.py
            📄 environment.py
            📄 init.py
            📄 train.py
        📁 phase2_scheduling/
            📄 environment_ls_copy.py
            📄 init.py
            📄 scheduler_agent_transformer.py
            📄 train.py
            📄 utils.py
            📁 code_context/
            📁 models/
                📁 stage2_Mar19_12-05-38/
            📁 runs/
                📁 stage2_multipath_Mar19_12-05-38/
        📁 runners/
            📄 run_lbsp_inference.py
            📄 run_lbsp_training.py
```

# 💻 源代码上下文 (Source Code Context)

## File: `phase1_lbsp_training.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase1_lbsp_training.py`

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATConv
import numpy as np
import networkx as nx
import math
from itertools import islice

# 引入 TSNKit 核心
from tsnkit import core as utils

# ==========================================
# 1. 工具函数
# ==========================================
def k_shortest_paths(G, source, target, k, weight=None):
    """基于 NetworkX 计算 K 条最短路径。"""
    try:
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
    except nx.NetworkXNoPath:
        return []

def calculate_harmonic_prior(period_i, period_j):
    """计算两个流周期的谐波先验 (仅在推理阶段使用)"""
    gcd = math.gcd(int(period_i), int(period_j))
    lcm = (int(period_i) * int(period_j)) // gcd
    return gcd / lcm

# ==========================================
# 2. 数据集构建 (TSNPhase1Dataset)
# ==========================================
class TSNPhase1Dataset(Dataset):
    def __init__(self, task_path, topo_path, k_paths=3, edge_metric="kpair_prob"):
        self.task_path = task_path
        self.topo_path = topo_path
        self.k_paths = k_paths
        self.edge_metric = edge_metric
        super().__init__(root=None, transform=None, pre_transform=None)
        self.data = self.process_data()

    def process_data(self):
        print(f"[Phase1] Loading TSNKit data from {self.task_path}...")
        
        # 1. 使用 TSNKit 原生加载器
        try:
            tsn_net = utils.load_network(self.topo_path)
            tsn_tasks = utils.load_stream(self.task_path)
        except Exception as e:
            print(f"[Error] Failed to load files using tsnkit: {e}")
            raise e

        # 获取 NetworkX 图对象和流列表
        G_nx = tsn_net.net_nx
        streams = tsn_tasks.streams
        num_streams = len(streams)
        
        print(f"[Phase1] Processing {num_streams} streams for Conflict Graph...")

        x_list = []
        stream_ids = []
        stream_paths_links = [] # 存储每条流的 K 条候选路径的链路集合 list[list[set]]

        # 2. 提取节点特征并预计算 KSP
        for s in streams:
            # 兼容单个目的节点或多播(多目的节点)的情况
            target_node = s.dst[0] if isinstance(s.dst, list) else s.dst
            # 获取 stream 的唯一标识
            s_id = s.name if hasattr(s, 'name') else s._id
            stream_ids.append(s_id)
            
            # 节点特征: [size, log(period+1), log(deadline+1)]
            x_list.append([
                s.size, 
                math.log(s.period + 1), 
                math.log(s.deadline + 1)
            ])
            
            # 计算 K 条最短路径
            paths = k_shortest_paths(G_nx, s.src, target_node, self.k_paths)
            # 转换为链路的集合
            path_sets = [set(zip(p[:-1], p[1:])) for p in paths if len(p) >= 2]
            stream_paths_links.append(path_sets)

        x = torch.tensor(x_list, dtype=torch.float)
        # 特征归一化
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)

        # 3. 构建流冲突图 (计算 e_ij)
        edge_index = []
        edge_attr = []

        for i in range(num_streams):
            for j in range(i + 1, num_streams):
                paths_i = stream_paths_links[i]
                paths_j = stream_paths_links[j]
                
                K_i = len(paths_i)
                K_j = len(paths_j)
                if K_i == 0 or K_j == 0:
                    continue
                
                # 计算 K * K 组合的链路重叠次数
                overlap_count = 0
                for p_i in paths_i:
                    for p_j in paths_j:
                        # 优化：使用 isdisjoint 检测交集，速度更快
                        if not p_i.isdisjoint(p_j):
                            overlap_count += 1
                
                e_ij = overlap_count / (K_i * K_j)
                
                if e_ij > 0:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.append([e_ij])
                    edge_attr.append([e_ij])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0, 1), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.stream_ids = stream_ids
        data.streams = streams # 保存流对象，供后续谐波先验计算
        return data

    def len(self):
        return 1

    def get(self, idx):
        return self.data

# ==========================================
# 3. GAT 模型定义
# ==========================================
class GNNPartitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super().__init__()
        # edge_dim=1 表示显式注入单维度的 e_ij 到注意力机制
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, edge_dim=1, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, edge_dim=1, concat=False)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x  # 返回 Embedding: u_i

# ==========================================
# 4. 训练函数 (自监督拓扑重构)
# ==========================================
def train_phase1(model, data, optimizer, epochs=100, device='cpu'):
    model.train()
    data = data.to(device)
    num_nodes = data.num_nodes

    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = model(data)
        
        if data.edge_index.numel() > 0:
            src, dst = data.edge_index
            
            # 1. 正样本 Loss (物理连边的 e_ij)
            u_src, u_dst = embeddings[src], embeddings[dst]
            # 余弦相似度映射到 [0, 1]
            sim_pos = (F.cosine_similarity(u_src, u_dst) + 1.0) / 2.0
            # 新增：防止浮点数越界和log(0)问题的inf异常
            sim_pos = torch.clamp(sim_pos, min=1e-7, max=1.0 - 1e-6)
            
            e_ij = data.edge_attr.squeeze()
            # 目标是重构拓扑概率
            loss_pos = F.binary_cross_entropy(sim_pos, e_ij)
            
            # 2. 负采样 Loss (推开不冲突的流)
            num_neg = src.size(0) // 2  # 负采样比例可调
            neg_src = torch.randint(0, num_nodes, (num_neg,), device=device)
            neg_dst = torch.randint(0, num_nodes, (num_neg,), device=device)
            
            u_neg_src, u_neg_dst = embeddings[neg_src], embeddings[neg_dst]
            sim_neg = (F.cosine_similarity(u_neg_src, u_neg_dst) + 1.0) / 2.0
            
            # 新增：防止浮点数越界和log(0)问题的inf异常
            sim_neg = torch.clamp(sim_neg, min=1e-7, max=1.0 - 1e-7)
            
            # 负样本的目标相似度为 0
            loss_neg = F.binary_cross_entropy(sim_neg, torch.zeros_like(sim_neg))
            
            loss = loss_pos + loss_neg
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=device)
            
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs:03d} | Loss: {loss.item():.4f} (Pos: {loss_pos.item():.4f}, Neg: {loss_neg.item():.4f})")
```

## File: `__init__.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\__init__.py`

```python

```

## File: `data_generater.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\data\data_generater.py`

```python
import os
import pandas as pd
import networkx as nx
import random
    
def generate_industrial_dataset(num_tasks=100, save_dir="data_storm"):
    os.makedirs(save_dir, exist_ok=True)
    
    # =========================================================
    # 1. 制造完整的“工业控制环网 + 跨界捷径”拓扑 (9个SW + 27个ES)
    # =========================================================
    G = nx.Graph()
    
    # 1.1 添加骨干交换机 (SW: 0~8)
    ring_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (7, 8), (8, 0)
    ]
    chord_edges = [(2, 7)] # 中间的捷径
    G.add_edges_from(ring_edges + chord_edges)
    
    # 1.2 添加端系统 (ES: 9~35) 并连接到对应的 SW
    es_nodes = []
    es_id = 9
    for sw_id in range(0, 9): # 对于每个交换机 0~8        
        for _ in range(3): # 每个交换机外挂 3 个ES
            G.add_edge(es_id, sw_id)
            es_nodes.append(es_id)
            es_id += 1
            
    # 转换为双向的有向图
    G_dir = G.to_directed()
    
    topo_data = []
    for u, v in G_dir.edges():
        # 默认参数 (宽阔的高速公路)
        rate = 1      # 1Gbps
        t_prop = 0    
        
        # # 🌟 制造不对称性：中间的捷径 (2, 7) 极窄且有延迟
        # # 注意：双向都要设置！
        # if (u == 2 and v == 7) or (u == 7 and v == 2):
        
        #     t_prop = 50000 # 增加一点传播延迟
            
        topo_data.append({
            'link': f"({u}, {v})",
            'q_num': 8,       
            'rate': rate,        
            't_proc': 2000,   
            't_prop': t_prop       
        })
    
    topo_df = pd.DataFrame(topo_data)
    topo_path = os.path.join(save_dir, "0_topo.csv")
    topo_df.to_csv(topo_path, index=False)
    print(f"✅ 完整异构拓扑生成完毕: 36个节点 (9 SW + 27 ES), {G_dir.number_of_edges()}条有向边。")

    # =========================================================
    # 2. 制造流量风暴 Task：随机生成 N 条复杂的异构流
    # =========================================================
        
    # 周期池 (纳秒): 250us, 500us, 750us, 1ms, 2ms, 3ms    
    period_pool = [250000, 500000, 750000, 1000000, 2000000, 3000000]
    
    MIN_FLOWS = 60
    MAX_FLOWS = 90 
    
    print(f"🌪️ 正在生成 {num_tasks} 份高压任务考卷...")
      
    for task_id in range(1, num_tasks + 1):
        # 为了让模型学会适应不同数量的流，我们在 MIN_FLOWS 到 MAX_FLOWS 之间随机抽取
        num_flows = random.randint(MIN_FLOWS,MAX_FLOWS) 
        
        task_data = []
        for stream_id in range(num_flows):
            src = random.choice(es_nodes)
            dst = random.choice(es_nodes)
            
            # 过滤逻辑：判断它们是不是属于同一个交换机
            # (es_id - 9) // 3 就可以算出它挂在哪个 SW (0~8) 下面
            while (src - 9) // 3 == (dst - 9) // 3:
                dst = random.choice(es_nodes)        
                
            period = random.choice(period_pool)
            size = random.randint(100, 1500) # 模拟 100B 到 1500B 的以太网帧
            
            task_data.append({
                'stream': stream_id,
                'src': src,
                'dst': f"[{dst}]", # 严格对齐 tsnkit 的 List 字符串格式
                'size': size,
                'period': period,
                'deadline': period, # Deadline 默认等于周期
                'jitter': period
            })
            
        task_df = pd.DataFrame(task_data)
        task_path = os.path.join(save_dir, f"{task_id}_task.csv")
        task_df.to_csv(task_path, index=False)
    
    print(f"🎉 成功生成 {num_tasks} 份多路径拥塞测试数据！全部保存在 {save_dir}/ 目录下。")
    print(f"👉 【最大流数量】 (MAX_FLOWS): {MAX_FLOWS}")

if __name__ == "__main__":
    generate_industrial_dataset()
```

## File: `dataset.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase1\dataset.py`

```python
"""流冲突图数据集构建，基于 TSNKit 数据接口。"""

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

    task_path = resolve_path(data_cfg["task_file"])
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
    torch.save(emb_dict, out_emb)

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
"""Phase 1 GAT 自监督训练：拓扑重构损失。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch
import torch.nn.functional as F

from sca_drl.common.utils import (
    set_seed, load_config, resolve_path, get_device, ensure_dir
)
from sca_drl.phase1.dataset import TSNPhase1Dataset
from sca_drl.phase1.model import GNNPartitionModel

def train_phase1(model, data, optimizer, n_epochs=150, neg_sample_ratio=0.5,
                 device='cpu'):
    """自监督拓扑重构训练。

    正样本: 冲突边，重构 e_ij；负样本: 随机节点对，目标相似度为 0。
    """
    model.train()
    data = data.to(device)
    num_nodes = data.num_nodes
    has_edges = data.edge_index.numel() > 0

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        embeddings = model(data)

        if has_edges:
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

            loss = loss_pos + loss_neg
        else:
            loss = loss_pos = loss_neg = torch.tensor(0.0, device=device)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{n_epochs} | "
                  f"Loss {loss.item():.4f} (pos {loss_pos.item():.4f}, neg {loss_neg.item():.4f})")

def main(config: dict):
    """从配置字典启动完整训练流程。"""
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    task_path = resolve_path(data_cfg["task_file"])
    topo_path = resolve_path(data_cfg["topo_file"])

    # 构建冲突图
    dataset = TSNPhase1Dataset(task_path, topo_path, k_paths=data_cfg["k_paths"])
    data = dataset[0]
    print(f"Graph: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")

    # 模型 & 优化器
    device = get_device()
    model = GNNPartitionModel(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
        heads=model_cfg["heads"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    # 训练
    train_phase1(
        model, data, optimizer,
        n_epochs=train_cfg["n_epochs"],
        neg_sample_ratio=train_cfg.get("neg_sample_ratio", 0.5),
        device=device,
    )

    # 保存
    ckpt_path = resolve_path(train_cfg["checkpoint_path"])
    ensure_dir(os.path.dirname(ckpt_path))
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved: {ckpt_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1 GAT Training")
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    main(cfg)
```

## File: `agent.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2\agent.py`

```python
"""Transformer-PPO 调度智能体。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class CongestionAwareTransformer(nn.Module):
    """流序列编码器，将 [N, d_feature] 编码为 [N, d_model]。

    Args:
        d_feature: 输入特征维度（由环境决定）
        d_model: Transformer 隐藏维度
        nhead: 注意力头数
        num_layers: Transformer 层数
        d_ff: FFN 中间层维度
        dropout: Dropout 比率
    """

    def __init__(self, d_feature, d_model=64, nhead=4, num_layers=2,
                 d_ff=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_feature, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Args:
            x: [B, N, d_feature] 流观测序列

        Returns:
            [B, N, d_model] 编码后的流表示
        """
        x = self.input_proj(x)
        return self.encoder(x)


class PPOAgent:
    """PPO 智能体，Actor-Critic 共享 Transformer 编码器。

    Args:
        d_feature: 环境观测特征维度
        k_max: 最大候选路径数（路径动作维度）
        config: agent 配置字典（来自 phase2.yaml 的 agent 段）
        device: 计算设备
    """

    def __init__(self, d_feature, k_max, config, device='cpu'):
        self.device = device
        self.k_max = k_max

        # PPO 超参数
        ppo_cfg = config["ppo"]
        self.gamma = ppo_cfg["gamma"]
        self.eps_clip = ppo_cfg["eps_clip"]
        self.k_epochs = ppo_cfg["k_epochs"]
        self.entropy_coef = ppo_cfg["entropy_coef"]

        # Transformer 编码器（Actor-Critic 共享）
        tf_cfg = config["transformer"]
        d_model = tf_cfg["d_model"]
        self.encoder = CongestionAwareTransformer(
            d_feature=d_feature,
            d_model=d_model,
            nhead=tf_cfg["nhead"],
            num_layers=tf_cfg["num_layers"],
            d_ff=tf_cfg["d_ff"],
            dropout=tf_cfg.get("dropout", 0.1),
        ).to(device)

        # Actor: 流选择头 + 路径选择头
        self.stream_head = nn.Linear(d_model, 1).to(device)       # [N,1] → softmax → 选流
        self.route_head = nn.Linear(d_model, k_max).to(device)    # [N,K] → 选路

        # Critic: 全局价值头
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)

        # 统一优化器
        all_params = (
            list(self.encoder.parameters())
            + list(self.stream_head.parameters())
            + list(self.route_head.parameters())
            + list(self.value_head.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=ppo_cfg["learning_rate"])

        # 经验缓冲
        self.buffer = []

    def select_action(self, obs, scheduled_mask):
        """根据观测选择动作 (stream_idx, route_idx)。

        Args:
            obs: numpy [N, d_feature]
            scheduled_mask: numpy [N], True 表示已调度

        Returns:
            (stream_idx, route_idx), log_prob
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # [1, N, d_f]
        encoded = self.encoder(obs_t)  # [1, N, d_model]

        # 流选择
        stream_logits = self.stream_head(encoded).squeeze(-1)  # [1, N]
        mask = torch.BoolTensor(scheduled_mask).unsqueeze(0).to(self.device)
        stream_logits = stream_logits.masked_fill(mask, -1e9)
        stream_dist = Categorical(logits=stream_logits.squeeze(0))
        stream_idx = stream_dist.sample()

        # 路径选择（基于选中流的编码）
        stream_enc = encoded[0, stream_idx]  # [d_model]
        route_logits = self.route_head(stream_enc)  # [K]
        route_dist = Categorical(logits=route_logits)
        route_idx = route_dist.sample()

        log_prob = stream_dist.log_prob(stream_idx) + route_dist.log_prob(route_idx)

        return (stream_idx.item(), route_idx.item()), log_prob.item()

    def store_transition(self, obs, action, reward, log_prob, done):
        """存储一步经验。"""
        self.buffer.append((obs, action, reward, log_prob, done))

    def update(self):
        """PPO 策略更新。"""
        if not self.buffer:
            return 0.0

        # 解包经验
        obs_list, actions, rewards, old_log_probs, dones = zip(*self.buffer)

        # 计算折扣回报
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - float(d))
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

        total_loss = 0.0
        for _ in range(self.k_epochs):
            for i in range(len(self.buffer)):
                obs_t = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
                encoded = self.encoder(obs_t)

                # 重新计算 log_prob
                s_idx, r_idx = actions[i]
                stream_logits = self.stream_head(encoded).squeeze(-1).squeeze(0)
                stream_dist = Categorical(logits=stream_logits)

                stream_enc = encoded[0, s_idx]
                route_logits = self.route_head(stream_enc)
                route_dist = Categorical(logits=route_logits)

                new_log_prob = stream_dist.log_prob(torch.tensor(s_idx).to(self.device)) \
                             + route_dist.log_prob(torch.tensor(r_idx).to(self.device))
                entropy = stream_dist.entropy() + route_dist.entropy()

                # Value
                value = self.value_head(encoded.mean(dim=1)).squeeze()

                # PPO loss
                ratio = torch.exp(new_log_prob - old_log_probs[i])
                advantage = returns[i] - value.detach()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

                loss = -torch.min(surr1, surr2) \
                       + 0.5 * F.mse_loss(value, returns[i]) \
                       - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        self.buffer.clear()
        return total_loss

    def save(self, path):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'stream_head': self.stream_head.state_dict(),
            'route_head': self.route_head.state_dict(),
            'value_head': self.value_head.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.stream_head.load_state_dict(ckpt['stream_head'])
        self.route_head.load_state_dict(ckpt['route_head'])
        self.value_head.load_state_dict(ckpt['value_head'])
```

## File: `environment.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2\environment.py`

```python
"""TSN 调度环境：DRL 物理引擎 + Gymnasium 接口。"""

import numpy as np
import collections
import gymnasium as gym
from gymnasium import spaces

from tsnkit import core as utils
from tsnkit.algorithms.ls import LuShi


class DRL_PhysicsEngine:
    """TSN 调度物理引擎，管理时隙分配（First-Fit 贪心）。

    Args:
        net: TSNKit 网络对象
        streams: TSNKit 流列表
        k_max: 每条流最大候选路径数
    """

    def __init__(self, net, streams, k_max=3):
        self.net = net
        self.streams = streams
        self.k_max = k_max
        self.num_links = len(net.links)

        # 链路时隙占用表: {link_id: set(occupied_slots)}
        self.link_slot_usage = collections.defaultdict(set)
        # 每条流的候选路径（预计算）
        self.candidate_routes = self._precompute_routes()

    def _precompute_routes(self):
        """为每条流预计算 K 条最短候选路径。"""
        from sca_drl.phase1.dataset import k_shortest_paths
        routes = []
        for s in self.streams:
            target = s.dst[0] if isinstance(s.dst, list) else s.dst
            paths = k_shortest_paths(self.net.net_nx, s.src, target, self.k_max)
            routes.append(paths)
        return routes

    def try_allocate_stream(self, stream_idx, route_idx):
        """尝试为指定流分配时隙。

        Args:
            stream_idx: 流索引
            route_idx: 候选路径索引

        Returns:
            bool: 分配是否成功
        """
        stream = self.streams[stream_idx]
        routes = self.candidate_routes[stream_idx]
        if route_idx >= len(routes):
            return False
        route = routes[route_idx]
        return self._first_fit_allocate(stream, route)

    def _first_fit_allocate(self, stream, route):
        """First-Fit 贪心时隙分配。"""
        links = list(zip(route[:-1], route[1:]))
        period = stream.period
        size = stream.size

        # 寻找第一个可用起始时隙
        for t_start in range(period):
            slots_needed = range(t_start, t_start + size)
            conflict = False
            for link in links:
                link_id = str(link)
                if any(s % period in self.link_slot_usage[link_id] for s in slots_needed):
                    conflict = True
                    break
            if not conflict:
                # 分配成功，记录占用
                for link in links:
                    link_id = str(link)
                    for s in slots_needed:
                        self.link_slot_usage[link_id].add(s % period)
                return True
        return False

    def get_link_utilization(self):
        """返回各链路利用率字典。"""
        util = {}
        for link_id, slots in self.link_slot_usage.items():
            util[link_id] = len(slots)
        return util

    def reset(self):
        """重置所有时隙占用。"""
        self.link_slot_usage = collections.defaultdict(set)


class TSNSchedulingEnv(gym.Env):
    """TSN 联合路由调度 Gymnasium 环境。

    观测空间: 每条流一个 token，包含流特征 + 候选路径特征 + Phase1 embedding。
    动作空间: 选择一条流及其候选路径。

    Args:
        config: 完整配置字典（phase2.yaml 解析结果）
        phase1_embeddings: Phase 1 输出的流 embedding 字典，可选
    """

    def __init__(self, config, phase1_embeddings=None):
        super().__init__()
        env_cfg = config["environment"]
        data_cfg = config["data"]

        self.k_max = env_cfg["k_max"]
        self.max_steps = env_cfg["max_steps_per_episode"]
        self.reward_cfg = env_cfg["reward"]

        # Phase 1 embedding（frozen feature）
        self.phase1_embeddings = phase1_embeddings
        self.emb_dim = 0
        if phase1_embeddings is not None:
            sample = next(iter(phase1_embeddings.values()))
            self.emb_dim = len(sample)

        # 加载 TSNKit 数据
        from sca_drl.common.utils import resolve_path
        task_path = resolve_path(data_cfg["task_file"])
        topo_path = resolve_path(data_cfg["topo_file"])
        self.tsn_net = utils.load_network(topo_path)
        self.tsn_tasks = utils.load_stream(task_path)
        self.streams = self.tsn_tasks.streams
        self.num_streams = len(self.streams)

        # 物理引擎
        self.engine = DRL_PhysicsEngine(self.tsn_net, self.streams, self.k_max)

        # 特征维度: 基础特征 + K条路径特征 + embedding
        self.d_base = 5  # [size, period, deadline, is_scheduled, priority]
        self.d_route = 3 * self.k_max  # 每条路径: [hop_count, available, congestion]
        self.d_feature = self.d_base + self.d_route + self.emb_dim

        # Gym 空间定义
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_streams, self.d_feature),
            dtype=np.float32,
        )
        # 动作 = (流索引, 路径索引)
        self.action_space = spaces.MultiDiscrete([self.num_streams, self.k_max])

        # 状态追踪
        self.scheduled = np.zeros(self.num_streams, dtype=bool)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        self.scheduled = np.zeros(self.num_streams, dtype=bool)
        self.current_step = 0
        return self._get_observation(), {}

    def step(self, action):
        stream_idx, route_idx = action[0], action[1]
        self.current_step += 1

        # 已调度的流不能重复选择
        if self.scheduled[stream_idx]:
            reward = self.reward_cfg["failure"]
            terminated = False
            truncated = self.current_step >= self.max_steps
            return self._get_observation(), reward, terminated, truncated, {}

        # 尝试分配
        success = self.engine.try_allocate_stream(stream_idx, route_idx)

        if success:
            self.scheduled[stream_idx] = True
            reward = self.reward_cfg["success"]
        else:
            reward = self.reward_cfg["failure"]

        # 终止条件
        all_done = self.scheduled.all()
        if all_done:
            reward += self.reward_cfg["completion_bonus"]

        terminated = all_done
        truncated = self.current_step >= self.max_steps

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self):
        """构建观测矩阵 [num_streams, d_feature]。"""
        obs = np.zeros((self.num_streams, self.d_feature), dtype=np.float32)

        for i, s in enumerate(self.streams):
            offset = 0

            # 基础特征
            obs[i, 0] = s.size / 1500.0
            obs[i, 1] = np.log(s.period + 1) / 10.0
            obs[i, 2] = np.log(s.deadline + 1) / 10.0
            obs[i, 3] = float(self.scheduled[i])
            obs[i, 4] = getattr(s, 'priority', 0) / 7.0
            offset = self.d_base

            # 候选路径特征
            routes = self.engine.candidate_routes[i]
            for k in range(self.k_max):
                if k < len(routes):
                    obs[i, offset + k * 3] = len(routes[k]) / 10.0  # hop count
                    obs[i, offset + k * 3 + 1] = 1.0                # available
                    obs[i, offset + k * 3 + 2] = 0.0                # congestion placeholder
                # else: 保持 0（无此路径）
            offset += self.d_route

            # Phase 1 embedding（frozen feature）
            if self.phase1_embeddings is not None:
                sid = s.name if hasattr(s, 'name') else s._id
                if sid in self.phase1_embeddings:
                    emb = self.phase1_embeddings[sid]
                    obs[i, offset:offset + self.emb_dim] = emb

        return obs
```

## File: `init.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2\init.py`

```python
"""Phase 2: Transformer-PPO 联合路由调度。"""
```

## File: `train.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2\train.py`

```python
"""Phase 2 Transformer-PPO 训练入口。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch

from sca_drl.common.utils import (
    set_seed, load_config, resolve_path, get_device, ensure_dir
)
from sca_drl.phase2.environment import TSNSchedulingEnv
from sca_drl.phase2.agent import PPOAgent


def load_phase1_outputs(config):
    """加载 Phase 1 的 embedding（如果配置中指定了路径）。"""
    data_cfg = config.get("data", {})
    emb_path = data_cfg.get("phase1_embeddings_pt")
    if emb_path:
        full_path = resolve_path(emb_path)
        if os.path.exists(full_path):
            print(f"Loading Phase 1 embeddings: {full_path}")
            return torch.load(full_path, map_location="cpu")
        else:
            print(f"Phase 1 embeddings not found at {full_path}, skipping.")
    return None


def main(config: dict):
    """从配置字典启动 Phase 2 训练。"""
    train_cfg = config["training"]
    device = get_device()

    # 加载 Phase 1 输出
    phase1_emb = load_phase1_outputs(config)

    # 初始化环境
    env = TSNSchedulingEnv(config, phase1_embeddings=phase1_emb)
    print(f"Env: {env.num_streams} streams, d_feature={env.d_feature}, "
          f"emb_dim={env.emb_dim}")

    # 初始化智能体
    agent = PPOAgent(
        d_feature=env.d_feature,
        k_max=env.k_max,
        config=config["agent"],
        device=device,
    )

    # 训练循环
    num_episodes = train_cfg["num_episodes"]
    log_interval = train_cfg["log_interval"]
    save_interval = train_cfg["save_interval"]

    episode_rewards = []

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(obs, env.scheduled)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, log_prob, done)
            obs = next_obs
            ep_reward += reward

        # Episode 结束，PPO 更新
        loss = agent.update()
        episode_rewards.append(ep_reward)

        if ep % log_interval == 0:
            avg_r = sum(episode_rewards[-log_interval:]) / log_interval
            scheduled_rate = env.scheduled.sum() / env.num_streams
            print(f"Episode {ep:5d}/{num_episodes} | "
                  f"Avg Reward: {avg_r:8.2f} | "
                  f"Scheduled: {scheduled_rate:.1%} | "
                  f"Loss: {loss:.4f}")

        if ep % save_interval == 0:
            ckpt_path = resolve_path(train_cfg["checkpoint_path"])
            ensure_dir(os.path.dirname(ckpt_path))
            agent.save(ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # 最终保存
    ckpt_path = resolve_path(train_cfg["checkpoint_path"])
    ensure_dir(os.path.dirname(ckpt_path))
    agent.save(ckpt_path)
    print(f"Training complete. Model saved: {ckpt_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2 PPO Training")
    parser.add_argument("--config", default="configs/phase2.yaml")
    args = parser
```

## File: `environment_ls_copy.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2_scheduling\environment_ls_copy.py`

```python
import numpy as np
import collections
import gymnasium as gym
from gymnasium import spaces

# 假设你的项目中可以通过以下方式导入 tsnkit 的 ls 算法类
# 如果路径不同，请根据你的项目结构调整
from tsnkit.algorithms.ls import ls 
# 在 environment_ls_copy.py 的顶部 imports 区域加上：
from tsnkit.core._constants import T_SLOT

# =====================================================================
# 模块一：物理引擎外壳 (DRL_PhysicsEngine)
# 职责：继承 tsnkit 的 ls.py，劫持其底层计算能力，提供单步交互接口
# =====================================================================
class DRL_PhysicsEngine(ls):
    def __init__(self, task, topo, workers=1):
        """
        初始化物理引擎，对接 tsnkit 的 Task 和 Network 对象
        """
        super().__init__(workers)
        self.task = task
        self.topo = topo
        self.task_routes = {s: self.topo.get_all_path(s.src, s.dst) for s in self.task.streams}
        # 初始化底层的状态账本
        self.deep_clear()

    def deep_clear(self):
        """
        彻底清空底层物理世界的残留物，用于 Episode 重置
        """
        self._delay = {}
        # 初始化空白的全局 GCL (基于连续区间)
        self._result = {l: [] for l in self.topo.links}
        self._paths = {}
        self._offset = {}

    def try_allocate_agent_action(self, task_stream, path):
        """
        核心物理执行器：接收单步动作，调用 tsnkit 检验，成功则写入账本
        返回: (是否成功: bool, 瓶颈链路利用率: float)
        """
        # 1. 呼叫 tsnkit 最核心的物理碰撞检测神仙函数
        inject_time = self.find_inject_time(task_stream, path)
        
        if inject_time == -1:
            # 物理法则判定：无处安放，直接驳回
            return False, 1.0 
            
        # 2. 如果成功，复用 ls.py 的写入逻辑，彻底锁定整个 LCM 内的时隙
        self._delay[task_stream] = self.get_nw_delay(task_stream, path)
        self._paths[task_stream] = path
        self._offset[task_stream] = inject_time

        _prev_end = inject_time
        max_util = 0.0

        for l in path.links:
            _start = _prev_end
            _end = _start + task_stream.get_t_trans(l)
            _prev_end = _start + l.t_proc + task_stream.get_t_trans(l)

            # 在全网 LCM 的时间轴上，铺满这个流的每一个复现周期！
            for k in task_stream.get_frame_indexes(self.task.lcm):
                # 记录格式: (start, end, queue)
                self._result[l].append((_start + k * task_stream.period, _end + k * task_stream.period, 0))
                
            # 必须重新排序，因为 tsnkit 的 match_time 依赖有序列表
            self._result[l].sort(key=lambda x: x[0], reverse=False)

            # 3. 顺手计算当前链路的准确利用率 (用于给 Agent 算惩罚)
            total_occupied = sum([entry[1] - entry[0] for entry in self._result[l]])
            util = total_occupied / self.task.lcm
            if util > max_util:
                max_util = util

        return True, max_util


# =====================================================================
# 模块二：强化学习环境 (TSNEnv)
# 职责：Agent 的翻译官，状态图的绘制者，马尔可夫决策过程的控制台
# =====================================================================
class TSNEnv(gym.Env):
    def __init__(self, env_config):
        super(TSNEnv, self).__init__()
        
        # 1. 解析物理数据，构建网络拓扑        
        self.topo = env_config['topo']   
        self.MAX_FLOWS = env_config.get('max_flows', 130)        
        self.K_MAX = env_config.get('k_max', 5) 
        
        # 获取所有边，并固化索引 (用于生成固定维度的 g_global)
        self.edges = self.topo.links 
        self.num_edges = len(self.edges)
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.edges)}
        
        # 2. 空间与维度裁剪设定 (Padding & Truncation)        
        # 动作空间永远固定为 MAX_FLOWS * K_MAX
        self.action_space = spaces.Discrete(self.MAX_FLOWS * self.K_MAX)
        
        # 设定观测窗口大小 (方案 B：Fixed-Window Rasterization)
        # 比如观察未来 1024 个量子化时隙
        self.T_slot = T_SLOT
        
        # =====借用初始 task 算出固定 W （第一个task的流的LCM）
        initial_task = env_config['task'] # 提前把初始考卷拿出来看一眼
        max_acceptable_W = env_config.get('max_obs_window', 10000) 
        
        # 提取初始考卷的 LCM
        if hasattr(initial_task, 'lcm'):
            actual_lcm_slots = initial_task.lcm
        elif hasattr(initial_task, '_lcm'):
            actual_lcm_slots = initial_task._lcm
        else:
            actual_lcm_slots = int(np.lcm.reduce([f.period for f in initial_task.streams]))

        # W 在这里被彻底定死！后续 load_new_task 绝对不许再改 self.W！
        if actual_lcm_slots <= max_acceptable_W:
            self.W = actual_lcm_slots
            print(f"🌍 [Env 初始化] 神经网络输入维度已锁定，观测窗口 W 设为初始 LCM: {self.W} 槽")
        else:
            self.W = max_acceptable_W
            print(f"⚠️ [Env 警告] 初始 LCM 过大！视野被强制截断为: {self.W} 槽") 
        #=================================================================
        
        # 特征维度计算 (4个基础属性 + 跳数 + 重叠度 + 各路利用率 + 先验 = 5 + 3*K_MAX)
        self.d_feature = 5 + 3 * self.K_MAX  
        
        # 3. 观测空间永远固定为 MAX_FLOWS
        self.observation_space = spaces.Dict({
            "flow_tokens": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.MAX_FLOWS, self.d_feature), dtype=np.float32
            ),
            "global_snapshot": spaces.Box(
                low=0.0, high=1.0, 
                shape=(self.num_edges * self.W,), dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=0, high=1, 
                shape=(self.MAX_FLOWS * self.K_MAX,), dtype=np.bool_
            )
        })
        
        self.lambda_2 = env_config.get('lambda_2', 0.5)
        self.xi = env_config.get('xi', 2.0)
        self.alpha = env_config.get('alpha', 10.0)
        
        # 加载初始任务
        self.load_new_task(env_config['task'])
        
    def load_new_task(self, task):
        """
        动态更换考卷的接口。用于 Stage 2 训练。
        """
        self.task = task
        self.flows = self.task.streams
        
        # 截断防御：如果流数量超出了我们的 MAX_FLOWS 容忍度，强行截断
        if len(self.flows) > self.MAX_FLOWS:
            print(f"⚠️ 警告: 任务流数量({len(self.flows)}) 超过 MAX_FLOWS({self.MAX_FLOWS})，执行截断！")
            self.flows = self.flows[:self.MAX_FLOWS]
            
        self.num_flows = len(self.flows)
        # 重新挂载物理引擎
        self.physics_engine = DRL_PhysicsEngine(self.task, self.topo)
        
        # 重新计算归一化基准
        self.max_period = max([f.period for f in self.flows]) if self.flows else 1.0
        self.max_size = max([f.size for f in self.flows]) if self.flows else 1.0
        self.max_deadline = max([f.deadline for f in self.flows]) if self.flows else 1.0
        self.max_hops = len(self.topo.nodes)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.flow_states = {}
        for i in range(self.num_flows):
            self.flow_states[i] = {'status': 1}
            
        # 核心：命令物理引擎进行深度清理，抹除上一个 Episode 的痕迹
        self.physics_engine.deep_clear()        
        return self._get_observation(), {}

    def _get_observation(self):
        """
        抽取物理世界的状态，包装成 Transformer 友好的 Tokens 和 Rasterized Window。
        """
        
        # ================= 🔍 照妖镜调试代码 =================
        # 我们只在第一把游戏的第 1 步打印一次，防止刷屏
        if not hasattr(self, 'debug_printed') and self.steps == 1:
            print("\n" + "="*50)
            print("🚀 [DEBUG INFO] 深入物理引擎底层探查！")
            print(f"1. 当前设定的窗口大小 W: {self.W}")
            print(f"2. 当前设定的时间槽 T_slot: {self.T_slot}")
            
            # 抽查第 0 条流的真实属性
            f_test = self.flows[0]
            print(f"3. 抽查流 0 的真实周期 (period): {f_test.period}")
            # print(f"   (期望值：如果是 250us 且 T_slot=1000, 这里应该是 250！)")
            print(f"4. 抽查流 0 的真实包大小 (size): {f_test.size}")
            
            # 抽查物理引擎已经排进去的真实区间
            if self.physics_engine._result:
                # 随便找一条有数据的边
                sample_edge = list(self.physics_engine._result.keys())[0]
                sample_intervals = self.physics_engine._result[sample_edge]
                print(f"5. 物理引擎真实分配的区间示例 (边 {sample_edge}):")
                print(f"   {sample_intervals[:5]} ...")
                
                # 检查有没有超出 W=2000 的区间被丢弃？
                max_end_time = max([iv[1] for iv in sample_intervals]) if sample_intervals else 0
                print(f"6. 当前这条边上，被分配的最晚结束时间: {max_end_time}")
                if max_end_time > self.W:
                    print("   🚨 [严重警告] 存在区间超出了 W=2000，Agent 的眼睛被蒙住了！")
            else:
                print("5. 当前没有任何流被排进去 (_result 为空)")
                
            print("="*50 + "\n")
            self.debug_printed = True
        # =====================================================
        
        obs_tokens = []
        pending_flows = [i for i in range(self.num_flows) if self.flow_states[i]['status'] == 1]
        
        # 步骤 A：构建动态重叠度热度图 (基于候选路径)
        link_heat_map = collections.defaultdict(int)
        for flow_idx in pending_flows:
            flow_obj = self.flows[flow_idx]
            routes = self.physics_engine.task_routes[flow_obj]
            # 仅取前 K_MAX 条路径参与热度计算
            for path in routes[:self.K_MAX]:
                for link in path.links:
                    link_heat_map[link] += 1
                    
        # 步骤 B：生成 Token 矩阵
        for i in range(self.num_flows):
            f = self.flows[i]
            status = self.flow_states[i]['status']
            
            norm_period = f.period / self.max_period
            norm_size = f.size / self.max_size
            norm_deadline = f.deadline / self.max_deadline
            
            H_i = np.zeros(self.K_MAX, dtype=np.float32)
            O_i = np.zeros(self.K_MAX, dtype=np.float32)
            # 🌟 新增：动态物理瓶颈利用率
            U_i = np.zeros(self.K_MAX, dtype=np.float32) 
            
            routes = self.physics_engine.task_routes[f]
            num_valid_paths = min(len(routes), self.K_MAX)
            
            for k in range(num_valid_paths):
                path = routes[k]
                links = path.links
                H_i[k] = len(links) / self.max_hops
                
                if status == 1:
                    overlap_score = sum(link_heat_map[link] for link in links)
                    O_i[k] = overlap_score - len(links)
                    
                    # 计算路径上的平均利用率
                    # 🌟 2. 新增：动态物理利用率 (Bottleneck Util)
                    path_link_utils = []
                    for link in links:
                        # 抄底层物理引擎的作业：直接读取真实账本 self._result
                        # self._result[link] 里存的是 [(start, end, queue), ...]
                        
                        # 把这条链路上所有被占用的时间段加起来
                        total_occupied = sum([entry[1] - entry[0] for entry in self.physics_engine._result[link]])
                        
                        # 除以 LCM 得到绝对真实的利用率！
                        util = total_occupied / self.physics_engine.task.lcm
                        
                        path_link_utils.append(util)
                    
                    # 这条候选路径的瓶颈，就是它所有链路中最堵的那条
                    U_i[k] = max(path_link_utils) if path_link_utils else 0.0

                    
            # padding 补齐 K_MAX，多余位置保持 0
            token = np.concatenate([
                [norm_period, norm_size, norm_deadline, status], 
                H_i, 
                O_i, 
                U_i,
                [0.0] # 假设无 prior，填 0
            ]).astype(np.float32)
            
            obs_tokens.append(token)
            
        # 步骤 B.5：Padding 幽灵流填充
        num_padding = self.MAX_FLOWS - self.num_flows
        for _ in range(num_padding):
            # 创建全 0 特征，但重点是把 status (索引 3) 设为 -2.0
            # 这样 Agent 的 _get_action_mask 就会立刻把它屏蔽掉！
            ghost_token = np.zeros(self.d_feature, dtype=np.float32)
            ghost_token[3] = -2.0 
            obs_tokens.append(ghost_token)

        # 步骤 C：实施方案 B —— 固定窗口时隙栅格化 (Rasterization)
        # 建立一个干净的 [边数, 窗口大小] 0/1 矩阵
        g_window = np.zeros((self.num_edges, self.W), dtype=np.float32)
        
        for link, intervals in self.physics_engine._result.items():
            if link not in self.edge_to_idx:
                continue
            link_idx = self.edge_to_idx[link]
            
            for start, end, _ in intervals:
                # 1. 物理时间 -> 离散格子索引
                # 只截取位于 0 到 W 窗口内的时隙部分
                s_idx = int(start)                
                e_idx = int(end)                
                
                # 3.边界截断保护
                s_idx = max(0, s_idx)
                e_idx = min(self.W, e_idx)
                
                # 4. 绘制到矩阵
                if s_idx < e_idx:
                    g_window[link_idx, s_idx:e_idx] = 1.0
                        
        # 🌟 新增：生成动作掩码 (Action Mask)
        # 动作空间大小是 num_flows * K_MAX
        action_mask = np.zeros(self.MAX_FLOWS * self.K_MAX, dtype=np.bool_)
        
        for i in range(self.num_flows):
            if self.flow_states[i]['status'] == 1: # 只有待调度的流才能选
                f = self.flows[i]
                routes = self.physics_engine.task_routes[f]
                num_valid_paths = min(len(routes), self.K_MAX)
                
                for k in range(num_valid_paths):
                    action_idx = i * self.K_MAX + k
                    action_mask[action_idx] = True # 这个动作是合法的！
                    
        # 把 mask 一并打包返回
        return {
            'flow_tokens': np.array(obs_tokens, dtype=np.float32),
            'global_snapshot': g_window.flatten(), # 展平该窗口，喂给 Agent
            'action_mask': action_mask # 🌟 塞进 obs 字典里
        }

    def step(self, action):
            """
            环境推演：解密动作，调用物理引擎，结算稠密奖励。
            """
            flow_idx = action // self.K_MAX
            path_idx = action % self.K_MAX
            
            # 🛡️ 致命拦截：如果在你的环境中经常触发这个，说明你的 Masking 代码写崩了！
            if flow_idx >= self.num_flows or self.flow_states[flow_idx]['status'] != 1:
                # 给出极大的负惩罚，并截断，防止死循环
                return self._get_observation(), -10.0, True, False, {"error": "Masking Failed: Invalid flow selected!"}
                
            f = self.flows[flow_idx]
            routes = self.physics_engine.task_routes[f]
            
            # 🛡️ 路径拦截
            if path_idx >= len(routes):
                return self._get_observation(), -5.0, True, False, {"error": "Masking Failed: Invalid path selected!"}
                
            path = routes[path_idx]
            
            # ==========================================
            # 🔓 架构师的万能开锁器：破解 tsnkit 封装的 Path
            # ==========================================
            hop_count = 1  # 默认保底跳数
            
            if hasattr(path, 'edges'):
                # 如果它封装了边列表，跳数 = 边的数量
                hop_count = len(path.edges)
            elif hasattr(path, 'links'):
                hop_count = len(path.links)
            elif hasattr(path, 'nodes'):
                # 如果它封装了节点列表，跳数 = 节点数 - 1
                hop_count = len(path.nodes) - 1
            else:
                # 🚨 如果全都没命中，直接启动 X光机，打印它的底裤！
                print(f"\n🚨 [架构师拦截] 未知 Path 结构！")
                print(f"   对象内容: {path}")
                print(f"   包含的属性和方法: {dir(path)}")
                
                # 既然不知道多长，为了让训练不崩溃，先强行给个跳数 2
                hop_count = 2 
                
            # 确保跳数绝对不能小于 1
            hop_count = max(1, hop_count)            
            
            hop_penalty = 0.05 * hop_count            
            
            # --- 核心：将真实对象抛给底层 tsnkit 引擎 ---
            is_success, util_score = self.physics_engine.try_allocate_agent_action(f, path)
            
            # 🌟 革命性的奖励重构 (稠密 & 宽容)
            if is_success:
                self.flow_states[flow_idx]['status'] = 0 
                # 成功排入，基础分 +1.0。
                # 引入负载均衡惩罚：如果这条路很堵(util_score高)，稍微扣一点分 (比如 1.0 - 0.5*0.8 = 0.6分)
                # 鼓励 Agent 去找不堵的、空闲的路径！
                # 加入防御性 max(..., 0.01)，防止扣分超标变成惩罚成功！
                reward = max(1.0 - (self.lambda_2 * util_score) - hop_penalty, 0.01)
                
            else:
                self.flow_states[flow_idx]['status'] = -1 
                # 失败了，不要给巨大的 -xi。我们给一个极其微小的惩罚，或者干脆给 0！
                # 这里的哲学是：没排进去，你只是没拿到那 1.0 分而已。不要因为物理无解而让梯度崩溃。
                reward = 0.0  # 👈 核心改变！你也可以写 -0.1，但绝对不能是 -2.0 这种大数字。
                
            self.steps += 1
            
            all_processed = all(state['status'] != 1 for state in self.flow_states.values())
            terminated = all_processed or self.steps >= self.num_flows
            
            info = {
                'flow_idx': flow_idx,
                'is_allocated': is_success,
                'bottleneck_util': util_score if is_success else 1.0,
                'hop_count': hop_count if is_success else 0  # 🌟 把跳数塞给上帝（你）看！
            }
            
            # 🌟 删除了原本在 terminated 时追加的 self.alpha * p_success 
            # 因为步骤上的 +1.0 累积起来，本身就已经在最大化全局成功率了！
            if terminated:
                success_count = sum(1 for state in self.flow_states.values() if state['status'] == 0)
                p_success = success_count / self.num_flows
                info['group_success_rate'] = p_success
                
                # ==========================================
                # 🏆 架构师的终极大奖 (Global Jackpot)
                # ==========================================
                if p_success == 1.0:
                    # 如果 100% 调度成功，在最后一步把大奖砸给它！
                    # self.alpha 目前配置里是 10.0。你可以根据刺激程度去调大它（比如 20.0 或 50.0）
                    jackpot_bonus = self.alpha 
                    reward += jackpot_bonus
                    
                    # (可选) 在终端里稍微撒个花，方便你观察它有没有“开窍”
                    print(f"\n🎉 [环境撒花] 达成 100% 完美调度！发放全局通关大奖: +{jackpot_bonus}")
                
            return self._get_observation(), reward, terminated, False, info
```

## File: `init.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2_scheduling\init.py`

```python

```

## File: `scheduler_agent_transformer.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2_scheduling\scheduler_agent_transformer.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

# =====================================================================
# 核心工具：CleanRL 标配的正交初始化函数
# 作用：保证深层网络梯度传播的稳定性，防止梯度消失或爆炸
# =====================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# =====================================================================
# 模块一：神经网络架构 (Congestion-Aware Transformer)
# =====================================================================
class CongestionAwareTransformer(nn.Module):
    def __init__(self, num_flows, k_max, d_feature, global_dim, d_model=64, n_heads=4, n_layers=3):
        super().__init__()
        self.num_flows = num_flows
        self.k_max = k_max
        self.d_model = d_model
        
        # 1. 状态嵌入层 (State Embedding) - 严格对齐论文 ReLU 公式
        self.flow_embedder = nn.Sequential(
            layer_init(nn.Linear(d_feature, d_model)),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        self.global_embedder = nn.Sequential(
            layer_init(nn.Linear(global_dim, d_model * 2)),
            nn.ReLU(),
            layer_init(nn.Linear(d_model * 2, d_model)),
            nn.LayerNorm(d_model)
        )
        
        # 2. 拥塞感知核心大脑 (Transformer Encoder 共享躯干)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4,
            batch_first=True,  
            norm_first=True    
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 3. Actor 策略输出头 (Actor Head)
        # 注意：Actor 最后一层初始化 std=0.01，让初始动作概率分布尽量均匀，鼓励早期探索
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(d_model, d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(d_model, k_max), std=0.01) 
        )
        
        # 4. Critic 价值评估头 (Critic Head)
        # 注意：Critic 最后一层初始化 std=1.0
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(d_model, d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(d_model, 1), std=1.0)
        )

    def forward(self, flow_tokens, global_snapshot):
        # ==========================================
        # 1. 构造 Transformer 的 Attention Mask (防污染)
        # ==========================================
        is_padding = (flow_tokens[..., 3] == -2.0) # [B, MAX_FLOWS]
        
        # global token 永远是真实的，不能被 mask 掉
        global_mask = torch.zeros((flow_tokens.shape[0], 1), dtype=torch.bool, device=flow_tokens.device)
        padding_mask = torch.cat([is_padding, global_mask], dim=1) # [B, MAX_FLOWS + 1]
        
        # ==========================================
        # 2. Embedding 与 纯净的 Transformer 编码
        # ==========================================
        
        flow_emb = self.flow_embedder(flow_tokens)         
        global_emb = self.global_embedder(global_snapshot).unsqueeze(1)         
        seq_emb = torch.cat([flow_emb, global_emb], dim=1) 
        
        # 🌟 核心：必须传给 Transformer，从源头掐断幽灵流的干扰！
        out_seq = self.transformer(seq_emb, src_key_padding_mask=padding_mask)                
        
        # 拆分特征
        out_flow = out_seq[:, :-1, :]       # [B, MAX_FLOWS, d_model] 所有Batch，从头取到倒数第二个token，所有特征
        global_repr = out_seq[:, -1, :]     # [B, d_model] 所有Batch，取最后一个token（全局token），所有特征
        
        # ==========================================
        # 3. 融合你的优雅版 Mask-aware Pooling (算 Critic)
        # ==========================================
        valid_mask = (~is_padding).float().unsqueeze(-1) # [B, MAX_FLOWS, 1]
        
        masked_flow_repr = out_flow * valid_mask
        sum_flow = masked_flow_repr.sum(dim=1)              # [B, d_model]
        num_flow = valid_mask.sum(dim=1).clamp(min=1.0)     # [B, 1]
        pooled_repr = sum_flow / num_flow                   # [B, d_model]
        
        # 🌟 把全局视野加回来，让 Critic 看到拥堵情况
        final_critic_state = pooled_repr + global_repr
        value = self.critic_head(final_critic_state)        # [B, 1]
        
        # ==========================================
        # 4. 算 Actor (策略分布)
        # ==========================================
        logits_2d = self.actor_head(out_flow)              
        logits = logits_2d.reshape(-1, self.num_flows * self.k_max) 
        
        return logits, value


# =====================================================================
# 模块二：PPO 智能体与训练引擎 (CleanRL Style)
# =====================================================================
class PPOAgent:
    def __init__(self, env, d_model=64, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_coef=0.2, ent_coef=0.0, vf_coef=0.5, k_epochs=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_flows = env.MAX_FLOWS
        self.k_max = env.K_MAX
        self.d_feature = env.d_feature
        self.global_dim = env.observation_space['global_snapshot'].shape[0]
        
        self.network = CongestionAwareTransformer(
            num_flows=self.num_flows, k_max=self.k_max, 
            d_feature=self.d_feature, global_dim=self.global_dim,
            d_model=d_model
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.k_epochs = k_epochs

# 🚨 彻底删除旧的 _get_action_mask 函数！我们不再自己算了！

    def get_action_and_value(self, obs, action=None):
        flow_tokens = torch.FloatTensor(obs['flow_tokens']).to(self.device) 
        g_global = torch.FloatTensor(obs['global_snapshot']).to(self.device)
        mask = torch.BoolTensor(obs['action_mask']).to(self.device) # 🌟 接收环境小抄
        
        # 智能升维
        if len(flow_tokens.shape) == 2:
            flow_tokens = flow_tokens.unsqueeze(0) 
        if len(g_global.shape) == 1:
            g_global = g_global.unsqueeze(0)
        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0)  # 🌟 Mask 同步升维
            
        logits, value = self.network(flow_tokens, g_global)
        
        if mask.sum() == 0:
            mask = torch.ones_like(mask, dtype=torch.bool)
            
        logits = logits.masked_fill(~mask, -1e8) 
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            return action.item(), probs.log_prob(action), probs.entropy(), value.squeeze()
        else:
            # 兼容 update 时传进来的 Tensor
            return action, probs.log_prob(action), probs.entropy(), value.squeeze()

    def update(self, rollouts):
        b_flow_tokens = torch.FloatTensor(np.array(rollouts['flow_tokens'])).to(self.device)
        b_global = torch.FloatTensor(np.array(rollouts['global_snapshot'])).to(self.device)
        b_actions = torch.LongTensor(rollouts['actions']).to(self.device)
        b_logprobs = torch.FloatTensor(rollouts['logprobs']).to(self.device)
        b_rewards = torch.FloatTensor(rollouts['rewards']).to(self.device)
        b_values = torch.FloatTensor(rollouts['values']).to(self.device)        
        # 🌟 修复核心：装载历史 Mask！
        b_masks = torch.BoolTensor(np.array(rollouts['action_masks'])).to(self.device) 
        b_dones = torch.BoolTensor(np.array(rollouts['dones'])).to(self.device) # 🌟 新增：装载历史 dones 信息
        
        batch_size = len(b_rewards)
        
        with torch.no_grad(): 
            advantages = torch.zeros_like(b_rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(batch_size)):
                if t == batch_size - 1:
                    nextnonterminal = 0.0
                    nextvalues = 0.0 
                else:
                    # 🌟 核心救命代码：如果是 True(1.0)，这步就是 0.0 (切断联系)
                    # 如果是 False(0.0)，这步就是 1.0 (保持相连)
                    if b_dones[t]: # 🌟 如果这一轮结束了
                        nextnonterminal = 0.0 # 🌟 切断联系 
                    else:
                        nextnonterminal = 1.0 # 🌟 保持相连
                        
                    nextvalues = b_values[t + 1]
                    # nextnonterminal = 1.0 
                    # nextvalues = b_values[t + 1]
                delta = b_rewards[t] + self.gamma * nextvalues * nextnonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + b_values

        # ===== 提取到这里！在整个 800 步的 batch 级别做归一化 =====
        advantages_std = advantages.std()
        if not (torch.isnan(advantages_std) or advantages_std == 0.0):
            advantages = (advantages - advantages.mean()) / (advantages_std + 1e-8)
        # ==========================================================
        
        b_inds = np.arange(batch_size)
        for epoch in range(self.k_epochs): 
            np.random.shuffle(b_inds) 
            
            for start in range(0, batch_size, 64):   
                end = start + 64
                mb_inds = b_inds[start:end]
                
                if len(mb_inds) <= 1:
                    continue
                
                logits, newvalues = self.network(b_flow_tokens[mb_inds], b_global[mb_inds])
                
                # 🌟 拿出当时的护盾，继续保护现在的网络计算
                mask = b_masks[mb_inds] 
                logits = logits.masked_fill(~mask, -1e8)
                
                probs = Categorical(logits=logits)
                newlogprob = probs.log_prob(b_actions[mb_inds]) 
                entropy = probs.entropy().mean() 
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = advantages[mb_inds]
                # std = mb_advantages.std()
                # if not (torch.isnan(std) or std == 0.0):
                #     mb_advantages = (mb_advantages - mb_advantages.mean()) / (std + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalues.squeeze() - returns[mb_inds]) ** 2).mean()
                loss = pg_loss - self.ent_coef * entropy + v_loss * self.vf_coef

                self.optimizer.zero_grad()  
                loss.backward()             
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) 
                self.optimizer.step()      
                
        return pg_loss.item(), v_loss.item(), entropy.item()
```

## File: `train.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2_scheduling\train.py`

```python
import sys
import os
import time
import glob
import random
import numpy as np
import torch
from tsnkit import core as utils 
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from environment_ls_copy import TSNEnv
from scheduler_agent_transformer import PPOAgent

def train():
    print("="*60)
    print("🚀 SCA-DRL Phase 2: 终极泛化训练引擎启动 (Padding & Masking) 🚀")
    print("="*60)

    # 1. 自动扫描数据风暴文件夹
    data_dir = "../data/data_storm"
    topo_path = os.path.join(data_dir, "0_topo.csv")
    task_files = glob.glob(os.path.join(data_dir, "*_task.csv"))
    
    if not os.path.exists(topo_path) or len(task_files) == 0:
        print("❌ 找不到数据！请先运行 generate_data.py 生成 data_storm 文件夹！")
        return

    print(f"[1/4] 成功发现多路径网格拓扑与 {len(task_files)} 份随机流量考卷！")
    topo = utils.load_network(topo_path)

    # 随意加载一个 task 用于初始化环境
    initial_task = utils.load_stream(task_files[0])

    env_config = {
        'task': initial_task,
        'topo': topo,
        'max_flows': 130,          
        'k_max': 5,                
        'obs_window_size': 5000,   
        'lambda_2': 0.5,           
        'xi': 2.0,                 
        'alpha': 10.0              
    }

    print("[2/4] 初始化 TSNEnv (容量: 100流) 与 Transformer PPOAgent...")
    # 初始学习率
    initial_lr = 3e-4
    
    env = TSNEnv(env_config)
    agent = PPOAgent(env=env, d_model=64, lr=initial_lr, k_epochs=4, clip_coef=0.2, ent_coef=0.01)
    
    # ==========================================
    # 🛡️ MLOps: 实验追踪与日志持久化系统
    # ==========================================
    # 1. 生成全局唯一的实验时间戳
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    
    # 2. 为本次实验建立【专属】的 TensorBoard 和 模型保存 文件夹
    run_dir = f"./runs/stage2_multipath_{current_time}"
    model_dir = f"./models/stage2_{current_time}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 3. 拦截器：把所有的 print() 输出同时打在屏幕上，并实时存入硬盘 txt
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()  # 👈 核心救命代码：强制每写一行就存入硬盘，就算立刻断电死机，日志也在！

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    # 将系统标准输出和报错输出全部接管到我们的 log 文件中
    sys.stdout = Logger(os.path.join(run_dir, "train_log.txt"))
    sys.stderr = sys.stdout  

    # 4. 启动 TensorBoard 记录器
    writer = SummaryWriter(log_dir=run_dir)

    print(f"🚀 SCA-DRL 训练引擎启动！")
    print(f"📁 本次实验日志路径: {run_dir}/train_log.txt")
    print(f"💾 本次模型保存路径: {model_dir}")
    
    total_iterations = 200         
    episodes_per_iter = 8         
    
    print(f"[3/4] 训练规划: 共 {total_iterations} 轮, 每轮收集 {episodes_per_iter} 份考卷")
    print("[4/4] ⚔️ 面对风暴吧！Transformer！\n")

    start_time = time.time()
    
    for iteration in range(1, total_iterations + 1):
        # ==========================================
        # 📉 架构师的微调法宝：线性学习率衰减 (Linear LR Annealing)
        # ==========================================
        # 计算当前剩余比例 (第 1 轮是 1.0，第 100 轮是 0.0)
        frac = 1.0 - (iteration - 1.0) / total_iterations
        # 计算当前应该使用的学习率
        lr_now = frac * initial_lr
        
        # 强行霸道地修改 PyTorch 优化器里的学习率
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = lr_now        
        
        rollouts = { 'flow_tokens': [], 'global_snapshot': [], 'action_masks': [], 'actions': [], 
                     'logprobs': [], 'rewards': [], 'values': [], 'dones': [] } # 新增 'dones' 用于存储每步是否结束的信息
        
        iter_rewards = []
        iter_success_rates = []
        iter_avg_hops = [] # 新增：每轮的平均跳数统计

        # --- 循环刷题阶段 ---
        for ep in range(episodes_per_iter):
            # 🌟 核心：随机抽一张考卷，并让环境更新它的物理引擎！
            random_task_file = random.choice(task_files)
            new_task = utils.load_stream(random_task_file)   
            # new_task = utils.load_stream(task_files[1])         
            env.load_new_task(new_task)
            
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_hops = []
            
            # 哪怕最大容量是 100，我们这一局实际只需要调度真实的 env.num_flows 次
            for step in range(env.num_flows):
                rollouts['flow_tokens'].append(obs['flow_tokens'])
                rollouts['global_snapshot'].append(obs['global_snapshot'])
                rollouts['action_masks'].append(obs['action_mask'])
                
                action, logprob, entropy, value = agent.get_action_and_value(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                rollouts['actions'].append(action)
                rollouts['logprobs'].append(logprob.item())
                rollouts['rewards'].append(reward)
                rollouts['values'].append(value.item())
                rollouts['dones'].append(terminated) # 🌟存储这一轮是否结束的信息,这步是不是把大结局记下来
                
                ep_reward += reward
                obs = next_obs
                
                # 🌟 新增：如果这步排流成功了，把 info 里传出来的跳数记下来
                if info.get('is_allocated', False) and 'hop_count' in info:
                    ep_hops.append(info['hop_count'])
                
                if terminated:
                    if 'group_success_rate' in info:
                        iter_success_rates.append(info['group_success_rate'])
                    
                    # 🌟 新增：计算这个 Episode 的平均跳数，存进全局列表
                    if ep_hops:
                        iter_avg_hops.append(sum(ep_hops) / len(ep_hops))
                    else:
                        iter_avg_hops.append(0.0)
                    
                    break
            
            iter_rewards.append(ep_reward)

        # --- PPO 梯度回传 ---
        pg_loss, v_loss, ent = agent.update(rollouts)

        avg_reward = np.mean(iter_rewards)
        avg_success = np.mean(iter_success_rates) * 100 if iter_success_rates else 0.0
        avg_hop = np.mean(iter_avg_hops) if iter_avg_hops else 0.0 # 🌟 新增：算出这 8 局的整体平均跳数
        
        # print(f"Iter {iteration:03d} | 成功率: {avg_success:5.1f}% | Reward: {avg_reward:7.2f} | P_Loss: {pg_loss:6.3f} | V_Loss: {v_loss:6.3f} | Ent: {ent:5.3f}")
        print(f"Iter {iteration:03d} | 总流数：{env.num_flows} | 成功率: {avg_success:5.1f}% | 均跳数: {avg_hop:4.2f} | Reward: {avg_reward:7.2f} | P_Loss: {pg_loss:6.3f} | V_Loss: {v_loss:6.3f} | Ent: {ent:5.3f}")

        writer.add_scalar("Metrics/1_Success_Rate", avg_success, iteration)
        writer.add_scalar("Metrics/2_Avg_Reward", avg_reward, iteration)
        writer.add_scalar("Metrics/3_Avg_Hops", avg_hop, iteration) # 🌟 新增：画出跳数下降的完美曲线！
        
        writer.add_scalar("Metrics/4_Learning_Rate", lr_now, iteration) # 🌟 新增：监控学习率的下降轨迹
        
        writer.add_scalar("Loss/1_Policy_Loss", pg_loss, iteration)
        writer.add_scalar("Loss/2_Value_Loss", v_loss, iteration)
        writer.add_scalar("Loss/3_Entropy", ent, iteration)

        # 每 50 轮或者最后一轮，保存一次模型
        if iteration % 50 == 0 or iteration == total_iterations:
            # 注意这里：把硬编码的 "./models" 换成了我们刚才动态生成的 model_dir
            save_path = os.path.join(model_dir, f"ppo_stage2_iter_{iteration}.pth")
            torch.save(agent.network.state_dict(), save_path)
            print(f"💾 模型已保存至: {save_path}")

    writer.close()
    print("\n🎉 训练完美收官！")

if __name__ == "__main__":
    train()
```

## File: `utils.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2_scheduling\utils.py`

```python

```

## File: `run_lbsp_inference.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\runners\run_lbsp_inference.py`

```python
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from sklearn.cluster import SpectralClustering
import pickle

# --- 路径设置 (与你原有结构一致) ---
curr_dir = os.path.dirname(os.path.abspath(__file__))
algorithm_dir = os.path.dirname(os.path.dirname(curr_dir))
tsnkit_dir = os.path.dirname(algorithm_dir)
sys.path.append(algorithm_dir)

from sca_drl.phase1_lbsp_training import (
    TSNPhase1Dataset, 
    GNNPartitionModel, 
    calculate_harmonic_prior
)
from tsnkit import core as utils

def compute_affinity_matrix(embeddings, streams, alpha=0.6):
    """
    实现白皮书公式: W_ij = alpha * s_gnn + (1 - alpha) * s_prior
    """
    num_streams = len(streams)
    W = np.zeros((num_streams, num_streams), dtype=np.float32)
    
    # 将 embedding 转为 tensor 算相似度
    #  O(N^2) 矩阵乘法计算全局余弦相似度
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    emb_norm = F.normalize(emb_tensor, p=2, dim=1)
    # s_gnn_matrix 的形状是 [N, N]，值域直接映射到了 [0, 1]
    s_gnn_matrix = ((torch.mm(emb_norm, emb_norm.t()) + 1.0) / 2.0).numpy()
    
    # 2. 融合时域谐波先验
    for i in range(num_streams):
        W[i, i] = 1.0
        for j in range(i + 1, num_streams):
            s_prior = calculate_harmonic_prior(streams[i].period, streams[j].period)
            W_ij = alpha * s_gnn_matrix[i, j] + (1 - alpha) * s_prior
            W[i, j] = W[j, i] = W_ij
            
    return W

def main():    
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 在 main() 开头调用：
    set_seed(42)
    
    # 1. 统一超参数配置
    config = {
        'k_paths': 3,
        'model': {'input_dim': 3, 'hidden_dim': 16, 'output_dim': 16, 'heads': 4},
        'inference': {
            'model_path': '../results/checkpoints/gnn_phase1_new.pth',
            'n_clusters': 4,
            'alpha': 0.6,
            'out_csv': '../results/phase1_groups.csv',
            'out_emb': '../results/phase1_embeddings.pt'  # 新增：供 Phase 2 调用的特征字典
        }
    }
    
    # 2. 数据加载
    tsn_data_dir = os.path.join(tsnkit_dir, 'test/data')
    task_file = os.path.join(tsn_data_dir, '1_task.csv')
    topo_file = os.path.join(tsn_data_dir, '1_topo.csv')
        
    print("Building Phase 1 Graph Data...")
    dataset = TSNPhase1Dataset(task_file, topo_file, k_paths=config['k_paths'])
    data = dataset[0]
    streams = data.streams
    
    # 3. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNPartitionModel(**config['model']).to(device)
    model.load_state_dict(torch.load(config['inference']['model_path'], map_location=device))
    model.eval()

    # 4. 前向传播获取 Embedding
    data = data.to(device)
    with torch.no_grad():
        print("Running GNN forward pass...")
        embeddings = model(data).cpu().numpy()  # [Num_Streams, 16]
        
    # 5. 计算混合亲和度矩阵与谱聚类
    alpha = config['inference']['alpha']
    W = compute_affinity_matrix(embeddings, streams, alpha)
    
    print(f"Executing Spectral Clustering (k={config['inference']['n_clusters']})...")
    sc = SpectralClustering(
        n_clusters=config['inference']['n_clusters'], 
        affinity='precomputed', 
        random_state=42
    )
    labels = sc.fit_predict(W)
    
    # ==========================================
    # 6. 数据落盘 (为 TSNKit 路由和 Phase 2 DRL 准备接口)
    # ==========================================
    # 6.1 保存 CSV 分组结果
    import csv
    os.makedirs(os.path.dirname(config['inference']['out_csv']), exist_ok=True)
    with open(config['inference']['out_csv'], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream_id", "group_id"])
        for sid, gid in zip(data.stream_ids, labels):
            w.writerow([sid, int(gid)])
            
    # 6.2 保存 Embedding 字典供 Phase 2 Transformer 调用！
    embedding_dict = {sid: embeddings[i] for i, sid in enumerate(data.stream_ids)}
    torch.save(embedding_dict, config['inference']['out_emb'])
    
    print(f"Phase 1 finished! Groups saved to {config['inference']['out_csv']}")
    print(f"Embeddings saved to {config['inference']['out_emb']} for Phase 2!")

if __name__ == '__main__':
    main()
```

## File: `run_lbsp_training.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\runners\run_lbsp_training.py`

```python
import sys
import os
import torch
import random
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))
algorithm_dir = os.path.dirname(os.path.dirname(curr_dir))
tsnkit_dir = os.path.dirname(algorithm_dir)
sys.path.append(algorithm_dir)

from sca_drl.phase1_lbsp_training import (
    TSNPhase1Dataset, 
    GNNPartitionModel, 
    train_phase1
)
from tsnkit import core as utils

def main():
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(42)
    
    # 对齐论文的统一超参表
    config = {
        'k_paths': 3,
        'model': {
            'input_dim': 3,  # [size, log(period), log(deadline)]
            'hidden_dim': 16,
            'output_dim': 16,
            'heads': 4
        },
        'train': {
            'lr': 0.005,
            'epochs': 150, # GNN 训练很快，可以适当提高 epoch
            'save_path': '../results/checkpoints/gnn_phase1_new.pth'
        }
    }
    
    # 读取数据
    tsn_data_dir = os.path.join(tsnkit_dir, 'test/data')
    task_file = os.path.join(tsn_data_dir, '1_task.csv')
    topo_file = os.path.join(tsn_data_dir, '1_topo.csv')
    
    if not os.path.exists(task_file) or not os.path.exists(topo_file):
        print(f"Error: Task file or Topology file not found in {tsn_data_dir}")
        return
    
    # 使用 TSNKit 原生加载器读取数据
    print(f"Building Graph Dataset with K={config['k_paths']}...")
    dataset = TSNPhase1Dataset(task_file, topo_file, k_paths=config['k_paths'])
    data = dataset[0]
    
    print(f"Graph built: {data.num_nodes} nodes, {data.edge_index.size(1)} physical conflict edges.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNPartitionModel(**config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])

    # 开始训练
    print(f"Training GAT Topology Reconstructor on {device}...")
    train_phase1(model, data, optimizer, epochs=config['train']['epochs'], device=device)

    # 保存权重
    os.makedirs(os.path.dirname(config['train']['save_path']), exist_ok=True)
    torch.save(model.state_dict(), config['train']['save_path'])
    print(f"Model saved to {config['train']['save_path']}")

if __name__ == '__main__':
    main()
```
