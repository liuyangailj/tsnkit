## 📂 项目文件结构 (Project Structure)
```text
📁 /
    📄 __init__.py
        📁 configs/
        📁 data/
            📄 data_generater.py
            📁 data_storm/
            📁 processed/
        📁 doc/
        📁 models/
            📁 phase2_Mar27_22-45-35/
            📁 phase2_Mar28_12-02-31/
        📁 phase1/
            📄 batch_infer.py
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
            📁 models/
                📁 phase2_Mar28_12-21-14/
            📁 runs/
                📁 phase2_ppo_Mar28_12-21-14/
        📁 runs/
```

# 💻 源代码上下文 (Source Code Context)

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
    
def generate_industrial_dataset(num_tasks, save_dir="data_storm"):
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
    chord_edges = [(2, 7),(3, 6)] # 中间的捷径
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
    
    MIN_FLOWS = 60
    MAX_FLOWS = 90            
    
    # # 周期池 (纳秒): 250us, 500us, 750us, 1ms, 2ms, 3ms    
    # period_pool = [250000, 500000, 750000, 1000000, 2000000, 3000000]
    # 周期池 (纳秒): 200us, 400us, 800us, 600us 1.5ms, 3ms    
    period_pool = [200000, 400000, 800000, 600000, 1500000, 3000000]
    
    # 核心节点
    core_es_nodes = [9,10,11,21,22,23] # SW_0 和 SW_4
    edge_es_nodes = [n for n in es_nodes if n not in core_es_nodes]    

    print(f"🌪️ 正在生成 {num_tasks} 份高压任务考卷...")       
    all_generated_tasks = [] # 用于暂存所有卷子
    
    for task_id in range(1, num_tasks + 1):
        # 为了让模型学会适应不同数量的流，我们在 MIN_FLOWS 到 MAX_FLOWS 之间随机抽取
        num_flows = random.randint(MIN_FLOWS,MAX_FLOWS) 
        
        task_data = []
        for stream_id in range(num_flows):            
            
            # 🛡️ 策略 3：80/20 法则制造严重的空间拥塞
            if random.random() < 0.8:
                # 80% 核心业务流：向心或离心
                if random.random() < 0.5:
                    src = random.choice(edge_es_nodes)
                    dst = random.choice(core_es_nodes)
                else:
                    src = random.choice(core_es_nodes)
                    dst = random.choice(edge_es_nodes)
            else:
                # 20% 背景杂波流
                src = random.choice(edge_es_nodes)
                dst = random.choice(edge_es_nodes)
            
            # 过滤逻辑：判断它们是不是属于同一个交换机，如果是则重新选目的节点
            while (src - 9) // 3 == (dst - 9) // 3 or src == dst:
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
        
        # 不直接存盘，而是打包放进列表
        all_generated_tasks.append({
            'num_flows': num_flows, 
            'data': task_data})
    
    # =========================================================
    # 🌟 3. 核心大招：按难度排序并执行 8:1:1 分层切分！
    # =========================================================
    print("⚖️ 正在执行严谨的难度分层切分 (Stratified Split 8:1:1)...")
    # 3.1 按照流数量排序
    all_generated_tasks.sort(key=lambda x: x['num_flows'])
    # 3.2 切分为训练集、验证集、测试集
    train_count, val_count, test_count = 0, 0, 0
    
    for i, task_bundle in enumerate(all_generated_tasks):
        mol_val = i % 10
        if mol_val < 8:
            prefix = ""
            train_count += 1
        elif mol_val == 8:
            prefix = "val_"
            val_count += 1
        else:
            prefix = "test_"
            test_count += 1
            
        file_name = f"{prefix}{i+1}_task.csv"
        pd.DataFrame(task_bundle['data']).to_csv(os.path.join(save_dir, file_name), index=False)
    
    print(f"✅ 完美切分完毕！生成了 {train_count} 份训练集，{val_count} 份验证集，{test_count} 份测试集。")
    print("📈 它们的难度分布已经达到了统计学上的完全一致！")

if __name__ == "__main__":
    # 建议至少生成 500 份，这样验证集和测试集各有 50 份，大数定律生效！
    generate_industrial_dataset(num_tasks=500)
```

## File: `batch_infer.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase1\batch_infer.py`

```python
import sys
import os
import glob
import time
import csv

import torch
import numpy as np
from sklearn.cluster import SpectralClustering

# 确保能找到项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from sca_drl.common.utils import load_config, resolve_path, get_device
from sca_drl.phase1.dataset import TSNPhase1Dataset
from sca_drl.phase1.model import GNNPartitionModel
from sca_drl.phase1.infer import compute_affinity_matrix

def batch_inference(config_path="configs/phase1.yaml"):
    print("=" * 60)
    print("🌉 Phase 1 -> Phase 2: 全量数据桥接工程启动")
    print("=" * 60)

    # 1. 加载配置与模型
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    infer_cfg = cfg["inference"]
    
    device = get_device()
    topo_path = resolve_path(data_cfg["topo_file"])
    k_paths = data_cfg.get("k_paths", 3)
    n_clusters = infer_cfg.get("n_clusters", 8) # 默认分成8组
    alpha = infer_cfg.get("alpha", 0.6)

    # 初始化并加载最优模型
    model = GNNPartitionModel(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
        heads=model_cfg["heads"],
    ).to(device)
    
    # 🌟 自动寻找 best 模型
    ckpt_path = resolve_path(cfg["training"]["checkpoint_path"]).replace(".pth", "_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 找不到最优模型 {ckpt_path}，请确认 Phase 1 已经训练并保存！")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"✅ GNN 最优大脑已挂载: {os.path.basename(ckpt_path)}")

    # 2. 搜集所有的考卷 (Train 和 Val 都要处理)
    task_dir = resolve_path(data_cfg["task_dir"])
    train_files = glob.glob(os.path.join(task_dir, "[0-9]*_task.csv"))
    val_files = glob.glob(os.path.join(task_dir, "val_*_task.csv"))
    all_task_files = train_files + val_files
    
    print(f"📂 共发现 {len(all_task_files)} 份考卷需要打标，准备大批量前向推理...")
    time.sleep(1)

    # 3. 流水线作业开始
    for i, task_path in enumerate(all_task_files):
        base_name = os.path.basename(task_path).replace(".csv", "") # e.g., "1_task" or "val_1_task"
        dir_name = os.path.dirname(task_path)
        
        out_emb_path = os.path.join(dir_name, f"{base_name}_emb.pt")
        out_group_path = os.path.join(dir_name, f"{base_name}_group.csv")
        
        # 如果已经生成过了，就跳过（方便中断后继续）
        if os.path.exists(out_emb_path) and os.path.exists(out_group_path):
            continue

        sys.stdout.write(f"\r⏳ 正在处理 [{i+1}/{len(all_task_files)}]: {base_name} ...")
        sys.stdout.flush()

        # [步骤 A] 构建图数据 (单张实时构建)
        # 注意：这里会产生寻路开销，但这只是离线生成一次，稍微等几分钟是值得的
        dataset = TSNPhase1Dataset(task_path, topo_path, k_paths=k_paths)
        data = dataset[0].to(device)

        # [步骤 B] GNN 瞬间算出高维 Embedding
        with torch.no_grad():
            embeddings = model(data).cpu().numpy()

        # [步骤 C] 谱聚类，计算分组
        periods = data.periods.tolist()
        W = compute_affinity_matrix(embeddings, periods, alpha=alpha)
        sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42)
        labels = sc.fit_predict(W)

        # [步骤 D] 存盘 (写入硬盘，供 Phase 2 读取)
        # 1. 保存 Embedding (转成字典格式 sid -> tensor)
        emb_dict = {sid: embeddings[idx] for idx, sid in enumerate(data.stream_ids)}
        emb_dict_torch = {k: torch.tensor(v) for k, v in emb_dict.items()}
        torch.save(emb_dict_torch, out_emb_path)

        # 2. 保存分组 Group CSV
        with open(out_group_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["stream_id", "group_id"])
            for sid, gid in zip(data.stream_ids, labels):
                writer.writerow([sid, int(gid)])

    print("\n🎉 批量桥接数据生成完毕！你的 Phase 2 粮草已经全部堆满仓库！")
    print("=" * 60)

if __name__ == "__main__":
    batch_inference()
```

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

    def __init__(self, task_files, topo_path, k_paths=3, cache_name="dataset"):
            self.task_files = task_files
            self.topo_path = topo_path
            self.k_paths = k_paths
            
            # 🌟 核心提速秘籍：构建缓存文件路径
            if len(task_files) > 0:
                data_dir = os.path.dirname(task_files[0])
                # 根据文件数量和名字生成独一无二的缓存名，防止冲突
                self.cache_path = os.path.join(data_dir, f"{cache_name}_{len(task_files)}_k{k_paths}.pt")
            else:
                self.cache_path = None

            super().__init__(root=None, transform=None, pre_transform=None)
            
            # 🌟 拦截器：如果硬盘上已经有了存好的图，直接秒读！
            if self.cache_path and os.path.exists(self.cache_path):
                print(f"⚡ [极速缓存] 瞬间加载 {len(task_files)} 份图数据: {os.path.basename(self.cache_path)}")
                self._graphs = torch.load(self.cache_path, weights_only=False)
            else:
                # 只有第一次运行，或者你删了缓存文件，才会乖乖去算
                self._graphs = self._build_all_graphs()
                if self.cache_path:
                    print(f"💾 [固化数据] 正在将处理好的图数据写入硬盘，下次秒开: {os.path.basename(self.cache_path)}")
                    torch.save(self._graphs, self.cache_path)

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
from datetime import datetime # 用于日志时间戳

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter # 用于训练日志记录

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
    train_dataset = MultiInstanceDataset(train_files, topo_path, k_paths, cache_name="train")
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
             val_dataset = MultiInstanceDataset(val_files, topo_path, k_paths, cache_name="val")
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
    
    # 🌟 3. 初始化 TensorBoard MLOps 记录器
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    run_dir = resolve_path(f"./runs/phase1_gnn_{current_time}")
    ensure_dir(run_dir)
    writer = SummaryWriter(log_dir=run_dir)
    print(f"📊 TensorBoard 日志已开启: tensorboard --logdir={run_dir}")

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
        
        # 🌟 写入训练 Loss
        writer.add_scalar("Loss/1_Train_Loss", train_loss, epoch)
        log = f"Epoch {epoch:03d}/{n_epochs} | Train Loss: {train_loss:.4f}"
        
        # 🌟 验证评估
        if val_loader:
            val_loss = evaluate(model, val_loader, neg_ratio, device)
            
            # 🌟 写入验证 Loss
            writer.add_scalar("Loss/2_Val_Loss", val_loss, epoch)
            log += f" | Val Loss: {val_loss:.4f}"            
            
            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = resolve_path(
                    train_cfg["checkpoint_path"].replace(".pth", "_best.pth")
                )
                ensure_dir(os.path.dirname(best_path))
                torch.save(model.state_dict(), best_path)
                log += " ★ [Best Model Saved]"
                
        log += f" | {elapsed:.1f}s"
        print(log)
        
    # 保存最终模型
    ckpt_path = resolve_path(train_cfg["checkpoint_path"])
    ensure_dir(os.path.dirname(ckpt_path))
    torch.save(model.state_dict(), ckpt_path)
    writer.close() # 关闭 TensorBoard 记录器
    
    print("-" * 65)
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

## File: `agent.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2\agent.py`

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
    def __init__(self, num_flows, k_max, d_feature, global_dim, d_model, n_heads, n_layers):
        super().__init__()
        self.num_flows = num_flows
        self.k_max = k_max
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
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
    def __init__(self, env, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 从配置字典中提取
        trans_cfg = config.get("transformer", {})
        ppo_cfg = config.get("ppo", {}) 
        
        d_model = trans_cfg.get("d_model", 64)
        n_heads = trans_cfg.get("nhead", 4)
        n_layers = trans_cfg.get("num_layers", 3)
        
        lr = ppo_cfg.get("learning_rate", 2.5e-4)        
        self.gamma = ppo_cfg.get("gamma", 0.99)
        self.clip_coef = ppo_cfg.get("eps_clip", 0.2)
        self.ent_coef = ppo_cfg.get("entropy_coef", 0.01)
        self.vf_coef = ppo_cfg.get("vf_coef", 0.5)
        self.k_epochs = ppo_cfg.get("k_epochs", 4)
        self.gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
                
        self.num_flows = env.MAX_FLOWS
        self.k_max = env.K_MAX
        self.d_feature = env.d_feature # 🌟 这里会自动读取包含 Embedding 后的真实维度
        self.global_dim = env.observation_space['global_snapshot'].shape[0]
        
        self.network = CongestionAwareTransformer(
            num_flows=self.num_flows, k_max=self.k_max, 
            d_feature=self.d_feature, global_dim=self.global_dim,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

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

## File: `environment.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2\environment.py`

```python
import os
import pandas as pd
import numpy as np
import collections
import gymnasium as gym
from gymnasium import spaces
import torch

# 假设你的项目中可以通过以下方式导入 tsnkit 的 ls 算法类
# 如果路径不同，请根据你的项目结构调整
from tsnkit.algorithms.ls import ls 
# 在 environment_ls_copy.py 的顶部 imports 区域加上：
from tsnkit.core._constants import T_SLOT
# from sca_drl.common.utils import resolve_path

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
    def __init__(self, env_config: dict, phase1_embeddings: dict = None):
        super(TSNEnv, self).__init__()
        
        # [🔧 Config重构] 读取配置
        self.topo = env_config['topo']
        env_params = env_config.get('environment',{})
        
        self.MAX_FLOWS = env_params.get('max_flows', 130)        
        self.K_MAX = env_params.get('k_max', 5) 
        
        # [🔧 Config重构] 读取奖励权重
        reward_cfg = env_params.get('reward', {}) 
        self.omega_0 = reward_cfg.get('all_success', 10.0)
        self.omega_1 = reward_cfg.get('one_success', 1.0)
        self.omega_2 = reward_cfg.get('hop_penalty', 0.01)
        self.omega_3 = reward_cfg.get('util_penalty', 0.1)
        self.omega_4 = reward_cfg.get('failure', 0.0)       
        
        # [🌟 Embedding接入] 动态获取 embedding 维度
        self.phase1_embeddings = phase1_embeddings
        if self.phase1_embeddings is not None and len(self.phase1_embeddings) > 0:
            sample_emb = next(iter(self.phase1_embeddings.values()))
            self.emb_dim = sample_emb.shape[0] if hasattr(sample_emb, 'shape') else len(sample_emb)
        else:
            self.emb_dim = 1 # 降级保护：如果没有传入，维度退回 1（等同于老版的 [0.0]） 
        
        # 获取所有边，并固化索引 (用于生成固定维度的 g_global)全局快照
        self.edges = self.topo.links 
        self.num_edges = len(self.edges)
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.edges)}
        
        # 动作空间与维度裁剪设定 (Padding & Truncation)，固定为 MAX_FLOWS * K_MAX        
        self.action_space = spaces.Discrete(self.MAX_FLOWS * self.K_MAX)
        
        # 设定观测窗口大小，比如观察未来 1024 个量子化时隙 
        self.T_slot = T_SLOT
        
        # 借用初始 task 算出固定 W （第一个task的流的LCM）
        initial_task = env_config['task'] # 提前把初始考卷拿出来看一眼
        max_acceptable_W = env_config.get('max_obs_window', 10000) 
        
        # ===========提取初始考卷的 LCM=================
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
        
        # 特征维度计算 (4个基础属性 + 跳数 + 链路(重叠度 + 各路利用率) + phase1_embedding)
        self.d_feature = 4 + 3 * self.K_MAX + self.emb_dim 
        
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
                
        # 加载初始任务
        # self.load_new_task(env_config['task'])
        self.load_new_task(env_config['task'], env_config.get('task_file'))
        
    def load_new_task(self, task, task_file_path=None):
        """动态更换考卷，并读取 Phase 1 的分组标签"""
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
        
        # 🌟 新增：读取教导主任的分组表
        self.flow_groups = {} # 字典: {流ID: 组号}
        if task_file_path is not None:
            group_csv_path = task_file_path.replace(".csv", "_group.csv")
            if os.path.exists(group_csv_path):
                df_group = pd.read_csv(group_csv_path)
                # 假设 csv 里有 'stream_id' 和 'group_id' 两列
                for _, row in df_group.iterrows():
                    self.flow_groups[str(row['stream_id'])] = int(row['group_id'])
            else:
                print(f"⚠️ 未找到分组表 {group_csv_path}，所有流默认归为 0 组！")
                
        # 🌟 2. 新增：动态读取这张考卷专属的 Embedding 特征
        self.phase1_embeddings = {}
        if task_file_path is not None:
            emb_pt_path = task_file_path.replace(".csv", "_emb.pt")
            if os.path.exists(emb_pt_path):
                # 读入这套卷子专属的 16 维特征字典 {sid: tensor}
                self.phase1_embeddings = torch.load(emb_pt_path, map_location='cpu', weights_only=False)
    

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

            # 🌟 核心修改：动态获取该流的 Embedding
            stream_id = getattr(f, 'name', str(getattr(f, 'id', i)))
            if self.phase1_embeddings is not None and stream_id in self.phase1_embeddings:
                emb = self.phase1_embeddings[stream_id]
                # 处理如果 phase1 吐出的是 GPU Tensor 的情况
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
            else:
                emb = np.zeros(self.emb_dim, dtype=np.float32)
                    
            # padding 补齐 K_MAX，多余位置保持 0
            token = np.concatenate([
                [norm_period, norm_size, norm_deadline, status], 
                H_i, 
                O_i, 
                U_i,
                emb # 假设无 prior，填 0
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
        
        # 🌟 1. 动态探测当前活跃组别
        # pending_flows 之前已经在上面算过了：pending_flows = [i for i in range(self.num_flows) if self.flow_states[i]['status'] == 1]
        current_active_group = 0
        if pending_flows:
            # 找到所有待排流的所属组，取最小的那个作为当前活动组
            active_groups = []
            for i in pending_flows:
                f = self.flows[i]
                sid = getattr(f, 'name', str(getattr(f, 'id', i)))
                active_groups.append(self.flow_groups.get(sid, 0)) # 查不到默认给 0
            current_active_group = min(active_groups)
        
        # 🌟 2. 实施三重门禁    
        for i in range(self.num_flows):
            f = self.flows[i]
            sid = getattr(f, 'name', str(getattr(f, 'id', i)))
            my_group = self.flow_groups.get(sid, 0)
            
            # 三重门禁：只有 status=1 的流，且属于当前活动组，才能被考虑；其他一律屏蔽
            
            if self.flow_states[i]['status'] == 1 and my_group == current_active_group: # 只有待调度的流才能选
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
            
            hop_penalty = self.omega_2 * hop_count #跳数/延时惩罚           
            
            # --- 核心：将真实对象抛给底层 tsnkit 引擎 ---
            is_success, util_score = self.physics_engine.try_allocate_agent_action(f, path)
            
            # 🌟 革命性的奖励重构 (稠密 & 宽容)
            if is_success:
                self.flow_states[flow_idx]['status'] = 0 
                # 成功排入，基础分 +1.0。
                # 引入负载均衡惩罚：如果这条路很堵(util_score高)，稍微扣一点分 (比如 1.0 - 0.5*0.8 = 0.6分)
                # 鼓励 Agent 去找不堵的、空闲的路径！
                # 加入防御性 max(..., 0.01)，防止扣分超标变成惩罚成功！
                reward = max(self.omega_1 - (self.omega_3 * util_score) - hop_penalty, 0.01)
                
            else:
                self.flow_states[flow_idx]['status'] = -1 
                # 失败了，不要给巨大的 -xi。我们给一个极其微小的惩罚，或者干脆给 0！
                # 这里的哲学是：没排进去，你只是没拿到那 1.0 分而已。不要因为物理无解而让梯度崩溃。
                reward = self.omega_4  # 👈 核心改变！你也可以写 -0.1，但绝对不能是 -2.0 这种大数字。
                
            self.steps += 1
            
            all_processed = all(state['status'] != 1 for state in self.flow_states.values())
            terminated = all_processed or self.steps >= self.num_flows
            
            info = {
                'flow_idx': flow_idx,
                'is_allocated': is_success,
                'bottleneck_util': util_score if is_success else 1.0,
                'hop_count': hop_count if is_success else 0  
            }
            
            # ==========================================
            # 🏆 架构师的终极大奖 (Global Jackpot)
            # ==========================================
            if terminated:
                success_count = sum(1 for state in self.flow_states.values() if state['status'] == 0)
                p_success = success_count / self.num_flows
                info['group_success_rate'] = p_success
                

                if p_success == 1.0:
                    # 如果 100% 调度成功，大奖
                    jackpot_bonus = self.omega_0 
                    reward += jackpot_bonus
                    
                    # (可选) 在终端里稍微撒个花，方便你观察它有没有“开窍”
                    print(f"\n🎉 [环境撒花] 达成 100% 完美调度！发放全局通关大奖: +{jackpot_bonus}")
                
            return self._get_observation(), reward, terminated, False, info
```

## File: `init.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2\init.py`

```python
"""Phase 2: Transformer-PPO 联合路由调度。"""
```

## File: `train.py`
> 路径: `D:\PY_Project\tsnkit\tsnkit\algorithms\sca_drl\phase2\train.py`

```python
from logging import config
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import time
import glob
import random
import numpy as np
import torch
from tsnkit import core as utils_tsnkit 
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from environment import TSNEnv
from agent import PPOAgent
from sca_drl.common.utils import (
    set_seed, load_config, resolve_path, get_device, ensure_dir
)

def load_phase1_embeddings(config):
    data_cfg = config.get("data", {})
    emb_path = data_cfg.get("phase1_embeddings_pt")
    if emb_path:
        emb_full_path = resolve_path(emb_path)
        if os.path.isfile(emb_full_path):
            print(f"Loading Phase 1 embeddings from: {emb_full_path}")
            return torch.load(emb_full_path, map_location="cpu")
        else:
            print(f"Warning: Phase 1 embedding file not found: {emb_full_path}")
    return None

# ... (保留最顶部的 imports, set_seed 等) ...

def train(config):
    print("="*60)
    print("🚀 SCA-DRL Phase 2: 终极泛化训练引擎启动 (带验证集闭环) 🚀")
    print("="*60)
    
    set_seed(config.get("seed", 42))   
    
    data_cfg = config.get("data", {})
    task_files_path = resolve_path(data_cfg["task_dir"])
    topo_path = resolve_path(data_cfg["topo_file"])    
    
    # 🌟 1. 严格隔离 Train 和 Val 数据集！(拒绝数据泄露)
    train_files = glob.glob(os.path.join(task_files_path, "[0-9]*_task.csv"))
    val_files = glob.glob(os.path.join(task_files_path, "val_*_task.csv"))
    
    if not train_files:
        print("❌ 找不到训练数据！")
        return   
    
    print(f"[1/4] 成功发现拓扑与考卷: 训练集 {len(train_files)} 份，验证集 {len(val_files)} 份！")
    topo = utils_tsnkit.load_network(topo_path)     
      
    print("[2/4] 初始化 TSNEnv 与 Transformer PPOAgent...")
    
    # 环境初始化
    initial_task_file = train_files[0]
    initial_task = utils_tsnkit.load_stream(initial_task_file)  
    env_config = {
        'environment': config.get("environment", {}),
        'task': initial_task,
        'task_file': initial_task_file, # 传入路径
        'topo': topo         
    }
    
    # 🌟 剔除了老的 phase1_emb 传参，环境现在内部动态读取
    env = TSNEnv(env_config)
    print(f"✅ 环境初始化成功！流数量: {env.num_flows}, 特征维度: {env.d_feature}")
    
    agent = PPOAgent(env=env, config=config)
    initial_lr = config.get('agent', {}).get('ppo', {}).get('learning_rate', 3e-4)
   
    # MLOps 配置
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')    
    run_dir = resolve_path(f"./phase2/runs/phase2_ppo_{current_time}")
    model_dir = resolve_path(f"./phase2/models/phase2_{current_time}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # === 恢复你的硬盘级双轨日志系统 ===
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush() 

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(os.path.join(run_dir, "train_log.txt"))
    sys.stderr = sys.stdout  
    print(f"📁 本次实验日志已挂载: {run_dir}/train_log.txt")
    
    writer = SummaryWriter(log_dir=run_dir)

    train_cfg = config.get("training", {})
    total_iterations = train_cfg.get("total_iterations", 200)
    episodes_per_iter = train_cfg.get("episodes_per_iter", 8)
    
    print(f"[3/4] 训练规划: 共 {total_iterations} 轮, 每轮收集 {episodes_per_iter} 份考卷")
    print("[4/4] ⚔️ 面对风暴吧！Transformer！\n")

    best_val_success = -1.0 # 用于保存最佳模型

    for iteration in range(1, total_iterations + 1):
        # --- 学习率衰减 ---
        frac = 1.0 - (iteration - 1.0) / total_iterations
        lr_now = frac * initial_lr
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = lr_now        
        
        rollouts = { 'flow_tokens': [], 'global_snapshot': [], 'action_masks': [], 'actions': [], 
                     'logprobs': [], 'rewards': [], 'values': [], 'dones': [] }
        
        iter_rewards, iter_success_rates, iter_avg_hops = [], [], []

        # ==========================================
        # 🏋️‍♂️ 阶段 A：训练模式 (Train Loop)
        # ==========================================
        agent.network.train()
        for ep in range(episodes_per_iter):
            # 严格从 train_files 里抽题！
            random_task_file = random.choice(train_files)
            new_task = utils_tsnkit.load_stream(random_task_file)   
            env.load_new_task(new_task, random_task_file)
            
            obs, _ = env.reset()
            ep_reward, ep_hops = 0.0, []
            
            for step in range(env.num_flows):
                rollouts['flow_tokens'].append(obs['flow_tokens'])
                rollouts['global_snapshot'].append(obs['global_snapshot'])
                rollouts['action_masks'].append(obs['action_mask'])
                
                # 采样动作 (探索)
                action, logprob, entropy, value = agent.get_action_and_value(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                rollouts['actions'].append(action)
                rollouts['logprobs'].append(logprob.item())
                rollouts['rewards'].append(reward)
                rollouts['values'].append(value.item())
                rollouts['dones'].append(terminated) 
                
                ep_reward += reward
                obs = next_obs
                
                if info.get('is_allocated', False) and 'hop_count' in info:
                    ep_hops.append(info['hop_count'])
                
                if terminated:
                    if 'group_success_rate' in info:
                        iter_success_rates.append(info['group_success_rate'])
                    iter_avg_hops.append(sum(ep_hops)/len(ep_hops) if ep_hops else 0.0)
                    break
            
            iter_rewards.append(ep_reward)

        # --- PPO 参数更新 ---
        pg_loss, v_loss, ent = agent.update(rollouts)

        avg_reward = np.mean(iter_rewards)
        avg_success = np.mean(iter_success_rates) * 100 if iter_success_rates else 0.0
        avg_hop = np.mean(iter_avg_hops) if iter_avg_hops else 0.0
        
        # print(f"Iter {iteration:03d} | Train 成功率: {avg_success:5.1f}% | Reward: {avg_reward:7.2f} | P_Loss: {pg_loss:6.3f}")

        # 终端打印 (你之前已经加回来的)
        print(f"Iter {iteration:03d} | 总流数：{env.num_flows} | Train 成功率: {avg_success:5.1f}% | 均跳数: {avg_hop:4.2f} | Reward: {avg_reward:7.2f} | P_Loss: {pg_loss:6.3f} | V_Loss: {v_loss:6.3f} | Ent: {ent:5.3f}")

        writer.add_scalar("Train/1_Success_Rate", avg_success, iteration)
        writer.add_scalar("Train/2_Avg_Reward", avg_reward, iteration)
        writer.add_scalar("Loss/1_Policy_Loss", pg_loss, iteration)
        writer.add_scalar("Loss/2_Value_Loss", v_loss, iteration)
        # 🌟 加上这极其重要的一行！恢复 Entropy 监控！
        writer.add_scalar("Loss/3_Entropy", ent, iteration)

        # ==========================================
        # 🧪 阶段 B：验证模式 (Validation Loop) - 每 10 轮考一次
        # ==========================================
        if iteration % 10 == 0 and val_files:
            agent.network.eval() # 关闭 Dropout 等
            val_success_rates = []
            
            # 去做 10 张固定的验证卷 (不求导，纯测试)
            with torch.no_grad():
                # 为了速度，我们随机挑 10 张验证卷来考
                eval_files = random.sample(val_files, min(10, len(val_files)))
                for v_file in eval_files:
                    v_task = utils_tsnkit.load_stream(v_file)
                    env.load_new_task(v_task, v_file)
                    obs, _ = env.reset()
                    
                    for step in range(env.num_flows):
                        # 测试时直接取 argmax，不引入随机性
                        flow_tokens = torch.FloatTensor(obs['flow_tokens']).unsqueeze(0).to(agent.device)
                        g_global = torch.FloatTensor(obs['global_snapshot']).unsqueeze(0).to(agent.device)
                        mask = torch.BoolTensor(obs['action_mask']).unsqueeze(0).to(agent.device)
                        
                        logits, _ = agent.network(flow_tokens, g_global)
                        logits = logits.masked_fill(~mask, -1e8)
                        action = torch.argmax(logits, dim=-1).item() # 🌟 绝对确定性的选择
                        
                        obs, _, terminated, _, info = env.step(action)
                        if terminated:
                            val_success_rates.append(info.get('group_success_rate', 0.0))
                            break
                            
            avg_val_success = np.mean(val_success_rates) * 100
            print(f"   => 🏆 [期中考试] 验证集成功率: {avg_val_success:5.1f}%")
            writer.add_scalar("Eval/1_Val_Success_Rate", avg_val_success, iteration)
            
            # 🌟 保存“泛化能力最强”的最佳模型
            if avg_val_success > best_val_success:
                best_val_success = avg_val_success
                best_path = os.path.join(model_dir, "phase2_ppo_best.pth")
                torch.save(agent.network.state_dict(), best_path)
                print(f"   => 🌟 [突破纪录] 最优模型已更新并保存!")

    writer.close()
    print("\n🎉 训练完美收官！")

if __name__ == "__main__":    
    import argparse
    from sca_drl.common.utils import load_config
    
    parser = argparse.ArgumentParser(description="Phase 2 PPO Training")
    parser.add_argument("--config", default="configs/phase2.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
```
