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