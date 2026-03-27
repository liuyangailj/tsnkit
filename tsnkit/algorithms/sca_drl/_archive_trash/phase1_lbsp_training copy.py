import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
import numpy as np
import networkx as nx
import math
from itertools import islice

# --- 引入 TSNKit 核心 ---
# 这是整个数据集构建的基础，保证读取逻辑与 benchmark 完全一致
from tsnkit import core as utils

# ==========================================
# 1. 工具函数 (KSP & Harmonic Prior)
# ==========================================

def k_shortest_paths(G, source, target, k, weight=None):
    """
    基于 NetworkX 计算 K 条最短路径。
    注意：TSNKit 的 _network.py 中 get_all_path 使用的是 all_simple_paths。
    为了限制 Phase 1 的搜索空间并提供多样性选择，这里使用 K-Shortest Paths。
    """
    try:
        # islice 用于从生成器中提取前 k 个结果，避免计算所有路径
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
    except nx.NetworkXNoPath:
        return []

def calculate_harmonic_prior(period_i, period_j):
    """
    计算两个流周期的谐波亲和度 (Harmonic Prior)
    公式: s_prior = GCD(Ti, Tj) / LCM(Ti, Tj)
    这个值反映了两个周期性流在时域上的重叠频率。
    """
    if period_i == 0 or period_j == 0: return 0.0
    gcd_val = math.gcd(period_i, period_j)
    lcm_val = (period_i * period_j) // gcd_val
    return float(gcd_val) / float(lcm_val)

# ==========================================
# 2. Dataset: 深度对接 TSNKit 的数据处理
# ==========================================

class TSNPhase1Dataset(Dataset):
    def __init__(self, task_path, topo_path, config=None):
        """
        Args:
            task_path: tsnkit 格式的流量 csv 路径
            topo_path: tsnkit 格式的拓扑 csv 路径
            config: 配置字典，包含 'k_paths' 等超参数
        """
        # 不调用 super().__init__() 以避免 PyG Dataset 的自动处理
        # 手动初始化 Dataset 的必要属性
        self.task_path = task_path
        self.topo_path = topo_path
        self.config = config or {}
        self.k = self.config.get('k_paths', 3)  # 默认 K=3

        # PyG Dataset 必要的属性
        self.transform = None
        self.pre_transform = None
        self.pre_filter = None

        # 结果缓存
        self.data_list = []

        # 执行处理
        self.process()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def process(self):
        print(f"[Phase1] Loading TSNKit data from {self.task_path}...")
        
        # --- 1. 使用 TSNKit 原生加载器 (核心一致性保证) ---
        try:
            # load_network 返回 tsnkit.core._network.Network 对象
            tsn_net = utils.load_network(self.topo_path)
            # load_stream 返回 tsnkit.core._stream.StreamSet 对象
            tsn_tasks = utils.load_stream(self.task_path)
        except Exception as e:
            print(f"[Error] Failed to load files using tsnkit: {e}")
            return

        # 获取 NetworkX 图对象 (tsnkit 内部维护了 .net_nx 属性)
        G_nx = tsn_net.net_nx
        
        # 获取所有流对象列表
        streams = tsn_tasks.streams
        num_streams = len(streams)
        
        print(f"[Phase1] Processing {num_streams} streams for Conflict Graph...")

        # --- 2. 构建节点特征 (Node Features) ---
        # 这里的“节点”指的是冲突图中的节点，即每一条“流”
        node_features = []
        
        
        # 存储每条流的 K 条路径列表 (List of Sets of Links)，用于检测冲突
        # stream_k_paths[i] = [ {link1, link2}, {link3, link4}, ... ] (共K个)
        stream_paths_links = [] 

        for s in streams:
            # 1.节点特征
            # 注意：依据 tsnkit/_stream.py，这些属性已经被归一化（除以 T_SLOT）
            # s.size: 占用的数据块数量 (chunk units)
            # s.period: 周期 (slots)
            # s.deadline: 截止时间 (slots)
            
            # 我们直接使用这些归一化后的值，因为它们代表了系统视角下的真实资源需求
            # 对 period 进行 log 处理以压缩数值范围，便于神经网络学习
            feat = [
                float(s.size), 
                math.log(float(s.period) + 1), 
                math.log(float(s.deadline) + 1)
            ]
            node_features.append(feat)
            
            # 2.路径计算
            # tsnkit 支持多播，dst 是一个列表。Phase 1 简化处理，暂时只取第一个目的节点作为单播路径计算
            # 这种处理方式与 ls.py 中的处理逻辑兼容
            target_node = s.dst[0] if isinstance(s.dst, list) else s.dst
            
            # 利用 tsnkit 加载的图计算 K 条最短路径
            paths = k_shortest_paths(G_nx, s.src, target_node, self.k)
            
            # 将路径转换为“链路集合”以便快速进行集合求交运算
            # path: [n1, n2, n3] -> links: {(n1,n2), (n2,n3)}
            s_links_set = set()
            for p in paths:
                # 生成路径上的所有边 (u, v)
                edges = list(zip(p[:-1], p[1:]))
                s_links_set.update(edges)
            stream_paths_links.append(s_links_set)

        # 转换为 Tensor
        x = torch.tensor(node_features, dtype=torch.float)

        # --- 3. 构建边和边属性 (Edges & Edge Attributes) ---
        # 核心逻辑：构建“流冲突图”
        # 节点 = 流
        # 边 = 两条流在空间上（K条路径中）存在潜在的链路争用
        
        edge_index = []
        edge_attr = []

        # 双重循环检测每对流 (Pairwise Check) O(N^2)
        for i in range(num_streams):
            for j in range(i + 1, num_streams):
                # A. 空间相关性 (Spatial Correlation)
                # 检查两个流的候选路径集合是否有交集
                if not stream_paths_links[i].isdisjoint(stream_paths_links[j]):
                    # 如果有重叠链路，说明存在潜在冲突，建立边
                    edge_index.append([i, j])
                    edge_index.append([j, i]) # 无向图，双向添加
                    
                    # B. 时间相关性 (Temporal Prior / Harmonic)
                    # 计算谐波先验作为边权重
                    prior = calculate_harmonic_prior(streams[i].period, streams[j].period)
                    edge_attr.append([prior])
                    edge_attr.append([prior])

        # 转换为 Tensor 格式
        if len(edge_index) > 0:
            # 1. 列表转Tensor: 此时形状[E, 2]
            # 2. .t(): 转置为[2, E](PyG标准格式： source_nodes在第一行， targets_nodes在第二行)
            # 3. .contigous(): 确保内存连续，加速后续 GAT 运算。
            # 4. dtype=torch.long: 索引必须是整数类型
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # 边属性保持 [E, 1] 形状，浮点数类型
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            # 处理无边的情况（例如流之间完全无冲突）
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        # --- 4. 封装 PyG Data 对象 ---
        # Data 对象是一个容器，它不改变内部 Tensor 的形状
        # x: [N, 3]
        # edge_index: [2, E]
        # edge_attr: [E, 1]
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = num_streams
        
        # 保存辅助信息，方便后续推理时映射回 stream ID
        data.stream_ids = [s._id for s in streams]
        
        # 将 Data 放入列表 (PyG Dataset 规范)
        self.data_list = [data]
        
        print(f"[Phase1] Graph Created: Nodes(Streams)={x.shape[0]}, Conflict Edges={edge_index.shape[1]}")

# ==========================================
# 3. GNN Model: 冲突图嵌入模型
# ==========================================

class GNNPartitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=16):
        super(GNNPartitionModel, self).__init__()
        # GATConv (Graph Attention Network)
        # edge_dim=1 允许我们将 harmonic prior (edge_attr) 注入到注意力机制中
        self.conv1 = GATConv(input_dim, hidden_dim, edge_dim=1) 
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=1)
        
        # 输出层：生成每个流的 Embedding
        self.lin = nn.Linear(hidden_dim, output_dim)
        
        # 输入归一化，加速收敛
        self.input_bn = nn.BatchNorm1d(input_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.input_bn(x)
        
        # Layer 1
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        
        # Final Embedding
        embedding = self.lin(x) 
        
        return embedding

# ==========================================
# 4. Trainer Function
# ==========================================

def train_phase1(model, dataset, config, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    model.train()
    
    print("--- Start Training Phase 1 ---")
    for epoch in range(config.get('epochs', 100)):
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            embeddings = model(data)
            
            # --- Contrastive Loss 实现 ---
            # 目标：根据冲突关系和谐波先验优化 Embedding 分布
            # 白皮书指导: W_ij = alpha * s_gnn + (1-alpha) * s_prior
            # s_gnn 是 embedding 的相似度
            
            if data.edge_index.numel() > 0:
                src, dst = data.edge_index
                
                # 1. 计算 Embedding 相似度 (s_gnn)
                # 使用余弦相似度，范围 [-1, 1]
                s_gnn = F.cosine_similarity(embeddings[src], embeddings[dst])
                
                # 2. 获取对应的 Harmonic Prior (s_prior)
                # edge_attr 范围 [0, 1]
                s_prior = data.edge_attr.squeeze()
                
                # 3. 定义训练目标 (Proxy Task)
                # 我们希望 s_gnn 能够捕捉到 s_prior 反映的物理规律
                # 即：若 s_prior 高（谐波相关强），则 embedding 应更相似（s_gnn 接近 1）
                # 这样后续聚类时它们会被分到一组，从而便于对齐
                
                # 简单的 MSE Loss 逼近
                loss = F.mse_loss(s_gnn, s_prior)
                
                # 进阶 Loss (可选): 结合对比损失，推开 s_prior 低的节点
                # neg_mask = s_prior < 0.1
                # loss += 0.5 * torch.mean(torch.clamp(s_gnn[neg_mask] - (-1), min=0)) 
                
            else:
                loss = torch.tensor(0.0, requires_grad=True).to(device)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    # 保存模型
    save_path = config.get('save_path', './gnn_phase1.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")