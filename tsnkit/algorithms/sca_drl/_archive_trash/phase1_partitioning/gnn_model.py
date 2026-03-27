import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEEncoder, self).__init__()
        # 第一层 SAGE: 聚合邻居信息
        # project=True 对应论文公式里的 W * CONCAT(...)
        self.conv1 = SAGEConv(in_channels, hidden_channels, project=True)
        # 第二层 SAGE
        self.conv2 = SAGEConv(hidden_channels, out_channels, project=True)

    def forward(self, x, edge_index):
        # x: [num_nodes, num_features]
        # edge_index: [2, num_edges]
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        # 最后一层通常不加 ReLU，直接输出 Embedding
        
        return x # [num_nodes, out_channels]

class ConflictPredictor(nn.Module):
    """
    MLP Head: 输入两个流的 Embedding，输出它们冲突的概率
    对应公式: s_gnn = Sigmoid(MLP(h_i || h_j))
    """
    def __init__(self, embedding_dim):
        super(ConflictPredictor, self).__init__()
        # 输入维度是 embedding_dim * 2 (因为是拼接)
        self.lin1 = nn.Linear(embedding_dim * 2, 64)
        self.lin2 = nn.Linear(64, 1)

    def forward(self, z_i, z_j):
        # z_i, z_j: [batch_size, embedding_dim]
        
        # 1. 拼接（concat）
        z = torch.cat([z_i, z_j], dim=-1)
        
        # 2. MLP
        x = F.relu(self.lin1(z))
        x = self.lin2(x)
        
        # 3. Sigmoid (输出 0-1 概率)
        return torch.sigmoid(x).squeeze()

class CorrelationModel(nn.Module):
    """
    将 Encoder 和 Predictor 封装在一起
    """
    def __init__(self, in_dim, hidden_dim, embed_dim):
        super(CorrelationModel, self).__init__()
        self.encoder = GraphSAGEEncoder(in_dim, hidden_dim, embed_dim)
        self.predictor = ConflictPredictor(embed_dim)

    def forward(self, x, edge_index, u, v):
        """
        x, edge_index: 用于生成所有节点的 Embedding
        u, v: 想要查询冲突概率的流索引列表 (例如 u=[0,1], v=[5,6])
        """
        # 1. 生成所有流的 Embedding
        embeddings = self.encoder(x, edge_index)
        
        # 2. 取出指定流对的 Embedding
        z_u = embeddings[u]
        z_v = embeddings[v]
        
        # 3. 预测
        score = self.predictor(z_u, z_v)
        return score, embeddings