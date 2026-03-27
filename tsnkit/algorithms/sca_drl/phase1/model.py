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