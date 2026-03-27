import torch
import numpy as np
from torch_geometric.data import Data
import sys
import os

# Add parent directory to path to import common module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

class FlowGraphDataset:
    def __init__(self, flows, topology=None, max_period=None):
        """
        Args:
            flows: list of flow dicts (from FlowGenerator or TSNKit reader)
                    Expected keys: id, src, dst, size, period, deadline, jitter, k_paths
            topology: networkx graph (optional, for backward compatibility)
            max_period: used for normalization
        """
        self.flows = flows
        self.num_flows = len(flows)
        # 如果没传 max_period，就从数据里找最大的
        self.max_period = max_period if max_period else max(f['period'] for f in flows)
        self.max_size = 1500  # Ethernet MTU
        self.max_deadline = max_period if max_period else max(f['deadline'] for f in flows)

    def _normalize(self, val, max_val):
        return val / max_val if max_val > 0 else 0

    def get_node_features(self):
        """
        构建节点特征矩阵 X [num_flows, num_features]
        特征包含: [Norm_Period, Norm_Size, Norm_Deadline, Norm_Jitter]
        """
        features = []
        for f in self.flows:
            # 特征：周期、大小、Deadline、Jitter
            # 未来可以把 KSP 的链路 ID embedding 加进来，但先跑通基础的
            jitter = f.get('jitter', 0)  # Jitter might not always be present
            x = [
                self._normalize(f['period'], self.max_period),
                self._normalize(f['size'], self.max_size),
                self._normalize(f['deadline'], self.max_deadline),
                self._normalize(jitter, self.max_period)  # Jitter normalized by period
            ]
            features.append(x)
        return torch.tensor(features, dtype=torch.float)

    # 新代码建议 使用所有 K 条路径来判断冲突
    def get_edge_index(self):
        src_nodes = []
        dst_nodes = []
        
        flow_links_union = []
        for f in self.flows:
            # 无论 k=1 还是 k=5，把所有路径涉及的链路都加到一个集合里
            # 形成该流的 "潜在势力范围"
            all_links = set()
            for path in f['k_paths']: # 遍历所有 K 条路径
                # 将路径转为链路元组
                links_in_path = set(zip(path[:-1], path[1:]))
                all_links.update(links_in_path)
            flow_links_union.append(all_links)

        # 两两比较 (复杂度还是 O(N^2)，只是集合操作稍微变大了一点点，完全可接受)
        for i in range(self.num_flows):
            for j in range(i + 1, self.num_flows):
                if not flow_links_union[i].isdisjoint(flow_links_union[j]):
                    src_nodes.extend([i, j])
                    dst_nodes.extend([j, i])

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        return edge_index
    
    def process(self):
        """返回 PyG 的 Data 对象"""
        x = self.get_node_features()
        edge_index = self.get_edge_index()
        
        # Data 对象是 PyG 的核心
        data = Data(x=x, edge_index=edge_index)
        return data