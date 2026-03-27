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