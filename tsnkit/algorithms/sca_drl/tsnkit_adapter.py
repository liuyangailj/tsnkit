# sca_drl/adapter.py

import sys
import os
import networkx as nx
import numpy as np

# 假设你已经安装了 tsnkit 或者将其路径加入到了 sys.path
# sys.path.append("/path/to/tsnkit") 
from tsnkit import core as utils

def load_data_from_tsnkit(task_csv_path, topo_csv_path):
    """
    使用 tsnkit 的原生加载器读取数据，并转换为 sca-drl 需要的格式
    """
    
    # 1. 直接调用 tsnkit API 加载 (黑盒复用)
    try:
        tsn_network = utils.load_network(topo_csv_path)
        tsn_tasks = utils.load_stream(task_csv_path)
    except Exception as e:
        print(f"Error loading tsnkit data: {e}")
        return None, None

    # 2. 转换拓扑 (Topology)
    # sca-drl 可能需要 networkx 图或者邻接矩阵
    # tsnkit 已经提供了 networkx 图，直接拿来用！
    nx_graph = tsn_network.net_nx 
    
    # 如果 sca-drl 需要特定的节点/边特征，可以在这里提取并添加
    # 例如，把链路带宽 (rate) 作为边属性
    for u, v, data in nx_graph.edges(data=True):
        # tsnkit 的 link 对象可以通过 (u, v) 索引找到
        link = tsn_network.get_link((u, v))
        data['bandwidth'] = link.rate # 假设 sca-drl 需要 'bandwidth'
        data['latency'] = link.t_prop # 假设需要传播时延

    # 3. 转换流 (Streams/Flows)
    # sca-drl 通常需要一个字典或列表来表示流
    flows = []
    for stream in tsn_tasks.streams:
        flow_data = {
            'flow_id': stream._id,   # 使用内部 id
            'src': stream.src,
            'dst': stream.dst,       # tsnkit 支持多播，这里暂时只取单播 dst[0] 如果是列表
            'size': stream.size,
            'period': stream.period,
            'deadline': stream.deadline,
            'jitter': stream.jitter
        }
        flows.append(flow_data)

    return nx_graph, flows

# 测试代码
if __name__ == "__main__":
    # 替换为你实际的文件路径
    task_file = "../tsnkit/tsnkit/test/data/1_task.csv"
    topo_file = "../tsnkit/tsnkit/test/data/1_topo.csv"
    
    if os.path.exists(task_file) and os.path.exists(topo_file):
        g, f = load_data_from_tsnkit(task_file, topo_file)
        print(f"Graph Nodes: {len(g.nodes())}, Edges: {len(g.edges())}")
        print(f"Flows Loaded: {len(f)}")
        print("First Flow:", f[0])
    else:
        print("Test files not found.")