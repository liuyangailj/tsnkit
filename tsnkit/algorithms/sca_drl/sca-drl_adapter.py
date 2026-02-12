import networkx as nx
from tsnkit import core as utils  # 确保 tsnkit 在 python path 中

def load_tsnkit_data(task_path, topo_path):
    """
    加载 tsnkit 的 csv 文件，并转换为 sca-drl 训练所需的格式。
    
    Args:
        task_path: 流量文件路径 (.csv)
        topo_path: 拓扑文件路径 (.csv)
        
    Returns:
        flows: list of dict, 流量列表
        G: networkx.DiGraph, 网络拓扑图
    """
    # 1. 使用 tsnkit 加载原始数据
    try:
        tsn_network = utils.load_network(topo_path)
        tsn_tasks = utils.load_stream(task_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load tsnkit data: {e}")

    # 2. 转换拓扑 (Topology) -> NetworkX
    # tsnkit 已经提供了 networkx 图: tsn_network.net_nx
    # 我们可能需要补充一些属性，比如链路带宽
    G = tsn_network.net_nx
    
    # 确保图是 DiGraph (有向图)
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    # 可选：遍历边，添加属性 (如果你的 GNN 需要 bandwidth 等特征)
    for u, v, data in G.edges(data=True):
        # tsnkit 的 link 对象
        link_obj = tsn_network.get_link((u, v))
        # 这里的 rate 单位通常是 Gbps，你可以根据需要转换
        data['bandwidth'] = link_obj.rate 
        data['weight'] = 1.0 # 默认权重

    # 3. 转换流量 (Flows) -> List of Dicts
    flows = []
    for stream in tsn_tasks.streams:
        # 注意：tsnkit 的 stream 对象属性可能经过了归一化 (除以 T_SLOT)
        # 如果你的 GNN 模型期望物理单位 (ns, bytes)，需要乘回来
        # 或者直接使用归一化后的值训练 (推荐保持一致)
        
        # tsnkit 加载后：
        # stream.period 是 slot 数
        # stream.size 是 (bytes/T_SLOT) 的归一化值
        
        # 这里为了保险，我们还原回 dataset_spec.py 生成时的原始物理概念
        # 但如果你的模型本来就是处理归一化数值的，可以直接用 stream.period
        
        flow_data = {
            'flow_id': stream._id,
            'src': stream.src,
            'dst': stream.dst[0] if isinstance(stream.dst, list) else stream.dst,
            'size': stream.size,         # 这是一个 int (Bytes / T_SLOT)
            'period': stream.period,     # int (ns / T_SLOT)
            'deadline': stream.deadline, # int (ns / T_SLOT)
            'jitter': stream.jitter
        }
        flows.append(flow_data)

    return flows, G