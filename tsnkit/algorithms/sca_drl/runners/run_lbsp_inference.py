import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

# --- 路径设置 ---
curr_dir = os.path.dirname(os.path.abspath(__file__)) # runners/
algorithm_dir = os.path.dirname(os.path.dirname(curr_dir)) # algorithm/ (tsnkit/algorithms)
tsnkit_dir = os.path.dirname(algorithm_dir) # tsnkit根目录

sys.path.append(algorithm_dir)

# 引用核心模块
# 注意：请确保这里 import 的文件名与你本地实际保留的文件名一致
from sca_drl.phase1_lbsp_training import (
    TSNPhase1Dataset, 
    GNNPartitionModel, 
    calculate_harmonic_prior
)

# 引入 tsnkit core 用于加载原始数据进行绘图
from tsnkit import core as utils

def compute_affinity_matrix(embeddings, streams, alpha=0.6):
    """
    实现白皮书公式: W_ij = alpha * s_gnn + (1 - alpha) * s_prior
    """
    num_streams = len(streams)
    W = np.zeros((num_streams, num_streams))
    
    # 1. 计算 s_gnn (Embedding 相似度矩阵)
    # 归一化 embedding 以便计算余弦相似度
    emb_norm = F.normalize(embeddings, p=2, dim=1)
    # s_gnn = emb * emb.T (Result: N x N matrix, range [-1, 1])
    s_gnn = torch.mm(emb_norm, emb_norm.t()).cpu().numpy()
    
    # 2. 混合计算
    print("Computing Hybrid Metric W...")
    for i in range(num_streams):
        for j in range(i + 1, num_streams):
            # s_prior: 谐波亲和度
            prior = calculate_harmonic_prior(streams[i].period, streams[j].period)
            
            # 混合度量
            # 注意: s_gnn range is [-1, 1], prior is [0, 1]
            # 为了谱聚类，我们需要非负的亲和度 (Affinity)
            # 简单处理：将 s_gnn 映射到 [0, 1] -> (sim + 1) / 2
            sim_score = (s_gnn[i, j] + 1) / 2
            
            w_val = alpha * sim_score + (1 - alpha) * prior
            
            W[i, j] = w_val
            W[j, i] = w_val # 对称矩阵
            
        W[i, i] = 1.0 # 对角线自相似度为 1
        
    return W

def visualize_clusters(W, labels, save_path):
    """可视化亲和度矩阵和聚类结果"""
    plt.figure(figsize=(10, 8))
    # 根据聚类标签对矩阵进行排序，以便观察块状结构
    sort_idx = np.argsort(labels)
    W_sorted = W[sort_idx][:, sort_idx]
    
    plt.imshow(W_sorted, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Hybrid Affinity W')
    plt.title(f'Clustered Affinity Matrix (Sorted by Group)')
    plt.savefig(save_path)
    print(f"Cluster Matrix visualization saved to {save_path}")
    plt.close()

def visualize_network_topology(G, save_path):
    """
    绘制网络拓扑图。
    优化布局以减少边交叉：优先尝试 planar_layout，其次使用 kamada_kawai_layout。
    """
    plt.figure(figsize=(12, 10))
    
    # --- 布局优化逻辑 ---
    layout_used = "spring"
    try:
        # 1. 优先尝试平面布局 (Planar Layout)
        # 如果图是平面的（没有边必须交叉才能画出来），这个布局保证无交叉
        print("Attempting planar layout...")
        pos = nx.planar_layout(G)
        layout_used = "planar"
    except nx.NetworkXException:
        # 2. 如果图不是平面的，退回到 Kamada-Kawai 布局
        # 这种布局算法基于物理能量最小化，通常比 spring_layout 更能减少交叉且更对称
        print("Graph is not planar. Falling back to Kamada-Kawai layout...")
        try:
            pos = nx.kamada_kawai_layout(G)
            layout_used = "kamada_kawai"
        except:
             # 3. 最后保底使用 Spring Layout (带更多迭代次数)
            print("Falling back to Spring layout...")
            pos = nx.spring_layout(G, seed=42, k=0.5, iterations=100)
            layout_used = "spring"

    print(f"Using {layout_used} layout.")

    # 区分节点类型
    es_nodes = [n for n, d in G.degree() if d == 1]
    sw_nodes = [n for n, d in G.degree() if d > 1]
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, nodelist=sw_nodes, node_color='orange', node_size=500, label='Switch')
    nx.draw_networkx_nodes(G, pos, nodelist=es_nodes, node_color='lightblue', node_size=300, label='End System')
    
    # 绘制边
    # connectionstyle="arc3,rad=0.1" 可以画出弯曲的线，但 nx.draw_networkx_edges 原生支持有限
    # 这里的简单优化主要是靠 pos 布局
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, edge_color='gray')
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # 图例和标题
    plt.title(f"Network Topology Visualization ({layout_used} layout)")
    plt.legend(scatterpoints=1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Topology visualization saved to {save_path}")
    plt.close()


def main():
    # --- 1. 配置 ---
    config = {
        'k_paths': 3,
        'model': {
            'input_dim': 3, 
            'hidden_dim': 64,
            'output_dim': 16
        },
        'inference': {
            'model_path': '../results/checkpoints/gnn_phase1_new.pth',
            'alpha': 0.6,       # 混合因子
            'n_clusters': 4,    # 期望将流分成的组数 (可根据流数量动态调整)
            'vis_dir': '../results/figures/' # 结果保存目录
        }
    }
    
    # 确保输出目录存在
    os.makedirs(config['inference']['vis_dir'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 2. 加载数据 ---
    # 使用 tsnkit 测试数据
    tsn_data_dir = os.path.join(tsnkit_dir, 'test/data')
    task_file = os.path.join(tsn_data_dir, '1_task.csv')
    topo_file = os.path.join(tsn_data_dir, '1_topo.csv')
    
    print(f"Loading data from {task_file}...")
    dataset = TSNPhase1Dataset(task_file, topo_file, config)
    data = dataset[0].to(device) # 获取整张图数据
    
    # 获取原始流对象以便访问 period 属性计算 s_prior
    tsn_tasks = utils.load_stream(task_file)
    streams = tsn_tasks.streams
    
    # --- 新增: 加载并绘制网络拓扑 ---
    print("Visualizing Network Topology...")
    tsn_net = utils.load_network(topo_file)
    G_nx = tsn_net.net_nx
    topo_vis_path = os.path.join(config['inference']['vis_dir'], 'network_topology.png')
    visualize_network_topology(G_nx, topo_vis_path)

    # --- 3. 加载模型 ---
    print(f"Loading model from {config['inference']['model_path']}...")
    model = GNNPartitionModel(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim']
    ).to(device)
    
    if os.path.exists(config['inference']['model_path']):
        model.load_state_dict(torch.load(config['inference']['model_path'], map_location=device))
    else:
        print("Error: Checkpoint not found! using random weights for testing.")

    model.eval()

    # --- 4. 推理 (Embedding) ---
    with torch.no_grad():
        print("Running GNN forward pass...")
        embeddings = model(data) # [Num_Streams, Output_Dim]
        
    print(f"Embeddings shape: {embeddings.shape}")

    # --- 5. 划分 (Partitioning / Clustering) ---
    alpha = config['inference']['alpha']
    W = compute_affinity_matrix(embeddings, streams, alpha)
    
    print(f"Executing Spectral Clustering (k={config['inference']['n_clusters']})...")
    # 使用 Precomputed affinity matrix 进行谱聚类
    sc = SpectralClustering(
        n_clusters=config['inference']['n_clusters'], 
        affinity='precomputed', 
        random_state=42
    )
    labels = sc.fit_predict(W)
    
    # --- 6. 输出结果 ---
    print("\n--- Partitioning Result ---")
    for group_id in range(config['inference']['n_clusters']):
        # 找出属于该组的流索引
        group_indices = np.where(labels == group_id)[0]
        # 映射回原始 Stream ID
        group_stream_ids = [data.stream_ids[i] for i in group_indices]
        print(f"Group {group_id}: count={len(group_indices)}, flows={group_stream_ids}")
        
    # 可视化聚类矩阵
    cluster_vis_path = os.path.join(config['inference']['vis_dir'], 'partition_matrix.png')
    visualize_clusters(W, labels, cluster_vis_path)

if __name__ == "__main__":
    main()