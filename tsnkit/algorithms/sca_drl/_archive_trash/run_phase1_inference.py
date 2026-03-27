import sys
import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import yaml
import numpy as np

# 1. 项目路径设置
    # 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (即 runners/ 的上一级)
project_root = os.path.dirname(current_dir)
    # 将根目录加入 python 查找路径
sys.path.append(project_root)

# 2. 导入模块
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from common.topology_gen import TopologyGenerator
from common.flow_gen import FlowGenerator
from phase1_partitioning.dataset import FlowGraphDataset
from phase1_partitioning.gnn_model import CorrelationModel
    # 导入配置加载工具
from common.config_utils import load_config

def compute_prior_score(flow_i, flow_j):
    """
    计算基于周期的先验分数 (Nie et al.)
    Score = 1 - GCD(Ti, Tj) / LCM(Ti, Tj)
    """
    Ti, Tj = flow_i['period'], flow_j['period']
    gcd = np.gcd(Ti, Tj)
    lcm = (Ti * Tj) // gcd
    return (gcd / lcm) # 这里使用谐波对齐度作为分数，值越大表示周期越接近，即为冲突概率越高。

def main():
    print("="*50)
    print("Starting Phase 1: GNN Partitioning Inference")
    print("="*50)
    
    # 3. 加载配置 (Base + Phase 1)
    print("[Step 0] Loading Configuration...")
    config = load_config("configs/base_config.yaml", "configs/phase1_config.yaml")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    # 4. 处理模型路径 (这是你觉得最绕的地方，现在改成了最简单的一行代码)
    # 逻辑：项目根目录 + 配置里的相对路径 = 真实绝对路径
    ckpt_relative_path = config.get('train', {}).get('save_path', "results/checkpoints/gnn_phase1.pth")
    save_path = os.path.join(project_root, ckpt_relative_path)

    print(f"  Looking for checkpoint at: {save_path}")
    
    # 5. 加载模型
    print("--- 2. Loading Trained Model ---")
    print("\n[Step 1] Loading Trained Model...")    
    gnn_hidden = config.get('model', {}).get('gnn_hidden_dim', 64)
    model = CorrelationModel(in_dim=3, hidden_dim=gnn_hidden, embed_dim=32)
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"  [Success] Model loaded successfully.")
    else:
        print(f"  [Error] Checkpoint not found! Please run 'runners/run_phase1_training.py' first.")
        return
    
    # 6.生成测试数据，保持和训练时一样的流数量，或者重新生成一批做测试
    config['traffic']['num_flows'] = 50  #测试用 50 条流    
    
    print("--- 1. Generating Test Data ---")
        # 使用配置中的拓扑参数
    print("\n[Step 2] Generating Test Data...")
    topo_gen = TopologyGenerator(config['topology']['num_nodes'])
    graph = topo_gen.generate_graph()
    
        # 初始化流量
    flow_gen = FlowGenerator(config, graph)
    flows = flow_gen.generate_flows()
    print(f"  Generated {len(flows)} flows on {len(graph.nodes)} nodes.")
    
    # 7. GNN 推理，准备图数据集
    print("\n[Step 2] Preparing Graph Dataset...")
    dataset = FlowGraphDataset(flows, graph)
    data = dataset.process().to(device)
    
    model.eval()    
        # 计算 Embedding
    with torch.no_grad():
        # 只需要 Encoder 部分算出 H
        embeddings = model.encoder(data.x, data.edge_index) # [N, 32]
        
    print(f"Generated Embeddings: {embeddings.shape}")

    # 8. 构建相似度矩阵 W (对应 Algorithm 1 的 Step 3)
        # W = lambda * s_gnn + (1-lambda) * s_prior
    print("--- 3. Constructing Similarity Matrix ---")
    print("\n[Step 3] Clustering...")
    num_flows = len(flows)
    W = np.zeros((num_flows, num_flows))
    lambda_val = 0.6  # 权重超参数，可调整
    
    # 这一步如果是几千条流，可以用矩阵运算优化，现在先用双重循环
    for i in range(num_flows):
        for j in range(i + 1, num_flows):
            # A. GNN Score (Conflict Prob)
            # 注意: 模型预测的是冲突概率，而这里我们需要“相似度/权重”。
            # 如果我们要把冲突的流分在一起，那权重就是冲突概率。
            emb_i = embeddings[i].unsqueeze(0)
            emb_j = embeddings[j].unsqueeze(0)
            with torch.no_grad():
                s_gnn = model.predictor(emb_i, emb_j).item()
            
            # B. Prior Score
            s_prior = compute_prior_score(flows[i], flows[j])
            
            # C. Fusion
            w_ij = lambda_val * s_gnn + (1 - lambda_val) * s_prior
            W[i][j] = W[j][i] = w_ij

    print(f"Similarity Matrix W constructed. Avg Weight: {np.mean(W):.4f}")

    # 谱聚类 (Spectral Clustering)
    print("--- 4. Spectral Clustering ---")
    num_groups = 4  # 假设我们想分成 4 组
    
    # 计算拉普拉斯矩阵
    L = laplacian(W, normed=True)
    
    # 计算前 k 个特征向量 (Eigenvectors)
    # which='SM' 表示 Smallest Magnitude (最小特征值对应的特征向量)
    vals, vecs = eigsh(L, k=num_groups, which='SM')
    
    # K-Means 聚类
    kmeans = KMeans(n_clusters=num_groups, random_state=42)
    labels = kmeans.fit_predict(vecs)
    
    # 6. 展示结果
    print("\n--- Clustering Result ---")
    groups = {i: [] for i in range(num_groups)}
    for idx, label in enumerate(labels):
        groups[label].append(flows[idx]['flow_id'])
        
    for g_id, f_ids in groups.items():
        print(f"Group {g_id}: {len(f_ids)} flows -> {f_ids}")
        
    # 简单验证：同一组内的流是否大概率冲突（路径重叠）？
    # 或者周期是否接近？(这取决于模型学到了什么)    
    

    # [新增] --- 5. 保存分组结果供 Phase 2 使用 ---
    print("\n[Step 4] Saving Partitioned Data for Phase 2...")
    import pickle

    # 1. 整理数据结构
    # 我们需要保存：原始图、分组后的流列表
    partitioned_data = {
        'original_graph': graph,  # 原始拓扑
        'groups': {}              # 格式: {group_id: [flow1, flow2...]}
    }

    for idx, label in enumerate(labels):
        label = int(label) # 确保是 Python int
        if label not in partitioned_data['groups']:
            partitioned_data['groups'][label] = []
        # 将原始流信息存入对应组
        partitioned_data['groups'][label].append(flows[idx])

    # 2. 保存文件
    output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    save_file = os.path.join(output_dir, "partition_result.pkl")

    with open(save_file, 'wb') as f:
        pickle.dump(partitioned_data, f)
    
    print(f"  [Success] Partitioned data saved to: {save_file}")
    print(f"  Stats: {len(partitioned_data['groups'])} groups created.")
    for gid, gflows in partitioned_data['groups'].items():
        print(f"    - Group {gid}: {len(gflows)} flows")

    print("\n[Success] Phase 1 Inference Completed.")

if __name__ == "__main__":
    main()