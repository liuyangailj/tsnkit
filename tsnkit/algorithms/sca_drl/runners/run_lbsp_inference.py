import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from sklearn.cluster import SpectralClustering
import pickle

# --- 路径设置 (与你原有结构一致) ---
curr_dir = os.path.dirname(os.path.abspath(__file__))
algorithm_dir = os.path.dirname(os.path.dirname(curr_dir))
tsnkit_dir = os.path.dirname(algorithm_dir)
sys.path.append(algorithm_dir)

from sca_drl.phase1_lbsp_training import (
    TSNPhase1Dataset, 
    GNNPartitionModel, 
    calculate_harmonic_prior
)
from tsnkit import core as utils

def compute_affinity_matrix(embeddings, streams, alpha=0.6):
    """
    实现白皮书公式: W_ij = alpha * s_gnn + (1 - alpha) * s_prior
    """
    num_streams = len(streams)
    W = np.zeros((num_streams, num_streams), dtype=np.float32)
    
    # 将 embedding 转为 tensor 算相似度
    #  O(N^2) 矩阵乘法计算全局余弦相似度
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    emb_norm = F.normalize(emb_tensor, p=2, dim=1)
    # s_gnn_matrix 的形状是 [N, N]，值域直接映射到了 [0, 1]
    s_gnn_matrix = ((torch.mm(emb_norm, emb_norm.t()) + 1.0) / 2.0).numpy()
    
    # 2. 融合时域谐波先验
    for i in range(num_streams):
        W[i, i] = 1.0
        for j in range(i + 1, num_streams):
            s_prior = calculate_harmonic_prior(streams[i].period, streams[j].period)
            W_ij = alpha * s_gnn_matrix[i, j] + (1 - alpha) * s_prior
            W[i, j] = W[j, i] = W_ij
            
    return W

def main():    
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 在 main() 开头调用：
    set_seed(42)
    
    # 1. 统一超参数配置
    config = {
        'k_paths': 3,
        'model': {'input_dim': 3, 'hidden_dim': 16, 'output_dim': 16, 'heads': 4},
        'inference': {
            'model_path': '../results/checkpoints/gnn_phase1_new.pth',
            'n_clusters': 4,
            'alpha': 0.6,
            'out_csv': '../results/phase1_groups.csv',
            'out_emb': '../results/phase1_embeddings.pt'  # 新增：供 Phase 2 调用的特征字典
        }
    }
    
    # 2. 数据加载
    tsn_data_dir = os.path.join(tsnkit_dir, 'test/data')
    task_file = os.path.join(tsn_data_dir, '1_task.csv')
    topo_file = os.path.join(tsn_data_dir, '1_topo.csv')
        
    print("Building Phase 1 Graph Data...")
    dataset = TSNPhase1Dataset(task_file, topo_file, k_paths=config['k_paths'])
    data = dataset[0]
    streams = data.streams
    
    # 3. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNPartitionModel(**config['model']).to(device)
    model.load_state_dict(torch.load(config['inference']['model_path'], map_location=device))
    model.eval()

    # 4. 前向传播获取 Embedding
    data = data.to(device)
    with torch.no_grad():
        print("Running GNN forward pass...")
        embeddings = model(data).cpu().numpy()  # [Num_Streams, 16]
        
    # 5. 计算混合亲和度矩阵与谱聚类
    alpha = config['inference']['alpha']
    W = compute_affinity_matrix(embeddings, streams, alpha)
    
    print(f"Executing Spectral Clustering (k={config['inference']['n_clusters']})...")
    sc = SpectralClustering(
        n_clusters=config['inference']['n_clusters'], 
        affinity='precomputed', 
        random_state=42
    )
    labels = sc.fit_predict(W)
    
    # ==========================================
    # 6. 数据落盘 (为 TSNKit 路由和 Phase 2 DRL 准备接口)
    # ==========================================
    # 6.1 保存 CSV 分组结果
    import csv
    os.makedirs(os.path.dirname(config['inference']['out_csv']), exist_ok=True)
    with open(config['inference']['out_csv'], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream_id", "group_id"])
        for sid, gid in zip(data.stream_ids, labels):
            w.writerow([sid, int(gid)])
            
    # 6.2 保存 Embedding 字典供 Phase 2 Transformer 调用！
    embedding_dict = {sid: embeddings[i] for i, sid in enumerate(data.stream_ids)}
    torch.save(embedding_dict, config['inference']['out_emb'])
    
    print(f"Phase 1 finished! Groups saved to {config['inference']['out_csv']}")
    print(f"Embeddings saved to {config['inference']['out_emb']} for Phase 2!")

if __name__ == '__main__':
    main()