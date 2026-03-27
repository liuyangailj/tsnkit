import sys
import os
import torch
import random
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))
algorithm_dir = os.path.dirname(os.path.dirname(curr_dir))
tsnkit_dir = os.path.dirname(algorithm_dir)
sys.path.append(algorithm_dir)

from sca_drl.phase1_lbsp_training import (
    TSNPhase1Dataset, 
    GNNPartitionModel, 
    train_phase1
)
from tsnkit import core as utils

def main():
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(42)
    
    # 对齐论文的统一超参表
    config = {
        'k_paths': 3,
        'model': {
            'input_dim': 3,  # [size, log(period), log(deadline)]
            'hidden_dim': 16,
            'output_dim': 16,
            'heads': 4
        },
        'train': {
            'lr': 0.005,
            'epochs': 150, # GNN 训练很快，可以适当提高 epoch
            'save_path': '../results/checkpoints/gnn_phase1_new.pth'
        }
    }
    
    # 读取数据
    tsn_data_dir = os.path.join(tsnkit_dir, 'test/data')
    task_file = os.path.join(tsn_data_dir, '1_task.csv')
    topo_file = os.path.join(tsn_data_dir, '1_topo.csv')
    
    if not os.path.exists(task_file) or not os.path.exists(topo_file):
        print(f"Error: Task file or Topology file not found in {tsn_data_dir}")
        return
    
    # 使用 TSNKit 原生加载器读取数据
    print(f"Building Graph Dataset with K={config['k_paths']}...")
    dataset = TSNPhase1Dataset(task_file, topo_file, k_paths=config['k_paths'])
    data = dataset[0]
    
    print(f"Graph built: {data.num_nodes} nodes, {data.edge_index.size(1)} physical conflict edges.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNPartitionModel(**config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])

    # 开始训练
    print(f"Training GAT Topology Reconstructor on {device}...")
    train_phase1(model, data, optimizer, epochs=config['train']['epochs'], device=device)

    # 保存权重
    os.makedirs(os.path.dirname(config['train']['save_path']), exist_ok=True)
    torch.save(model.state_dict(), config['train']['save_path'])
    print(f"Model saved to {config['train']['save_path']}")

if __name__ == '__main__':
    main()