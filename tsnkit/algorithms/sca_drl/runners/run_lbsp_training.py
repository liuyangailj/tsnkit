import sys
import os
import yaml
import torch

# 路径 Hack: 确保能找到 sca_drl 和 tsnkit
curr_dir = os.path.dirname(os.path.abspath(__file__)) # runner/
algorithm_dir = os.path.dirname(os.path.dirname(curr_dir)) # algorithm/
tsnkit_dir = os.path.dirname(algorithm_dir) #tsnkit/

sys.path.append(algorithm_dir)

# 引入合并后的 Core 模块
from sca_drl.phase1_lbsp_training import (
    TSNPhase1Dataset, 
    GNNPartitionModel, 
    train_phase1
)

def main():
    # 1. 基础配置
    # 实际使用时建议使用 argparse 解析命令行参数
    config = {
        'k_paths': 3,
        'model': {
            'input_dim': 3,  # [size, period, deadline]
            'hidden_dim': 64,
            'output_dim': 16
        },
        'train': {
            'lr': 0.005,
            'epochs': 50,
            'save_path': '../results/checkpoints/gnn_phase1_new.pth'
        }
    }
    
    # 2. 指定 TSNKit 数据路径
    # 这里直接指向 tsnkit 的测试数据
    tsn_data_dir = os.path.join(tsnkit_dir, 'test/data')
    task_file = os.path.join(tsn_data_dir, '1_task.csv')
    topo_file = os.path.join(tsn_data_dir, '1_topo.csv')
    
    if not os.path.exists(task_file):
        print(f"Error: Data file not found at {task_file}")
        return

    # 3. 初始化 Dataset (这一步会自动调用 tsnkit 加载并构建冲突图)
    dataset = TSNPhase1Dataset(task_file, topo_file, config)
    
    # 4. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNPartitionModel(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim']
    ).to(device)
    
    # 5. 开始训练
    train_phase1(model, dataset, config['train'], device)

if __name__ == "__main__":
    main()