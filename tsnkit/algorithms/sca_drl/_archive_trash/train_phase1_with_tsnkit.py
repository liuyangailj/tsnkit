"""
Phase 1 Training Script with TSNKit CSV Data

Trains the GNN model for stream partitioning using TSNKit CSV format.
"""

import sys
import os
import torch
import argparse
import importlib.util

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (即 runners/ 的上一级)
project_root = os.path.dirname(current_dir)
# 将根目录加入 python 查找路径 (insert at 0 to avoid conflicts)
sys.path.insert(0, project_root)

from phase1_partitioning.dataset import FlowGraphDataset
from phase1_partitioning.gnn_model import CorrelationModel
from phase1_partitioning.train import GNNTrainer

# Load tsnkit_reader module explicitly to avoid import conflicts
tsnkit_reader_path = os.path.join(project_root, 'common', 'tsnkit_reader.py')
tsnkit_reader_spec = importlib.util.spec_from_file_location("tsnkit_reader", tsnkit_reader_path)
tsnkit_reader = importlib.util.module_from_spec(tsnkit_reader_spec)
tsnkit_reader_spec.loader.exec_module(tsnkit_reader)


def get_default_config():
    """Return default configuration for Phase 1 training."""
    return {
        'model': {
            'learning_rate': 0.001,
            'gnn_hidden_dim': 64,
        },
        'train': {
            'epochs': 100,
            'save_path': 'phase1_gnn.pth'
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Train Phase 1 GNN with TSNKit CSV data')
    parser.add_argument('--task', type=str, required=True, help='Path to TSNKit task CSV file')
    parser.add_argument('--topo', type=str, required=True, help='Path to TSNKit topology CSV file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='GNN hidden dimension')
    parser.add_argument('--output', type=str, default=None, help='Model save path (default: models/phase1_gnn.pth)')
    parser.add_argument('--k', type=int, default=5, help='Number of K-shortest paths')

    args = parser.parse_args()

    print(f"Loading TSNKit data from {args.task} and {args.topo}...")

    # Load TSNKit CSV data
    flows, G = tsnkit_reader.read_tsnkit_data(args.task, args.topo, k=args.k)

    print(f"Loaded {len(flows)} flows")

    # Prepare dataset
    print("Preparing graph dataset...")
    dataset = FlowGraphDataset(flows, G)
    data = dataset.process()
    print(f"Graph Info: {data.num_nodes} nodes, {data.num_edges} edges")

    # Initialize model
    # Input feature dimension = 4 (Period, Size, Deadline, Jitter)
    print("Initializing model...")
    config = get_default_config()
    config['model']['learning_rate'] = args.lr
    config['model']['gnn_hidden_dim'] = args.hidden_dim
    config['train']['epochs'] = args.epochs

    model = CorrelationModel(in_dim=4,  # 4 features: period, size, deadline, jitter
                             hidden_dim=args.hidden_dim,
                             embed_dim=32)

    print("Start training...")
    trainer = GNNTrainer(model, data, config)
    trainer.train()

    # Determine save path
    if args.output:
        save_path = args.output
    else:
        # Default save to models/ directory
        models_dir = os.path.join(project_root, 'models')
        os.makedirs(models_dir, exist_ok=True)
        save_path = os.path.join(models_dir, 'phase1_gnn.pth')

    # Save model
    trainer.save_model(save_path)

    print(f"\n[SUCCESS] Phase 1 Training Complete! Model saved to: {save_path}")
    print(f"   - Model: CorrelationModel (input_dim=4, hidden_dim={args.hidden_dim}, embed_dim=32)")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Flows: {len(flows)}")


if __name__ == "__main__":
    main()
