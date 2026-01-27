import sys
import os
import torch
import numpy as np
import networkx as nx

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (即 runners/ 的上一级)
project_root = os.path.dirname(current_dir)
# 将根目录加入 python 查找路径
sys.path.append(project_root)

from common.topology_gen import TopologyGenerator
from common.flow_gen import FlowGenerator
from phase1_partitioning.dataset import FlowGraphDataset
from phase1_partitioning.gnn_model import CorrelationModel
from phase1_partitioning.train import GNNTrainer
# 导入配置加载工具
from common.config_utils import load_config

def main():
    # # 1. 加载配置    
    # 加载 Base，然后用 Phase 1 覆盖 Base
    config = load_config("configs/base_config.yaml", "configs/phase1_config.yaml")

    print(config['traffic']['num_flows'])  # 输出: 100 (来自 phase1_config.yaml)
    print(config['topology']['num_nodes']) # 输出: 20 (来自 base_config.yaml)

    print("--- 1. Generating Training Data ---")
    
    # 为了演示训练效果，稍微加大一点流的数量，让冲突图复杂一点
    config['traffic']['num_flows'] = 50 
    
    topo_gen = TopologyGenerator(config['topology']['num_nodes'])
    G = topo_gen.generate_graph()
    
    flow_gen = FlowGenerator(config, G)
    flows = flow_gen.generate_flows()
    
    print("--- 2. Preparing Graph Dataset ---")
    dataset = FlowGraphDataset(flows, G)
    data = dataset.process()
    print(f"Graph Info: {data.num_nodes} nodes, {data.num_edges} edges")

    print("--- 3. Initializing Model ---")
    # 输入特征维度=3 (Period, Size, Deadline)
    model = CorrelationModel(in_dim=3, 
                             hidden_dim=config['model']['gnn_hidden_dim'], 
                             embed_dim=32)

    print("--- 4. Start Training ---")
    trainer = GNNTrainer(model, data, config)
    trainer.train()
    
    # ------------------【修改点 2：构建绝对保存路径】------------------
    # 获取 yaml 中的相对路径 "results/checkpoints/gnn_phase1.pth"
    raw_save_path = config['train']['save_path']
    
    # 拼接成 D:\SCA-DRL\results\checkpoints\gnn_phase1.pth
    abs_save_path = os.path.join(project_root, raw_save_path)
    
    # 【新增】标准化路径 (把混合斜杠全部统一为 Windows 的反斜杠)
    abs_save_path = os.path.normpath(abs_save_path)
    
    # ------------------【修改点 3：确保目录存在】------------------
    # 根据绝对路径计算文件夹路径，并创建
    save_dir = os.path.dirname(abs_save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")
    
    # 保存模型，传入绝对路径给 trainer
    trainer.save_model(abs_save_path)
    # -----------------------------------------------------------
    
    print(f"\n✅ Phase 1 Training Complete! Model saved to: {abs_save_path}")

if __name__ == "__main__":
    main()
    