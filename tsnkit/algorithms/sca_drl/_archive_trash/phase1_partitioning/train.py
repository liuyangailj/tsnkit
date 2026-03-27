import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.utils import negative_sampling

class GNNTrainer:
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.data.to(self.device)
        
        # 优化器
        lr = config['model']['learning_rate']
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 损失函数: 二元交叉熵 (因为我们预测 0~1 的概率)
        self.criterion = nn.BCELoss()

    def train(self):
        """执行完整的训练流程"""
        epochs = self.cfg['train']['epochs'] # 需要在 config 里补上这个参数
        self.model.train()

        # 边索引
        edge_index = self.data.edge_index

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # --- 1. 正样本 (Positive Edges) ---
            # 直接使用图中的真实边
            pos_u, pos_v = edge_index[0], edge_index[1]
            
            # 前向传播预测正样本
            pos_pred, _ = self.model(self.data.x, edge_index, pos_u, pos_v)
            pos_label = torch.ones_like(pos_pred) # 标签全是 1

            # --- 2. 负样本 (Negative Edges) ---
            # 随机采样不存在的边
            neg_edge_index = negative_sampling(
                edge_index=edge_index, 
                num_nodes=self.data.num_nodes,
                num_neg_samples=pos_u.size(0) # 保持 1:1 的正负比例
            )
            neg_u, neg_v = neg_edge_index[0], neg_edge_index[1]
            
            # forward 前向传播预测负样本
            neg_pred, _ = self.model(self.data.x, edge_index, neg_u, neg_v)
            neg_label = torch.zeros_like(neg_pred) # 标签全是 0

            # --- 3. 计算 Loss ---
            # forward 拼接正负样本的预测和标签
            all_pred = torch.cat([pos_pred, neg_pred])
            all_label = torch.cat([pos_label, neg_label])
            
            loss = self.criterion(all_pred, all_label)

            # --- 4. backward 反向传播 调整参数矩阵W---
            loss.backward()
            self.optimizer.step()

            # 打印日志
            if (epoch + 1) % 10 == 0:
                acc = ((all_pred > 0.5) == (all_label > 0.5)).float().mean()
                print(f"Epoch {epoch+1:03d}/{epochs}: Loss = {loss.item():.4f}, Acc = {acc:.4f}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")