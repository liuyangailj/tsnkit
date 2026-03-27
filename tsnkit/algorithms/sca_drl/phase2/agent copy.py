"""
Phase 2 Transformer-PPO 智能体模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CongestionAwareTransformer(nn.Module):
    """
    Transformer 编码器，将输入状态特征序列编码成上下文感知的表示。
    """

    def __init__(self, d_feature, d_model=64, nhead=4, num_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_feature, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_proj(x)
        return self.encoder(x)


class PPOAgent:
    """
    PPO 智能体，采用共享 Transformer 编码器结构，分别生成流选择和路径选择策略分布。
    """
    def __init__(self, d_feature, k_max, config, device='cpu'):
        """
        Args:
            d_feature (int): 输入特征维度（流状态维度）
            k_max (int): 每条流最大候选路径数（动作空间维度）
            config (dict): 包含 'transformer' 和 'ppo' 配置参数
            device (str or torch.device): 运行设备
        """
        self.device = device
        self.k_max = k_max

        # 超参数
        ppo_cfg = config["ppo"]
        self.gamma = ppo_cfg["gamma"]
        self.eps_clip = ppo_cfg["eps_clip"]
        self.k_epochs = ppo_cfg["k_epochs"]
        self.entropy_coef = ppo_cfg["entropy_coef"]

        # Transformer 编码器
        tf_cfg = config["transformer"]
        self.encoder = CongestionAwareTransformer(
            d_feature=d_feature,
            d_model=tf_cfg["d_model"],
            nhead=tf_cfg["nhead"],
            num_layers=tf_cfg["num_layers"],
            d_ff=tf_cfg["d_ff"],
            dropout=tf_cfg.get("dropout", 0.1),
        ).to(device)

        # 策略头和价值头
        self.stream_head = nn.Linear(tf_cfg["d_model"], 1).to(device)      # 流选择 logits
        self.route_head = nn.Linear(tf_cfg["d_model"], k_max).to(device)  # 路径选择 logits

        self.value_head = nn.Sequential(
            nn.Linear(tf_cfg["d_model"], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)

        # 优化器
        params = list(self.encoder.parameters()) + \
                 list(self.stream_head.parameters()) + \
                 list(self.route_head.parameters()) + \
                 list(self.value_head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=ppo_cfg["learning_rate"])

        # 经验回放缓冲
        self.buffer = []

    def select_action(self, obs, scheduled_mask):
        """
        根据当前状态选择动作。

        Args:
            obs (np.ndarray): [N, d_feature] 流状态矩阵
            scheduled_mask (np.ndarray): [N], 已调度流的掩码

        Returns:
            action (tuple): (stream_idx, route_idx) 的整数索引
            log_prob (float): 动作对数概率，用于训练
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # [1, N, d_f]
        encoded = self.encoder(obs_t)                                # [1, N, d_model]

        # 流选择
        stream_logits = self.stream_head(encoded).squeeze(-1)  # [1, N]
        mask_tensor = torch.BoolTensor(scheduled_mask).unsqueeze(0).to(self.device)
        stream_logits = stream_logits.masked_fill(mask_tensor, float('-1e9'))
        stream_dist = Categorical(logits=stream_logits.squeeze(0))
        stream_idx = stream_dist.sample()

        # 路径选择
        stream_enc = encoded[0, stream_idx]                      # [d_model]
        route_logits = self.route_head(stream_enc)               # [k_max]
        route_dist = Categorical(logits=route_logits)
        route_idx = route_dist.sample()

        log_prob = (stream_dist.log_prob(stream_idx) + route_dist.log_prob(route_idx)).item()
        return (stream_idx.item(), route_idx.item()), log_prob

    def store_transition(self, obs, action, reward, log_prob, done):
        self.buffer.append((obs, action, reward, log_prob, done))

    def update(self):
        """
        PPO策略更新。函数实现可根据需要详细补充。
        这里请确保实现标准的 PPO 算法阶段。
        """
        # 训练代码实现细节略，可先沿用你已有版本。
        pass

    def save(self, path):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'stream_head': self.stream_head.state_dict(),
            'route_head': self.route_head.state_dict(),
            'value_head': self.value_head.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.stream_head.load_state_dict(ckpt['stream_head'])
        self.route_head.load_state_dict(ckpt['route_head'])
        self.value_head.load_state_dict(ckpt['value_head'])