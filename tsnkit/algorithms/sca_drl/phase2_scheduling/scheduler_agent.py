import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os

# -----------------------------------------
# 1. 辅助工具与网络定义 (Actor-Critic Network)
# -----------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    正交初始化 (Orthogonal Initialization)
    有助于 PPO 在训练初期保持梯度稳定，防止梯度消失或爆炸。
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=64):
        """
        PPO 核心网络：同时包含 Actor (策略) 和 Critic (价值)。
        采用了 Tanh 激活函数和正交初始化。
        """
        super(ActorCritic, self).__init__()

        # === Critic 网络 (Value Function V(s)) ===
        # 结构: Linear -> Tanh -> Linear -> Tanh -> Linear
        self.critic = nn.Sequential(
            layer_init(nn.Linear(num_inputs, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

        # === Actor 网络 (Policy Function pi(a|s)) ===
        # 结构: Linear -> Tanh -> Linear -> Tanh -> Linear
        self.actor = nn.Sequential(
            layer_init(nn.Linear(num_inputs, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            # 最后一层 std=0.01，使初始策略接近均匀分布，最大化探索
            layer_init(nn.Linear(hidden_dim, num_actions), std=0.01)
        )

    def get_value(self, x):
        """仅计算状态价值 V(s)"""
        return self.critic(x)

    def get_action_and_value(self, x, action=None, action_mask=None):
        """
        获取动作、对数概率、熵和价值。
        支持 Action Masking (防止选择非法动作)。
        """
        # 1. Critic 推理
        value = self.critic(x)
        
        # 2. Actor 推理 (Logits)
        logits = self.actor(x)
        
        # 3. Action Masking (关键逻辑)
        # 如果提供了 mask，将非法动作的 logits 设为极小的负数
        if action_mask is not None:               
            # 确保 mask 和 logits 在同一个 device
            if action_mask.device != logits.device:
                action_mask = action_mask.to(logits.device)
            # Masking: value 1 means valid, 0 means invalid
            # 我们将 invalid (0) 的位置设为 -1e8
            # 注意: 这里假设 action_mask 已经被扩展到与 logits 相同的维度 (num_actions)
            logits = torch.where(action_mask > 0, logits, torch.tensor(-1e8).to(logits.device))

        # 4. 构建分布 (Categorical 内部包含 Softmax)
        probs = Categorical(logits=logits)
        
        # 5. 采样或评估动作
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value

# -----------------------------------------
# 2. PPO 智能体封装 (Agent Wrapper)
# -----------------------------------------

class PPOAgent:
    def __init__(self, obs_dim, num_flows, num_paths, 
                 lr=1e-3, gamma=0.99, eps_clip=0.2, k_epochs=4, device='cpu'):
        """
        PPO Agent 管理器：严格保留了原始定义的接口参数。
        
        Args:
            obs_dim (int): 状态空间维度
            num_flows (int): 流的数量
            num_paths (int): 每个流的候选路径数量 (K)
            lr (float): 学习率
            gamma (float): 折扣因子
            eps_clip (float): PPO 裁剪参数
            k_epochs (int): 每次更新的迭代次数
            device (str): 运行设备 'cpu' 或 'cuda'
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = k_epochs
        
        self.num_flows = num_flows
        self.num_paths = num_paths
        
        # --- 动作空间计算 ---
        # 既然动作是"选哪个流" + "走哪条路"，我们将动作空间展平。
        # Action ID 0 ~ (num_flows * num_paths - 1)
        # 例如: Action 0 = Flow 0 on Path 0; Action 1 = Flow 0 on Path 1...
        self.action_dim = num_flows * num_paths
        
        self.buffer = RolloutBuffer() # 使用 RolloutBuffer 类实例

        # 初始化策略网络
        self.policy = ActorCritic(obs_dim, self.action_dim).to(self.device)
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        # 旧策略网络 (用于计算 Ratio)
        self.policy_old = ActorCritic(obs_dim, self.action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, action_mask=None):
        """
        在推理/采样阶段选择动作，并将状态、动作、LogProb 存入 Buffer。
        """
        with torch.no_grad():
            # 确保 state 是 tensor 且在正确的 device 上
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0) # 增加 batch 维度 ,把[128]变成[1,128]
            
            # 处理 mask
            if action_mask is not None:
                if isinstance(action_mask, list):
                    action_mask = np.array(action_mask)
                if isinstance(action_mask, np.ndarray):
                    action_mask = torch.BoolTensor(action_mask).to(self.device)
                if action_mask.dim() == 1:
                    action_mask = action_mask.unsqueeze(0)
                
                # === 关键修复 1：Mask 维度扩展 ===
                # 如果传入的 Mask 是 [Batch, Flows]，扩展为 [Batch, Flows * Paths]
                if action_mask.shape[1] == self.num_flows and self.action_dim == self.num_flows * self.num_paths:
                    action_mask = action_mask.unsqueeze(-1).repeat(1, 1, self.num_paths).reshape(action_mask.shape[0], -1)
            
            # 使用新接口获取动作 (扁平化 ID, 神经网络使用)
            action_flat, log_prob, _, value = self.policy_old.get_action_and_value(state, action_mask=action_mask)
            
            # === 将数据存入 Buffer ===
            # run_sca_experiment.py 中后续会 append rewards，但 states/actions/logprobs 需要在这里存
            self.buffer.states.append(state)  # 当时的状态
            self.buffer.actions.append(action_flat) # 当时选了啥
            self.buffer.logprobs.append(log_prob)  # 当时选它的概率（取对数）

            # === 关键修复 2：动作解码 ===
            # 将扁平化 ID 解码为 (Flow_ID, Path_ID) 供 Environment 使用
            action_item = action_flat.item() 
            flow_id = int(action_item // self.num_paths)
            path_idx = int(action_item % self.num_paths)
            
        # 返回元组，解决 TypeError: cannot unpack non-iterable int object
        return (flow_id, path_idx)

    def update(self):
        """
        PPO 核心更新逻辑。
        不再接收 memory 参数，直接使用 self.buffer。
        """
        memory = self.buffer
        
        # 1. 转换数据类型
        # Memory.actions 可能包含元组 (flow_id, path_idx) 或 扁平化 Tensor
        try:
            # 尝试直接堆叠 (如果存的是 Tensors)
            old_actions_raw = torch.stack(memory.actions).detach().to(self.device)
        except:
            # 如果存的是 Python Tuples/Lists
            old_actions_raw = torch.tensor(memory.actions).detach().to(self.device)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(self.device)
        
        # === 关键修复 3：动作重编码 ===
        # 如果 old_actions 是二维 [Batch, 2] (即 flow_id, path_idx)，我们需要将其转回一维 flat index
        # 否则 PPO 的 log_prob 计算会出错
        if old_actions_raw.dim() > 1 and old_actions_raw.shape[-1] == 2:
            # Encoding: Flow_ID * Num_Paths + Path_ID
            old_actions = old_actions_raw[:, 0] * self.num_paths + old_actions_raw[:, 1]
        else:
            old_actions = torch.squeeze(old_actions_raw)

        # 2. 纯蒙特卡洛回报，仅累加折扣奖励，无V(s)参与
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # 2. 标准GAE计算示例 。。。 
        

        # 3. PPO 更新循环
        for _ in range(self.K_epochs):
            # 评估旧状态下的新策略
            # 注意: update 时通常无需 mask，因为我们是在回放已知轨迹
            _, logprobs, dist_entropy, state_values = self.policy.get_action_and_value(old_states, old_actions)
            
            state_values = torch.squeeze(state_values)
            
            # Ratios
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Advantage
            advantages = rewards - state_values.detach()
            
            # Loss (Clipped Surrogate Objective)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # 反向传播与优化
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer after update
        self.buffer.clear()

    def save(self, checkpoint_path):
        directory = os.path.dirname(checkpoint_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))