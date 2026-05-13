import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

# =====================================================================
# 核心工具：CleanRL 标配的正交初始化函数
# 作用：保证深层网络梯度传播的稳定性，防止梯度消失或爆炸
# =====================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# =====================================================================
# 模块一：神经网络架构 (Congestion-Aware Transformer)
# =====================================================================
class CongestionAwareTransformer(nn.Module):
    def __init__(self, num_flows, k_max, d_feature, global_dim, d_model, n_heads, n_layers, d_ff=None):
        super().__init__()
        self.num_flows = num_flows
        self.k_max = k_max
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # 1. 状态嵌入层 (State Embedding) - 严格对齐论文 ReLU 公式
        self.flow_embedder = nn.Sequential(
            layer_init(nn.Linear(d_feature, d_model)),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        self.global_embedder = nn.Sequential(
            layer_init(nn.Linear(global_dim, d_model * 2)),
            nn.ReLU(),
            layer_init(nn.Linear(d_model * 2, d_model)),
            nn.LayerNorm(d_model)
        )
        
        # 2. 拥塞感知核心大脑 (Transformer Encoder 共享躯干)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff if d_ff is not None else d_model * 4,
            batch_first=True,  
            norm_first=True    
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 3. Actor 策略输出头 (Actor Head)
        # 注意：Actor 最后一层初始化 std=0.01，让初始动作概率分布尽量均匀，鼓励早期探索
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(d_model, d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(d_model, k_max), std=0.01) 
        )
        
        # 4. Critic 价值评估头 (Critic Head)
        # 注意：Critic 最后一层初始化 std=1.0
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(d_model, d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(d_model, 1), std=1.0)
        )

    def forward(self, flow_tokens, global_snapshot):
        # ==========================================
        # 1. 构造 Transformer 的 Attention Mask (防污染)
        # ==========================================
        is_padding = (flow_tokens[..., 3] == -2.0) # [B, MAX_FLOWS]
        
        # global token 永远是真实的，不能被 mask 掉
        global_mask = torch.zeros((flow_tokens.shape[0], 1), dtype=torch.bool, device=flow_tokens.device)
        padding_mask = torch.cat([is_padding, global_mask], dim=1) # [B, MAX_FLOWS + 1]
        
        # ==========================================
        # 2. Embedding 与 纯净的 Transformer 编码
        # ==========================================
        
        flow_emb = self.flow_embedder(flow_tokens)         
        global_emb = self.global_embedder(global_snapshot).unsqueeze(1)         
        seq_emb = torch.cat([flow_emb, global_emb], dim=1) 
        
        # 🌟 核心：必须传给 Transformer，从源头掐断幽灵流的干扰！
        out_seq = self.transformer(seq_emb, src_key_padding_mask=padding_mask)                
        
        # 拆分特征
        out_flow = out_seq[:, :-1, :]       # [B, MAX_FLOWS, d_model] 所有Batch，从头取到倒数第二个token，所有特征
        global_repr = out_seq[:, -1, :]     # [B, d_model] 所有Batch，取最后一个token（全局token），所有特征
        
        # ==========================================
        # 3. 融合你的优雅版 Mask-aware Pooling (算 Critic)
        # ==========================================
        valid_mask = (~is_padding).float().unsqueeze(-1) # [B, MAX_FLOWS, 1]
        
        masked_flow_repr = out_flow * valid_mask
        sum_flow = masked_flow_repr.sum(dim=1)              # [B, d_model]
        num_flow = valid_mask.sum(dim=1).clamp(min=1.0)     # [B, 1]
        pooled_repr = sum_flow / num_flow                   # [B, d_model]
        
        # 🌟 把全局视野加回来，让 Critic 看到拥堵情况
        final_critic_state = pooled_repr + global_repr
        value = self.critic_head(final_critic_state)        # [B, 1]
        
        # ==========================================
        # 4. 算 Actor (策略分布)
        # ==========================================
        logits_2d = self.actor_head(out_flow)              
        logits = logits_2d.reshape(-1, self.num_flows * self.k_max) 
        
        return logits, value


# =====================================================================
# 模块二：PPO 智能体与训练引擎 (CleanRL Style)
# =====================================================================
class PPOAgent:
    def __init__(self, env, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 从配置字典中提取（config 结构为 {agent: {transformer: {}, ppo: {}}}）
        agent_cfg = config.get("agent", {})
        trans_cfg = agent_cfg.get("transformer", {})
        ppo_cfg   = agent_cfg.get("ppo", {})
        
        d_model = trans_cfg.get("d_model", 64)
        n_heads = trans_cfg.get("nhead", 4)
        n_layers = trans_cfg.get("num_layers", 3)
        d_ff = trans_cfg.get("d_ff", None)

        lr = ppo_cfg.get("learning_rate", 2.5e-4)
        self.gamma = ppo_cfg.get("gamma", 0.99)
        self.clip_coef = ppo_cfg.get("eps_clip", 0.2)
        self.ent_coef = ppo_cfg.get("entropy_coef", 0.01)
        self.vf_coef = ppo_cfg.get("vf_coef", 0.5)
        self.k_epochs = ppo_cfg.get("k_epochs", 4)
        self.gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
                
        self.num_flows = env.MAX_FLOWS
        self.k_max = env.K_MAX
        self.d_feature = env.d_feature # 🌟 这里会自动读取包含 Embedding 后的真实维度
        self.global_dim = env.observation_space['global_snapshot'].shape[0]
        
        self.network = CongestionAwareTransformer(
            num_flows=self.num_flows, k_max=self.k_max,
            d_feature=self.d_feature, global_dim=self.global_dim,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

# 🚨 彻底删除旧的 _get_action_mask 函数！我们不再自己算了！

    def get_action_and_value(self, obs, action=None):
        flow_tokens = torch.FloatTensor(obs['flow_tokens']).to(self.device) 
        g_global = torch.FloatTensor(obs['global_snapshot']).to(self.device)
        mask = torch.BoolTensor(obs['action_mask']).to(self.device) # 🌟 接收环境小抄
        
        # 智能升维
        if len(flow_tokens.shape) == 2:
            flow_tokens = flow_tokens.unsqueeze(0) 
        if len(g_global.shape) == 1:
            g_global = g_global.unsqueeze(0)
        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0)  # 🌟 Mask 同步升维
            
        logits, value = self.network(flow_tokens, g_global)
        
        if mask.sum() == 0:
            mask = torch.ones_like(mask, dtype=torch.bool)
            
        logits = logits.masked_fill(~mask, -1e8) 
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            return action.item(), probs.log_prob(action), probs.entropy(), value.squeeze()
        else:
            # 兼容 update 时传进来的 Tensor
            return action, probs.log_prob(action), probs.entropy(), value.squeeze()

    def update(self, rollouts):
        b_flow_tokens = torch.FloatTensor(np.array(rollouts['flow_tokens'])).to(self.device)
        b_global = torch.FloatTensor(np.array(rollouts['global_snapshot'])).to(self.device)
        b_actions = torch.LongTensor(rollouts['actions']).to(self.device)
        b_logprobs = torch.FloatTensor(rollouts['logprobs']).to(self.device)
        b_rewards = torch.FloatTensor(rollouts['rewards']).to(self.device)
        b_values = torch.FloatTensor(rollouts['values']).to(self.device)        
        # 🌟 修复核心：装载历史 Mask！
        b_masks = torch.BoolTensor(np.array(rollouts['action_masks'])).to(self.device) 
        b_dones = torch.BoolTensor(np.array(rollouts['dones'])).to(self.device) # 🌟 新增：装载历史 dones 信息
        
        batch_size = len(b_rewards)
        
        with torch.no_grad(): 
            advantages = torch.zeros_like(b_rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(batch_size)):
                if t == batch_size - 1:
                    nextnonterminal = 0.0
                    nextvalues = 0.0 
                else:
                    # 🌟 核心救命代码：如果是 True(1.0)，这步就是 0.0 (切断联系)
                    # 如果是 False(0.0)，这步就是 1.0 (保持相连)
                    if b_dones[t]: # 🌟 如果这一轮结束了
                        nextnonterminal = 0.0 # 🌟 切断联系 
                    else:
                        nextnonterminal = 1.0 # 🌟 保持相连
                        
                    nextvalues = b_values[t + 1]
                    # nextnonterminal = 1.0 
                    # nextvalues = b_values[t + 1]
                delta = b_rewards[t] + self.gamma * nextvalues * nextnonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + b_values

        # ===== 提取到这里！在整个 800 步的 batch 级别做归一化 =====
        advantages_std = advantages.std()
        if not (torch.isnan(advantages_std) or advantages_std == 0.0):
            advantages = (advantages - advantages.mean()) / (advantages_std + 1e-8)
        # ==========================================================
        
        b_inds = np.arange(batch_size)
        for epoch in range(self.k_epochs): 
            np.random.shuffle(b_inds) 
            
            for start in range(0, batch_size, 64):   
                end = start + 64
                mb_inds = b_inds[start:end]
                
                if len(mb_inds) <= 1:
                    continue
                
                logits, newvalues = self.network(b_flow_tokens[mb_inds], b_global[mb_inds])
                
                # 🌟 拿出当时的护盾，继续保护现在的网络计算
                mask = b_masks[mb_inds] 
                logits = logits.masked_fill(~mask, -1e8)
                
                probs = Categorical(logits=logits)
                newlogprob = probs.log_prob(b_actions[mb_inds]) 
                entropy = probs.entropy().mean() 
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = advantages[mb_inds]
                # std = mb_advantages.std()
                # if not (torch.isnan(std) or std == 0.0):
                #     mb_advantages = (mb_advantages - mb_advantages.mean()) / (std + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalues.squeeze() - returns[mb_inds]) ** 2).mean()
                loss = pg_loss - self.ent_coef * entropy + v_loss * self.vf_coef

                self.optimizer.zero_grad()  
                loss.backward()             
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) 
                self.optimizer.step()      
                
        return pg_loss.item(), v_loss.item(), entropy.item()