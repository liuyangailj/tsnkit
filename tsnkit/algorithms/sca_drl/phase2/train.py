from logging import config
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import time
import glob
import random
import numpy as np
import torch
from tsnkit import core as utils_tsnkit 
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from environment import TSNEnv
from agent import PPOAgent
from sca_drl.common.utils import (
    set_seed, load_config, resolve_path, get_device, ensure_dir
)

def load_phase1_embeddings(config):
    data_cfg = config.get("data", {})
    emb_path = data_cfg.get("phase1_embeddings_pt")
    if emb_path:
        emb_full_path = resolve_path(emb_path)
        if os.path.isfile(emb_full_path):
            print(f"Loading Phase 1 embeddings from: {emb_full_path}")
            return torch.load(emb_full_path, map_location="cpu")
        else:
            print(f"Warning: Phase 1 embedding file not found: {emb_full_path}")
    return None

# ... (保留最顶部的 imports, set_seed 等) ...

def train(config):
    print("="*60)
    print("🚀 SCA-DRL Phase 2: 终极泛化训练引擎启动 (带验证集闭环) 🚀")
    print("="*60)
    
    set_seed(config.get("seed", 42))   
    
    data_cfg = config.get("data", {})
    task_files_path = resolve_path(data_cfg["task_dir"])
    topo_path = resolve_path(data_cfg["topo_file"])    
    
    # 🌟 1. 严格隔离 Train 和 Val 数据集！(拒绝数据泄露)
    train_files = glob.glob(os.path.join(task_files_path, "[0-9]*_task.csv"))
    val_files = glob.glob(os.path.join(task_files_path, "val_*_task.csv"))
    
    if not train_files:
        print("❌ 找不到训练数据！")
        return   
    
    print(f"[1/4] 成功发现拓扑与考卷: 训练集 {len(train_files)} 份，验证集 {len(val_files)} 份！")
    topo = utils_tsnkit.load_network(topo_path)     
      
    print("[2/4] 初始化 TSNEnv 与 Transformer PPOAgent...")
    
    # 环境初始化
    initial_task_file = train_files[0]
    initial_task = utils_tsnkit.load_stream(initial_task_file)  
    env_config = {
        'environment': config.get("environment", {}),
        'task': initial_task,
        'task_file': initial_task_file, # 传入路径
        'topo': topo         
    }
    
    # 🌟 剔除了老的 phase1_emb 传参，环境现在内部动态读取
    env = TSNEnv(env_config)
    print(f"✅ 环境初始化成功！流数量: {env.num_flows}, 特征维度: {env.d_feature}")
    
    agent = PPOAgent(env=env, config=config)
    initial_lr = config.get('agent', {}).get('ppo', {}).get('learning_rate', 3e-4)
   
    # MLOps 配置
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')    
    run_dir = resolve_path(f"./phase2/runs/phase2_ppo_{current_time}")
    model_dir = resolve_path(f"./phase2/models/phase2_{current_time}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # === 恢复你的硬盘级双轨日志系统 ===
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush() 

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(os.path.join(run_dir, "train_log.txt"))
    sys.stderr = sys.stdout  
    print(f"📁 本次实验日志已挂载: {run_dir}/train_log.txt")
    
    writer = SummaryWriter(log_dir=run_dir)

    train_cfg = config.get("training", {})
    total_iterations = train_cfg.get("total_iterations", 200)
    episodes_per_iter = train_cfg.get("episodes_per_iter", 8)
    
    print(f"[3/4] 训练规划: 共 {total_iterations} 轮, 每轮收集 {episodes_per_iter} 份考卷")
    print("[4/4] ⚔️ 面对风暴吧！Transformer！\n")

    best_val_success = -1.0 # 用于保存最佳模型

    for iteration in range(1, total_iterations + 1):
        # --- 学习率衰减 ---
        frac = 1.0 - (iteration - 1.0) / total_iterations
        lr_now = frac * initial_lr
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = lr_now        
        
        rollouts = { 'flow_tokens': [], 'global_snapshot': [], 'action_masks': [], 'actions': [], 
                     'logprobs': [], 'rewards': [], 'values': [], 'dones': [] }
        
        iter_rewards, iter_success_rates, iter_avg_hops = [], [], []

        # ==========================================
        # 🏋️‍♂️ 阶段 A：训练模式 (Train Loop)
        # ==========================================
        agent.network.train()
        for ep in range(episodes_per_iter):
            # 严格从 train_files 里抽题！
            random_task_file = random.choice(train_files)
            new_task = utils_tsnkit.load_stream(random_task_file)   
            env.load_new_task(new_task, random_task_file)
            
            obs, _ = env.reset()
            ep_reward, ep_hops = 0.0, []
            
            for step in range(env.num_flows):
                rollouts['flow_tokens'].append(obs['flow_tokens'])
                rollouts['global_snapshot'].append(obs['global_snapshot'])
                rollouts['action_masks'].append(obs['action_mask'])
                
                # 采样动作 (探索)
                action, logprob, entropy, value = agent.get_action_and_value(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                rollouts['actions'].append(action)
                rollouts['logprobs'].append(logprob.item())
                rollouts['rewards'].append(reward)
                rollouts['values'].append(value.item())
                rollouts['dones'].append(terminated) 
                
                ep_reward += reward
                obs = next_obs
                
                if info.get('is_allocated', False) and 'hop_count' in info:
                    ep_hops.append(info['hop_count'])
                
                if terminated:
                    if 'group_success_rate' in info:
                        iter_success_rates.append(info['group_success_rate'])
                    iter_avg_hops.append(sum(ep_hops)/len(ep_hops) if ep_hops else 0.0)
                    break
            
            iter_rewards.append(ep_reward)

        # --- PPO 参数更新 ---
        pg_loss, v_loss, ent = agent.update(rollouts)

        avg_reward = np.mean(iter_rewards)
        avg_success = np.mean(iter_success_rates) * 100 if iter_success_rates else 0.0
        avg_hop = np.mean(iter_avg_hops) if iter_avg_hops else 0.0
        
        # print(f"Iter {iteration:03d} | Train 成功率: {avg_success:5.1f}% | Reward: {avg_reward:7.2f} | P_Loss: {pg_loss:6.3f}")

        # 终端打印 (你之前已经加回来的)
        print(f"Iter {iteration:03d} | 总流数：{env.num_flows} | Train 成功率: {avg_success:5.1f}% | 均跳数: {avg_hop:4.2f} | Reward: {avg_reward:7.2f} | P_Loss: {pg_loss:6.3f} | V_Loss: {v_loss:6.3f} | Ent: {ent:5.3f}")

        writer.add_scalar("Train/1_Success_Rate", avg_success, iteration)
        writer.add_scalar("Train/2_Avg_Reward", avg_reward, iteration)
        writer.add_scalar("Loss/1_Policy_Loss", pg_loss, iteration)
        writer.add_scalar("Loss/2_Value_Loss", v_loss, iteration)
        # 🌟 加上这极其重要的一行！恢复 Entropy 监控！
        writer.add_scalar("Loss/3_Entropy", ent, iteration)

        # ==========================================
        # 🧪 阶段 B：验证模式 (Validation Loop) - 每 10 轮考一次
        # ==========================================
        if iteration % 10 == 0 and val_files:
            agent.network.eval() # 关闭 Dropout 等
            val_success_rates = []
            
            # 去做 10 张固定的验证卷 (不求导，纯测试)
            with torch.no_grad():
                # 为了速度，我们随机挑 10 张验证卷来考
                eval_files = random.sample(val_files, min(10, len(val_files)))
                for v_file in eval_files:
                    v_task = utils_tsnkit.load_stream(v_file)
                    env.load_new_task(v_task, v_file)
                    obs, _ = env.reset()
                    
                    for step in range(env.num_flows):
                        # 测试时直接取 argmax，不引入随机性
                        flow_tokens = torch.FloatTensor(obs['flow_tokens']).unsqueeze(0).to(agent.device)
                        g_global = torch.FloatTensor(obs['global_snapshot']).unsqueeze(0).to(agent.device)
                        mask = torch.BoolTensor(obs['action_mask']).unsqueeze(0).to(agent.device)
                        
                        logits, _ = agent.network(flow_tokens, g_global)
                        logits = logits.masked_fill(~mask, -1e8)
                        action = torch.argmax(logits, dim=-1).item() # 🌟 绝对确定性的选择
                        
                        obs, _, terminated, _, info = env.step(action)
                        if terminated:
                            val_success_rates.append(info.get('group_success_rate', 0.0))
                            break
                            
            avg_val_success = np.mean(val_success_rates) * 100
            print(f"   => 🏆 [期中考试] 验证集成功率: {avg_val_success:5.1f}%")
            writer.add_scalar("Eval/1_Val_Success_Rate", avg_val_success, iteration)
            
            # 🌟 保存“泛化能力最强”的最佳模型
            if avg_val_success > best_val_success:
                best_val_success = avg_val_success
                best_path = os.path.join(model_dir, "phase2_ppo_best.pth")
                torch.save(agent.network.state_dict(), best_path)
                print(f"   => 🌟 [突破纪录] 最优模型已更新并保存!")

    writer.close()
    print("\n🎉 训练完美收官！")

if __name__ == "__main__":    
    import argparse
    from sca_drl.common.utils import load_config
    
    parser = argparse.ArgumentParser(description="Phase 2 PPO Training")
    parser.add_argument("--config", default="configs/phase2.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)