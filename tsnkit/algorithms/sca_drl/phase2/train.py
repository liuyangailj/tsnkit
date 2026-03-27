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

def train(config):
    print("="*60)
    print("🚀 SCA-DRL Phase 2: 终极泛化训练引擎启动 (Padding & Masking) 🚀")
    print("="*60)
    
    # 1. 🔧 设置随机数种子
    set_seed(config.get("seed", 42))   
    
    data_cfg = config.get("data", {})
    task_files_path = resolve_path(data_cfg["task_dir"])
    topo_path = resolve_path(data_cfg["topo_file"])    
    data_dir = os.path.dirname(task_files_path)
    
    train_cfg = config.get("training", {})
    total_iterations = train_cfg.get("total_iterations", 200)
    episodes_per_iter = train_cfg.get("episodes_per_iter", 8)
    
    # 2. 自动扫描数据风暴文件夹    
    task_files = glob.glob(os.path.join(data_dir, "*_task.csv"))    
    if not os.path.exists(topo_path) or len(task_files) == 0:
        print("❌ 找不到数据！请先运行 generate_data.py 生成 data_storm 文件夹！")
        return   
    
    print(f"[1/4] 成功发现多路径网格拓扑与 {len(task_files)} 份随机流量考卷！")
    topo = utils_tsnkit.load_network(topo_path)     
    # 随意加载一个 task 用于初始化环境
    initial_task = utils_tsnkit.load_stream(task_files[0])  
      
    # 3. 🌟 加载 Phase1 产生的 Embeddings
    print("[2/4] 初始化 TSNEnv 与 Transformer PPOAgent...")
    phase1_emb = load_phase1_embeddings(config)
    
    env_config = {
    'environment': config.get("environment", {}),
    'task': initial_task,
    'topo': topo         
}
    
    # 4. 初始化环境与智能体
    env = TSNEnv(env_config, phase1_embeddings=phase1_emb)
    print(f"✅ 环境初始化成功！流数量: {env.num_flows}, 特征维度: {env.d_feature}")
    
    agent = PPOAgent(env=env,config=config)

    initial_lr = config.get('agent', {}).get('ppo', {}).get('learning_rate', 3e-4)  # 记录初始学习率，供线性衰减使用
   
    # ==========================================
    # 🛡️ MLOps: 实验追踪与日志持久化系统
    # ==========================================
    # 1. 生成全局唯一的实验时间戳
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')    
    # 2. 为本次实验建立【专属】的 TensorBoard 和 模型保存 文件夹
    run_dir = f"./runs/stage2_multipath_{current_time}"
    model_dir = f"./models/stage2_{current_time}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 3. 拦截器：把所有的 print() 输出同时打在屏幕上，并实时存入硬盘 txt
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()  # 👈 核心救命代码：强制每写一行就存入硬盘，就算立刻断电死机，日志也在！

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    # 将系统标准输出和报错输出全部接管到我们的 log 文件中
    sys.stdout = Logger(os.path.join(run_dir, "train_log.txt"))
    sys.stderr = sys.stdout  
    # 4. 启动 TensorBoard 记录器
    writer = SummaryWriter(log_dir=run_dir)

    print(f"🚀 SCA-DRL 训练引擎启动！")
    print(f"📁 本次实验日志路径: {run_dir}/train_log.txt")
    print(f"💾 本次模型保存路径: {model_dir}")
    
        
    
    print(f"[3/4] 训练规划: 共 {total_iterations} 轮, 每轮收集 {episodes_per_iter} 份考卷")
    print("[4/4] ⚔️ 面对风暴吧！Transformer！\n")

    start_time = time.time()
    
    for iteration in range(1, total_iterations + 1):
        # ==========================================
        # 📉 架构师的微调法宝：线性学习率衰减 (Linear LR Annealing)
        # ==========================================
        # 计算当前剩余比例 (第 1 轮是 1.0，第 100 轮是 0.0)
        frac = 1.0 - (iteration - 1.0) / total_iterations
        # 计算当前应该使用的学习率
        lr_now = frac * initial_lr
        
        # 强行霸道地修改 PyTorch 优化器里的学习率
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = lr_now        
        
        rollouts = { 'flow_tokens': [], 'global_snapshot': [], 'action_masks': [], 'actions': [], 
                     'logprobs': [], 'rewards': [], 'values': [], 'dones': [] } # 新增 'dones' 用于存储每步是否结束的信息
        
        iter_rewards = []
        iter_success_rates = []
        iter_avg_hops = [] # 新增：每轮的平均跳数统计

        # --- 循环刷题阶段 ---
        for ep in range(episodes_per_iter):
            # 🌟 核心：随机抽一张考卷，并让环境更新它的物理引擎！
            random_task_file = random.choice(task_files)
            new_task = utils_tsnkit.load_stream(random_task_file)   
            # new_task = utils.load_stream(task_files[1])         
            env.load_new_task(new_task)
            
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_hops = []
            
            # 哪怕最大容量是 100，我们这一局实际只需要调度真实的 env.num_flows 次
            for step in range(env.num_flows):
                rollouts['flow_tokens'].append(obs['flow_tokens'])
                rollouts['global_snapshot'].append(obs['global_snapshot'])
                rollouts['action_masks'].append(obs['action_mask'])
                
                action, logprob, entropy, value = agent.get_action_and_value(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                rollouts['actions'].append(action)
                rollouts['logprobs'].append(logprob.item())
                rollouts['rewards'].append(reward)
                rollouts['values'].append(value.item())
                rollouts['dones'].append(terminated) # 🌟存储这一轮是否结束的信息,这步是不是把大结局记下来
                
                ep_reward += reward
                obs = next_obs
                
                # 🌟 新增：如果这步排流成功了，把 info 里传出来的跳数记下来
                if info.get('is_allocated', False) and 'hop_count' in info:
                    ep_hops.append(info['hop_count'])
                
                if terminated:
                    if 'group_success_rate' in info:
                        iter_success_rates.append(info['group_success_rate'])
                    
                    # 🌟 新增：计算这个 Episode 的平均跳数，存进全局列表
                    if ep_hops:
                        iter_avg_hops.append(sum(ep_hops) / len(ep_hops))
                    else:
                        iter_avg_hops.append(0.0)
                    
                    break
            
            iter_rewards.append(ep_reward)

        # --- PPO 梯度回传 ---
        pg_loss, v_loss, ent = agent.update(rollouts)

        avg_reward = np.mean(iter_rewards)
        avg_success = np.mean(iter_success_rates) * 100 if iter_success_rates else 0.0
        avg_hop = np.mean(iter_avg_hops) if iter_avg_hops else 0.0 # 🌟 新增：算出这 8 局的整体平均跳数
        
        # print(f"Iter {iteration:03d} | 成功率: {avg_success:5.1f}% | Reward: {avg_reward:7.2f} | P_Loss: {pg_loss:6.3f} | V_Loss: {v_loss:6.3f} | Ent: {ent:5.3f}")
        print(f"Iter {iteration:03d} | 总流数：{env.num_flows} | 成功率: {avg_success:5.1f}% | 均跳数: {avg_hop:4.2f} | Reward: {avg_reward:7.2f} | P_Loss: {pg_loss:6.3f} | V_Loss: {v_loss:6.3f} | Ent: {ent:5.3f}")

        writer.add_scalar("Metrics/1_Success_Rate", avg_success, iteration)
        writer.add_scalar("Metrics/2_Avg_Reward", avg_reward, iteration)
        writer.add_scalar("Metrics/3_Avg_Hops", avg_hop, iteration) # 🌟 新增：画出跳数下降的完美曲线！
        
        writer.add_scalar("Metrics/4_Learning_Rate", lr_now, iteration) # 🌟 新增：监控学习率的下降轨迹
        
        writer.add_scalar("Loss/1_Policy_Loss", pg_loss, iteration)
        writer.add_scalar("Loss/2_Value_Loss", v_loss, iteration)
        writer.add_scalar("Loss/3_Entropy", ent, iteration)

        # 每 50 轮或者最后一轮，保存一次模型
        if iteration % 50 == 0 or iteration == total_iterations:
            # 注意这里：把硬编码的 "./models" 换成了我们刚才动态生成的 model_dir
            save_path = os.path.join(model_dir, f"ppo_stage2_iter_{iteration}.pth")
            torch.save(agent.network.state_dict(), save_path)
            print(f"💾 模型已保存至: {save_path}")

    writer.close()
    print("\n🎉 训练完美收官！")

if __name__ == "__main__":    
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 PPO Training")
    parser.add_argument("--config", default="configs/phase2.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    train(cfg) 