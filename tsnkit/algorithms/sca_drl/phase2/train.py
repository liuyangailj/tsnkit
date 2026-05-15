import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import time
import glob
from datetime import timedelta
import random
import numpy as np
from collections import defaultdict
import torch
from tsnkit import core as utils_tsnkit 
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from environment import TSNEnv
from agent import PPOAgent
from sca_drl.common.utils import (
    set_seed, load_config, resolve_path, get_device, ensure_dir
)

def sample_flow_count(progress, easy_n_pool, hard_n_pool, easy_ratio):
    if progress < easy_ratio:
        return random.choice(easy_n_pool)
    else:
        return random.choice(hard_n_pool)


def train(config, resume_path=None, version="v1"):
    print("="*60)
    print(f"🚀 SCA-DRL Phase 2: 终极泛化训练引擎启动 [{version}] (带验证集闭环) 🚀")
    print("="*60)

    set_seed(config.get("seed", 42))

    # 从 data_config.yaml 覆盖数据路径（单一事实来源）
    dc = load_config("configs/data_config.yaml")
    data_dir = resolve_path(dc["data_dir"])
    config["data"]["task_dir"] = data_dir
    config["data"]["topo_file"] = os.path.join(data_dir, "0_topo.csv")

    topo_path = resolve_path(config["data"]["topo_file"])

    # 根据 version 选择 train/val 子目录
    if version == "v2":
        train_subdir = dc.get("train_subdir_v2", "train_v2")
        val_subdir   = dc.get("val_subdir_v2",   "val_v2")
    else:
        train_subdir = "train"
        val_subdir   = "val"

    # 🌟 1. 严格隔离 Train 和 Val 数据集！(拒绝数据泄露)
    train_files_by_n = {}
    for n_dir in sorted(glob.glob(os.path.join(data_dir, train_subdir, "N*"))):
        n = int(os.path.basename(n_dir)[1:])
        files = glob.glob(os.path.join(n_dir, "*_task.csv"))
        if files:
            train_files_by_n[n] = files
    all_n_values = sorted(train_files_by_n.keys())

    # Val：v2 按 N 子目录分档，v1 平铺
    val_files_by_n = {}
    if version == "v2":
        for n_dir in sorted(glob.glob(os.path.join(data_dir, val_subdir, "N*"))):
            n = int(os.path.basename(n_dir)[1:])
            files = sorted(glob.glob(os.path.join(n_dir, "*_task.csv")))
            if files:
                val_files_by_n[n] = files
        val_files = [f for files in val_files_by_n.values() for f in files]
    else:
        val_files = glob.glob(os.path.join(data_dir, val_subdir, "*_task.csv"))

    if not train_files_by_n:
        print("❌ 找不到训练数据！")
        return

    total_train = sum(len(v) for v in train_files_by_n.values())
    print(f"[1/4] 成功发现拓扑与考卷: 训练集 {total_train} 份 ({len(all_n_values)} 档难度)，验证集 {len(val_files)} 份！")
    topo = utils_tsnkit.load_network(topo_path)     
      
    print("[2/4] 初始化 TSNEnv 与 Transformer PPOAgent...")
    
    # 环境初始化
    initial_task_file = train_files_by_n[all_n_values[0]][0]
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
   
    # MLOps 配置（断点续训时复用旧目录，保证 TensorBoard 曲线连续）
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    run_prefix = f"phase2_ppo_{version}"
    ckpt_data = None
    if resume_path and os.path.exists(resume_path):
        ckpt_data = torch.load(resume_path, map_location='cpu', weights_only=False)
        run_dir   = ckpt_data['run_dir']
        model_dir = os.path.dirname(os.path.abspath(resume_path))
    else:
        run_dir   = resolve_path(f"./runs/phase2/{run_prefix}_{current_time}")
        model_dir = resolve_path(f"./models/phase2/phase2_{version}_{current_time}")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # === 硬盘级双轨日志系统 ===
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
    total_iterations    = train_cfg.get("total_iterations", 200)
    episodes_per_iter   = train_cfg.get("episodes_per_iter", 8)
    checkpoint_interval = train_cfg.get("checkpoint_interval", 10)

    # 课程学习参数（从 yaml 读取，不再硬编码）
    curr_cfg    = train_cfg.get("curriculum", {})
    easy_ratio  = curr_cfg.get("easy_ratio",  0.4)
    easy_n_pool = curr_cfg.get("easy_n_pool", [140, 150, 160, 170])
    hard_n_pool = curr_cfg.get("hard_n_pool", [170, 180, 190, 200])

    # 恢复断点状态（Logger 之后执行，确保日志同时写入文件）
    start_iteration  = 1
    best_val_success = -1.0
    if ckpt_data is not None:
        agent.network.load_state_dict(ckpt_data['network_state_dict'])
        agent.optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
        start_iteration  = ckpt_data['iteration'] + 1
        best_val_success = ckpt_data['best_val_success']
        print(f"✅ 断点恢复成功: 从第 {start_iteration} 轮继续，历史最优验证率 {best_val_success:.1f}%")

    print(f"[3/4] 训练规划: 共 {total_iterations} 轮（当前从第 {start_iteration} 轮开始）, 每轮收集 {episodes_per_iter} 份考卷")
    print("[4/4] ⚔️ 面对风暴吧！Transformer！\n")

    train_start = time.time()

    for iteration in range(start_iteration, total_iterations + 1):
        iter_start = time.time()
        # --- 学习率衰减 ---
        frac = 1.0 - (iteration - 1.0) / total_iterations
        lr_now = frac * initial_lr
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = lr_now        
        
        rollouts = { 'flow_tokens': [], 'global_snapshot': [], 'action_masks': [], 'actions': [],
                     'logprobs': [], 'rewards': [], 'values': [], 'dones': [] }

        iter_rewards, iter_success_rates, iter_avg_hops, iter_n_values = [], [], [], []

        # ==========================================
        # 🏋️‍♂️ 阶段 A：训练模式 (Train Loop)
        # ==========================================
        agent.network.train()
        for ep in range(episodes_per_iter):
            progress = (iteration - 1) / total_iterations
            target_n = sample_flow_count(progress, easy_n_pool, hard_n_pool, easy_ratio)
            n_key = min(all_n_values, key=lambda k: abs(k - target_n))
            random_task_file = random.choice(train_files_by_n[n_key])
            new_task = utils_tsnkit.load_stream(random_task_file)
            env.load_new_task(new_task, random_task_file)

            obs, _ = env.reset()
            ep_reward, ep_hops, ep_success = 0.0, [], 0.0

            for _ in range(env.num_flows):
                rollouts['flow_tokens'].append(obs['flow_tokens'])
                rollouts['global_snapshot'].append(obs['global_snapshot'])
                rollouts['action_masks'].append(obs['action_mask'])

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
                    ep_success = info.get('group_success_rate', 0.0)
                    iter_success_rates.append(ep_success)
                    iter_avg_hops.append(sum(ep_hops)/len(ep_hops) if ep_hops else 0.0)
                    break

            iter_rewards.append(ep_reward)
            iter_n_values.append(n_key)
            ep_success = float(ep_success)
            mark = "100%" if ep_success >= 1.0 else f"{ep_success*100:4.1f}%"
            print(f"  ep{ep+1}/{episodes_per_iter} N={n_key:3d} [{mark}] R={ep_reward:6.2f}")

        # --- PPO 参数更新 ---
        pg_loss, v_loss, ent = agent.update(rollouts)

        avg_reward = np.mean(iter_rewards)
        avg_success = np.mean(iter_success_rates) * 100 if iter_success_rates else 0.0

        # 按 N 分档统计成功率（打印和 TensorBoard 共用）
        n_sr_map = defaultdict(list)
        for n_val, sr in zip(iter_n_values, iter_success_rates):
            n_sr_map[n_val].append(sr)
        n_sr_summary = " ".join(
            f"N{n}={np.mean(srs)*100:.0f}%" for n, srs in sorted(n_sr_map.items())
        )
        iter_time = time.time() - iter_start
        iters_done = iteration - start_iteration + 1
        elapsed = time.time() - train_start
        eta = elapsed / iters_done * (total_iterations - iteration)
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        eta_str     = str(timedelta(seconds=int(eta)))
        print(f"Iter {iteration:03d}/{total_iterations} | {n_sr_summary} | "
              f"avg:{avg_success:5.1f}% | R:{avg_reward:6.2f} | "
              f"PG:{pg_loss:6.3f} V:{v_loss:6.3f} H:{ent:5.3f} | "
              f"{iter_time:.0f}s/iter | elapsed {elapsed_str} | ETA {eta_str}")

        writer.add_scalar("Train/1_Success_Rate", avg_success, iteration)
        writer.add_scalar("Train/2_Avg_Reward", avg_reward, iteration)
        writer.add_scalar("Loss/1_Policy_Loss", pg_loss, iteration)
        writer.add_scalar("Loss/2_Value_Loss", v_loss, iteration)
        writer.add_scalar("Loss/3_Entropy", ent, iteration)
        for n_val, srs in sorted(n_sr_map.items()):
            writer.add_scalar(f"Train/SR_N{n_val}", np.mean(srs) * 100, iteration)

        # ==========================================
        # 🧪 阶段 B：验证模式 (Validation Loop) - 每 5 轮全量验证集
        # ==========================================
        if iteration % 5 == 0 and val_files:
            agent.network.eval()
            val_success_rates = []
            val_sr_by_n = defaultdict(list)   # 仅 v2 使用

            with torch.no_grad():
                for v_file in val_files:
                    v_task = utils_tsnkit.load_stream(v_file)
                    env.load_new_task(v_task, v_file)
                    obs, _ = env.reset()

                    for step in range(env.num_flows):
                        flow_tokens = torch.FloatTensor(obs['flow_tokens']).unsqueeze(0).to(agent.device)
                        g_global    = torch.FloatTensor(obs['global_snapshot']).unsqueeze(0).to(agent.device)
                        mask        = torch.BoolTensor(obs['action_mask']).unsqueeze(0).to(agent.device)

                        logits, _ = agent.network(flow_tokens, g_global)
                        logits = logits.masked_fill(~mask, -1e8)
                        action = torch.argmax(logits, dim=-1).item()

                        obs, _, terminated, _, info = env.step(action)
                        if terminated:
                            sr = float(info.get('group_success_rate', 0.0))
                            val_success_rates.append(sr)
                            # v2：按 N 子目录分档记录
                            if version == "v2" and val_files_by_n:
                                for n_val, files in val_files_by_n.items():
                                    if v_file in files:
                                        val_sr_by_n[n_val].append(sr)
                                        break
                            break

            avg_val_success = np.mean(val_success_rates) * 100
            writer.add_scalar('Eval/1_Val_SR_Overall', avg_val_success, iteration)

            # v2 分档 TensorBoard
            val_n_log = ""
            if version == "v2" and val_sr_by_n:
                for n_val in sorted(val_sr_by_n):
                    n_sr = np.mean(val_sr_by_n[n_val]) * 100
                    key_idx = sorted(val_sr_by_n.keys()).index(n_val) + 2
                    writer.add_scalar(f'Eval/{key_idx}_Val_SR_N{n_val}', n_sr, iteration)
                    val_n_log += f" N{n_val}:{n_sr:5.1f}%"

            n_val_total = len(val_files)
            val_log = f'  └─ Val({n_val_total}套) Overall:{avg_val_success:5.1f}%{val_n_log}'

            if avg_val_success > best_val_success:
                best_val_success = avg_val_success
                best_path = os.path.join(model_dir, 'phase2_ppo_best.pth')
                torch.save(agent.network.state_dict(), best_path)
                print(val_log + ' * 新最优')
            else:
                print(val_log)

        # 定期断点存档（覆盖写，只保留最新一份）
        if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
            resume_ckpt = os.path.join(model_dir, "resume.pth")
            torch.save({
                'iteration':            iteration,
                'network_state_dict':   agent.network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'best_val_success':     best_val_success,
                'run_dir':              run_dir,
            }, resume_ckpt)
            print(f"   => 💾 [断点存档] 第 {iteration} 轮 → {os.path.basename(resume_ckpt)}")

    writer.close()
    print("\n🎉 训练完美收官！")

if __name__ == "__main__":    
    import argparse
    from sca_drl.common.utils import load_config
    
    parser = argparse.ArgumentParser(description="Phase 2 PPO Training")
    parser.add_argument("--config",  default="configs/phase2.yaml", help="Path to config file")
    parser.add_argument("--resume",  default=None, help="断点续训: 传入 resume.pth 的路径")
    parser.add_argument("--version", default="v1", choices=["v1", "v2"],
                        help="v2: 使用 train_v2/val_v2 数据集与 v2 课程学习")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_path=args.resume, version=args.version)