import sys
import os
import torch
import numpy as np
import pickle
import yaml
from tqdm import tqdm

# 为加载模块，进行路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from phase2_scheduling.environment import TSNSchedulingEnv
from phase2_scheduling.scheduler_agent import PPOAgent
from common.config_utils import load_config

def main():
    print("="*60)
    print("Starting SCA-DRL Joint Experiment (Partitioning + Scheduling)")
    print("="*60)

    # ------------------------------------------------------------
    # 1. 配置加载
    # ------------------------------------------------------------
    base_config_path = os.path.join(project_root, 'configs', 'base_config.yaml')
    phase2_config_path = os.path.join(project_root, 'configs', 'phase2_config.yaml')
    
    # 检查文件是否存在
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config not found at {base_config_path}")
    if not os.path.exists(phase2_config_path):
        raise FileNotFoundError(f"Phase 2 config not found at {phase2_config_path}")
    
    # 加载并合并配置：Base <- Phase2
    # 假设 load_config 接受 *args 并按顺序合并
    config = load_config(base_config_path, phase2_config_path)       
    
    agent_params = config.get('agent_params', {})
    env_params = config.get('env_params', {})
    # train_params 即使为空也先获取，用于后续扩展
    train_params = config.get('train_phase2', {})

    print(f"[Config] Agent Params: {agent_params}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Running on device: {device}")
    
    # ---------------------------------------------------------
    # 2. 数据加载
    # ---------------------------------------------------------
       
    # 2. 读取 Phase 1 的分组数据
    data_path = os.path.join(project_root, "data", "processed", "partition_result.pkl")
    if not os.path.exists(data_path):
        print(f"[Error] Partition data not found at {data_path}")
        print("Please run 'runners/run_phase1_inference.py' first!")
        return
    
    print(f"[Info] Loading partition data from: {data_path}")
    with open(data_path, 'rb') as f:
        partition_data = pickle.load(f)
    
    graph = partition_data['original_graph']
    groups = partition_data['groups'] # Dict {0: [flows...], 1: [flows...]}
    
    print(f"[Info] Loaded {len(groups)} groups from Phase 1.")

    # ---------------------------------------------------------
    # 3. 初始化全局资源状态
    # ---------------------------------------------------------
    
    # 初始为空 (None)，随着 Group 的调度逐渐填满
    global_resource_snapshot = None 
    
    # 初始化 PPO Agent
    # 注意：为了简单起见，我们所有 Group 共用一个 Agent 进行训练/微调
    # 或者你可以每组都 Reset Agent，但这通常不符合 Transfer Learning 的思想
    # 这里我们要根据第一组的数据来初始化 Agent 维度
    first_group_flows = groups[0]
    dummy_env = TSNSchedulingEnv(first_group_flows, graph, env_params)
    state_dim = dummy_env.observation_space.shape[0]
    
    # 从配置或环境获取 Action Dim 动作空间维度？
    num_paths = 3
    if hasattr(dummy_env.action_space, 'nvec'):
        num_paths = int(dummy_env.action_space.nvec[1])
    
    # 因为每个 Group 的流数量可能不同，所以 Agent 的 num_flows 需要动态处理
    # 或者我们在设计 Agent 时就让它支持变长输入 (GNN/Transformer)，或者取一个 max_flows
    # 现在的 PPOAgent 是固定维度的 (Linear层)，这在变长 Group 下会报错。
    # *** 这是一个工程上的难点 ***
    # 临时方案：假设所有 Group 的流数量不超过 config 中的设置，或者重新初始化 Agent
    # 为了跑通 SCA-DRL 流程，我们在这个 Demo 中采取：**为每个 Group 重新初始化一个 Agent**
    # (理想情况下应该是同一个 Agent 学会泛化，但那需要更复杂的 Network)
    
    # ---------------------------------------------------------
    # 4. 迭代调度 (Iterative Scheduling Loop)
    # ---------------------------------------------------------    
    
    # 开始 SCA-DRL 循环 (Stage-wise Scheduling)
    total_success_flows = 0
    total_flows = 0
    
    for group_id in sorted(groups.keys()):
        group_flows = groups[group_id]
        print(f"\n>>> Processing Group {group_id} ({len(group_flows)} flows) <<<")
        
        # A. 创建环境，并传入上一轮的资源快照
        env = TSNSchedulingEnv(
            group_flows, 
            graph, 
            env_params, 
            initial_resource_state=global_resource_snapshot # 核心：资源继承
        )
        
        # [新增] 动态获取当前环境的状态维度
        current_obs_dim = env.observation_space.shape[0]
        
        # B. 初始化/重置 Agent (适配当前 Group 的流数量)
        current_num_flows = len(group_flows)
        agent = PPOAgent(
            obs_dim=current_obs_dim, # [修正] 使用动态获取的 current_obs_dim
            # [修正] 这里原来是 state_dim=state_dim，改为 obs_dim=state_dim
            # obs 是 observation（观测值）的缩写，
            # 在强化学习（RL）中，obs_dim 指的是智能体（Agent）
            # 所能感知到的环境状态向量的维度。            
            num_flows=current_num_flows, 
            num_paths=num_paths,
            lr=agent_params.get('learning_rate', 1e-3), 
            gamma=agent_params.get('gamma',0.99), 
            eps_clip=agent_params.get('eps_clip',0.2), 
            k_epochs=agent_params.get('k_epochs',4)
        )
        
        # C. 针对当前 Group 进行训练 (Short Training)
        # 论文里可能是直接 Inference，也可能是微调。我们这里跑 50 个 Episode 尝试调度
        best_reward = -float('inf')
        best_snapshot = None
        best_success_count = 0 # [新增] 记录本组最佳成功数
          
        print("   Training/Scheduling...")
        train_episodes = train_params.get('train_episodes', 50) # 若无参数输入，则默认50
        print(f"当前使用的训练轮数: {train_episodes}")
        
        # [修改] 使用 tqdm 显示进度条
        # desc: 进度条左边的文字
        # leave=False: 跑完一组后进度条消失，保持界面清爽，或者设为True保留
        training_loop = tqdm(range(train_episodes), desc=f"   Scheduling Group {group_id}", leave=True)
                
        for ep in training_loop: # 快速训练
            state = env.reset()
            ep_reward = 0
            
            # Mask
            mask = np.ones(current_num_flows, dtype=np.float32)
            
            for t in range(100):
                action = agent.select_action(state, mask)
                next_state, reward, done, info = env.step(action)
                
                # [修正] 兼容 buffer 是字典的情况 (Run Phase 2 Training 中遇到的问题)
                if hasattr(agent, 'buffer'):
                    if isinstance(agent.buffer, dict):
                        agent.buffer['rewards'].append(reward)
                        agent.buffer['is_terminals'].append(done)
                    else:
                        agent.buffer.rewards.append(reward)
                        agent.buffer.is_terminals.append(done)
                
                state = next_state
                ep_reward += reward
                if done: break
            
            agent.update()
            
            # [新增] 统计当前 Episode 成功调度的流数量
            # 依据 environment.py，flow_states[i]['scheduled'] 是布尔值
            current_success = sum(1 for f in env.flow_states.values() if f['scheduled'])
            
            # 记录表现最好的一次资源状态
            if ep_reward > best_reward:
                best_reward = ep_reward
                # 保存这一轮调度后的资源占用情况
                best_snapshot = env.slot_manager.get_state_snapshot()
                # 统计成功调度的流 (假设 info 里有 success 信息，或者简单用 reward 判断)
                # 这里暂且略过精确统计，假设 reward 高就是好
                best_success_count = current_success # 记录对应的成功数
                
            # [新增] 在进度条尾部实时显示 Reward 和 Success
            training_loop.set_postfix({
                'Reward': f"{ep_reward:.2f}", 
                'Succ': f"{current_success}/{current_num_flows}"
            })
        
        print(f" > Group {group_id} Done. Best Reward: {best_reward:.2f} | Success: {best_success_count}/{current_num_flows}")
        
        # D. 更新全局资源快照
        # 如果这一组调度成功了（或者找到了最佳解），就把它的资源占用锁定，传给下一组
        if best_snapshot is not None:
            global_resource_snapshot = best_snapshot
            # print("   [System] Global resources updated (Inherited).")
        
        # [新增] 更新全局资源快照
        total_success_flows += best_success_count
        total_flows += len(group_flows)

    # [修改] 最终统计信息
    print("\n" + "="*60)
    print("SCA-DRL Experiment Completed.")
    print(f"Processed Total Flows: {total_flows}")
    print(f"Total Successfully Scheduled: {total_success_flows}")
    
    if total_flows > 0:
        success_rate = (total_success_flows / total_flows) * 100
        print(f"Global Schedulability (Success Rate): {success_rate:.2f}%")
    else:
        print("Global Schedulability (Success Rate): 0.00%")
    print("="*60)
    
if __name__ == "__main__":
    main()