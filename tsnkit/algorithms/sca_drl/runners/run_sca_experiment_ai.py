import sys
import os
import torch
import numpy as np
import pickle
import yaml

# Add project root to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))

from phase2_scheduling.environment import TSNSchedulingEnv
from phase2_scheduling.scheduler_agent import PPOAgent
from common.config_utils import load_config

def main():
    print("="*60)
    print("Starting SCA-DRL Joint Experiment (Partitioning + Scheduling)")
    print("="*60)

    # ---------------------------------------------------------
    # 1. 配置加载
    # ---------------------------------------------------------
    config_path = os.path.join(project_root, 'configs', 'phase2_config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config = load_config(config_path)
    
    agent_params = config.get('agent_params', {})
    env_params = config.get('env_params', {})
    train_params = config.get('train_params', {})

    print(f"[Config] Agent Params: {agent_params}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Running on device: {device}")

    # ---------------------------------------------------------
    # 2. 数据加载
    # ---------------------------------------------------------
    partition_result_path = os.path.join(project_root, 'data', 'processed', 'partition_result.pkl')
    if not os.path.exists(partition_result_path):
        print(f"[Error] Partition result not found at {partition_result_path}.")
        return

    print(f"[Info] Loading partition data from: {partition_result_path}")
    with open(partition_result_path, 'rb') as f:
        partition_data = pickle.load(f)

    original_graph = None
    if isinstance(partition_data, dict) and 'original_graph' in partition_data:
        original_graph = partition_data['original_graph']
    
    flow_groups = []
    if isinstance(partition_data, dict) and 'groups' in partition_data:
        raw_groups = partition_data['groups']
        if isinstance(raw_groups, dict):
            flow_groups = [raw_groups[k] for k in sorted(raw_groups.keys())]
        elif isinstance(raw_groups, list):
            flow_groups = raw_groups
    elif isinstance(partition_data, list):
        flow_groups = partition_data
    
    print(f"[Info] Successfully loaded {len(flow_groups)} groups from Phase 1.")

    # ---------------------------------------------------------
    # 3. 初始化全局资源状态
    # ---------------------------------------------------------
    global_resource_state = None 
    total_flows = 0
    total_scheduled = 0

    # ---------------------------------------------------------
    # 4. 迭代调度 (Iterative Scheduling Loop)
    # ---------------------------------------------------------
    for group_id, flow_group in enumerate(flow_groups):
        num_flows = len(flow_group)
        print(f"\n>>> Processing Group {group_id} ({num_flows} flows) <<<")
        
        # 4.1 初始化环境 (兼容性处理)
        try:
            # 尝试传入 device (新版接口)
            env = TSNSchedulingEnv(
                flows=flow_group, 
                topology=original_graph, 
                config=env_params, 
                initial_resource_state=global_resource_state,
                device=device 
            )
        except TypeError:
            # 如果 Environment 不支持 device 参数 (旧版接口)，则移除该参数
            # print("[Warning] Environment does not support 'device' param. Using default.")
            env = TSNSchedulingEnv(
                flows=flow_group, 
                topology=original_graph, 
                config=env_params, 
                initial_resource_state=global_resource_state
            )

        # 4.2 重置环境 (兼容 Gym vs Gymnasium)
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            state, _ = reset_result # Gymnasium Style: (obs, info)
        else:
            state = reset_result    # Old Gym Style: obs only
        
        # 确定维度
        if isinstance(state, np.ndarray):
            obs_dim = state.size
        else:
            obs_dim = len(state)
            
        k_paths = env_params.get('k_paths', 3)

        # 4.3 初始化 Agent
        agent = PPOAgent(
            obs_dim=obs_dim,
            num_flows=num_flows,
            num_paths=k_paths,
            lr=agent_params.get('learning_rate', 1e-3),
            gamma=agent_params.get('gamma', 0.99),
            eps_clip=agent_params.get('eps_clip', 0.2),
            k_epochs=agent_params.get('k_epochs', 4),
            device=device.type
        )

        # 4.4 针对当前 Group 进行短时训练
        train_episodes = train_params.get('group_train_episodes', 50)
        
        best_reward = -float('inf')
        best_snapshot = None
        success_count = 0
        
        # print(f"   Training for {train_episodes} episodes...")

        for ep in range(train_episodes):
            # Reset again for episode start
            reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                state, _ = reset_result
            else:
                state = reset_result
                
            ep_reward = 0
            
            # Mask 兼容性处理
            if hasattr(env, 'get_mask'):
                mask = env.get_mask()
            elif hasattr(env, 'get_action_mask'):
                mask = env.get_action_mask()
            else:
                mask = None
            
            for t in range(num_flows + 5): 
                action = agent.select_action(state, action_mask=mask)
                
                # Step 兼容性处理
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, done, truncated, info = step_result
                else:
                    next_state, reward, done, info = step_result # Old Gym
                
                # 存储数据
                if hasattr(agent, 'buffer'):
                    if isinstance(agent.buffer, dict):
                        agent.buffer['rewards'].append(reward)
                        agent.buffer['is_terminals'].append(done)
                    else:
                        agent.buffer.rewards.append(reward)
                        agent.buffer.is_terminals.append(done)
                
                state = next_state
                ep_reward += reward
                
                # Update Mask
                if hasattr(env, 'get_mask'):
                    mask = env.get_mask()
                elif hasattr(env, 'get_action_mask'):
                    mask = env.get_action_mask()
                
                if done:
                    break
            
            agent.update()
            
            # 记录最佳结果
            # 尝试从 info 获取 success_count，如果不存在则使用默认值
            current_success = info.get('success_count', 0) if isinstance(info, dict) else 0
            
            if ep_reward > best_reward:
                best_reward = ep_reward
                success_count = current_success
                
                # Snapshot 获取
                if hasattr(env, 'slot_manager'):
                    best_snapshot = env.slot_manager.get_state_snapshot()
                elif hasattr(env, 'get_resource_state'):
                    best_snapshot = env.get_resource_state()
        
        print(f"   Group {group_id} Done. Best Reward: {best_reward:.2f}, Success: {success_count}/{num_flows}")
        
        # 4.5 更新全局资源
        if best_snapshot is not None:
            global_resource_state = best_snapshot
            print("   [System] Global resources updated (Inherited).")

        total_scheduled += success_count
        total_flows += num_flows

    # ---------------------------------------------------------
    # 5. 实验总结
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("SCA-DRL Experiment Completed.")
    print(f"Total Flows Processed: {total_flows}")
    print(f"Total Successfully Scheduled: {total_scheduled}")
    if total_flows > 0:
        print(f"Global Schedulability (Success Rate): {total_scheduled/total_flows*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()