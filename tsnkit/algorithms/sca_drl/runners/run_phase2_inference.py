import sys
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 项目路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))

# 2. 导入自定义模块
from phase2_scheduling.environment import TSNSchedulingEnv
from phase2_scheduling.scheduler_agent import PPOAgent
from common.topology_gen import TopologyGenerator
from common.flow_gen import FlowGenerator

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_phase2_inference():
    print("="*50)
    print("Starting Phase 2: Inference & Evaluation")
    print("="*50)

    # 3. 加载配置
    config_path = os.path.join(project_root, 'configs', 'base_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # --- 配置完整性补全 (与 Training 保持一致) ---
    if 'traffic' not in config:
        config['traffic'] = {'num_flows': 5, 'period_list': [1000], 'size_range': [64, 1518]}
    if 'ksp' not in config: 
        config['ksp'] = {'k': 3}
    if 'env_params' in config and 'ksp' not in config['env_params']:
        config['env_params']['ksp'] = {'k': 3}
    if 'agent_params' not in config:
        config['agent_params'] = {
            'hidden_dim': 64, 'learning_rate': 0.001, 'gamma': 0.99, 'eps_clip': 0.2, 'k_epochs': 4
        }
    # -------------------------------------------

    device = get_device()
    print(f"Using device: {device}")

    # 4. 准备测试环境 (生成新的测试数据)
    print("\n[Step 1] Setting up Test Environment...")
    topo_gen = TopologyGenerator(num_nodes=config['topology']['num_nodes'])
    graph = topo_gen.generate_graph()
    
    flow_gen = FlowGenerator(config, graph)
    flows = flow_gen.generate_flows()
    print(f"  Generated {len(flows)} flows for testing.")
    
    # 初始化环境 (注意参数顺序: flows, graph)
    try:
        env = TSNSchedulingEnv(flows, graph, config['env_params'])
    except:
        env = TSNSchedulingEnv(graph, flows, config['env_params'])

    # 5. 初始化 Agent 并加载模型
    print("\n[Step 2] Loading Trained Model...")
    
    # 解析维度
    state_dim = env.observation_space.shape[0] if hasattr(env, 'observation_space') else 64
    num_flows = 5
    num_paths = 3
    if hasattr(env, 'action_space') and hasattr(env.action_space, 'nvec'):
        dims = env.action_space.nvec
        if len(dims) >= 2:
            num_flows = int(dims[0])
            num_paths = int(dims[1])
    
    agent_kwargs = {
        'lr': config['agent_params'].get('learning_rate', 0.001),
        'gamma': config['agent_params'].get('gamma', 0.99),
        'eps_clip': config['agent_params'].get('eps_clip', 0.2),
        'k_epochs': config['agent_params'].get('k_epochs', 4)
    }
    
    # 初始化
    agent = PPOAgent(state_dim, num_flows, num_paths, **agent_kwargs)
    
    # 加载权重
    checkpoint_path = os.path.join(project_root, 'results', 'checkpoints', 'ppo_scheduler_final.pth')
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"  [Success] Model loaded from {checkpoint_path}")
    else:
        print(f"  [Warning] Checkpoint not found at {checkpoint_path}. Using random weights!")

    # 切换到评估模式 (如果 Agent 实现了 eval 方法)
    if hasattr(agent, 'policy'):
        agent.policy.eval()

    # 6. 推理循环
    print("\n[Step 3] Running Inference Loop...")
    
    num_test_episodes = 5 # 测试 5 个 Episode
    all_rewards = []

    for ep in range(num_test_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < config['env_params']['max_steps']:
            # 生成 Mask
            mask = None
            if hasattr(env, 'get_mask'): mask = env.get_mask()
            if mask is None: mask = np.ones(num_flows, dtype=np.float32)
            
            # 选择动作 (Inference 时通常不需要梯度)
            with torch.no_grad():
                action = agent.select_action(state, mask)
            
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # 这里可以打印每一步的详细调度结果，例如：
            # print(f"    Step {steps}: Flow {action[0]} -> Path {action[1]} (Reward: {reward:.2f})")

        all_rewards.append(total_reward)
        print(f"  Episode {ep+1}: Total Reward = {total_reward:.4f}, Steps = {steps}")

    print("\n" + "="*50)
    print(f"Inference Summary (over {num_test_episodes} episodes):")
    print(f"  Mean Reward: {np.mean(all_rewards):.4f}")
    print(f"  Std Reward:  {np.std(all_rewards):.4f}")
    print("="*50)

if __name__ == "__main__":
    run_phase2_inference()