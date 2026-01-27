import sys
import os
import yaml
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

# 项目路径设置
# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (即 runners/ 的上一级)
project_root = os.path.dirname(current_dir)
# 将根目录加入 python 查找路径
sys.path.append(project_root)

# 导入自定义模块
from phase2_scheduling.environment import TSNSchedulingEnv
from phase2_scheduling.scheduler_agent import PPOAgent
from common.topology_gen import TopologyGenerator
from common.flow_gen import FlowGenerator
# 导入配置加载工具
from common.config_utils import load_config

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_phase2_training():
    print("="*50)
    print("Starting Phase 2: Scheduling Training (PPO)")
    print("="*50)
    
    # 1. 加载配置
    # 加载 Base，然后用 Phase 2 覆盖 Base
    config = load_config("configs/base_config.yaml", "configs/phase2_config.yaml")

    print(config['traffic']['num_flows'])  # 输出: 100 (来自 phase2_config.yaml)
    print(config['topology']['num_nodes']) # 输出: 20 (来自 base_config.yaml)
    
    # --- 配置完整性检查与补全 ---
    if 'traffic' not in config:
        print("[Warn] 'traffic' config missing. Using default fallback.")
        config['traffic'] = {'num_flows': 5, 'period_list': [1000], 'size_range': [64, 1518]}
    
    if 'ksp' not in config: 
        config['ksp'] = {'k': 3}
        # 同时也补全 env_params 里的 ksp
        if 'env_params' in config and 'ksp' not in config['env_params']:
            config['env_params']['ksp'] = {'k': 3}
    
    if 'agent_params' not in config:
        print("[Warn] 'agent_params' config missing. Using default fallback.")
        config['agent_params'] = {
            'hidden_dim': 64,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'k_epochs': 4
        }
    # ---------------------------

    device = get_device()
    print(f"Using device: {device}")

    # 2. 数据准备 (Topology & Flows)
    print("\n[Step 1] Initializing Environment Data...")
    
    # 初始化拓扑
    topo_gen = TopologyGenerator(num_nodes=config['topology']['num_nodes'])
    graph = topo_gen.generate_graph()
    
    # 初始化流量
    flow_gen = FlowGenerator(config, graph)
    flows = flow_gen.generate_flows()
    print(f"  Generated {len(flows)} flows on {len(graph.nodes)} nodes.")
    
    # 3. 初始化环境
    # 顺序: (flows, graph, config)
    print("\n[Step 2] Initializing TSNSchedulingEnv...")
    try:
        env = TSNSchedulingEnv(flows, graph, config['env_params'])
    except Exception as e:
        print(f"  [Error] Init failed with (flows, graph). Trying (graph, flows)...")
        env = TSNSchedulingEnv(graph, flows, config['env_params'])
    
    # 4. 初始化 Agent
    print("\n[Step 3] Initializing PPO Agent...")
    
    # 解析 State Dim
    state_dim = env.observation_space.shape[0] if hasattr(env, 'observation_space') else 64
    
    # 解析 Action Dim (num_flows, num_paths)
    num_flows = 5 # Default
    num_paths = 3 # Default
    
    if hasattr(env, 'action_space') and hasattr(env.action_space, 'nvec'):
        dims = env.action_space.nvec
        if len(dims) >= 2:
            num_flows = int(dims[0])
            num_paths = int(dims[1])
            print(f"  Detected MultiDiscrete Action Space: Flows={num_flows}, Paths={num_paths}")
    
    # 提取超参数
    agent_kwargs = {
        'lr': config['agent_params'].get('learning_rate', 0.001),
        'gamma': config['agent_params'].get('gamma', 0.99),
        'eps_clip': config['agent_params'].get('eps_clip', 0.2),
        'k_epochs': config['agent_params'].get('k_epochs', 4)
    }
    
    # 初始化 Agent
    agent = PPOAgent(state_dim, num_flows, num_paths, **agent_kwargs)
    
    # 5. 训练循环
    print("\n[Step 4] Starting Training Loop...")
    
    train_cfg = config.get('train_phase2', {})
    num_episodes = train_cfg.get('num_episodes', 1000)
    max_steps = train_cfg.get('max_steps_per_episode', 100)
    update_timestep = train_cfg.get('update_timestep', 2000) # 每多少步更新一次网络
    
    timestep = 0
    rewards_history = []
    
    # 创建保存目录
    checkpoint_dir = os.path.join(project_root, 'results', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        
        for t in range(max_steps):
            timestep += 1
            
            # Mask Generation
            mask = None
            if hasattr(env, 'get_mask'):
                mask = env.get_mask()
            
            if mask is None:
                # Fallback: 全 1 Mask (假设所有流都可选)
                mask = np.ones(num_flows, dtype=np.float32)
            
            # 选择动作
            action = agent.select_action(state, mask)
            
            # 环境步进
            next_state, reward, done, info = env.step(action)
            
            # PPO 保存奖励和终止状态
            # [修正] 兼容 buffer 是字典的情况
            if hasattr(agent, 'buffer'):
                if isinstance(agent.buffer, dict):
                    # 如果 buffer 是字典，使用 ['key'] 访问
                    agent.buffer['rewards'].append(reward)
                    agent.buffer['is_terminals'].append(done)
                else:
                    # 如果 buffer 是对象，使用 .key 访问
                    agent.buffer.rewards.append(reward)
                    agent.buffer.is_terminals.append(done)
            
            state = next_state
            episode_reward += reward
            
            # PPO 更新
            if timestep % update_timestep == 0:
                agent.update()
            
            if done:
                break
        
        rewards_history.append(episode_reward)
        
        # 定期打印和保存
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            tqdm.write(f"Episode {episode+1} | Avg Reward: {avg_reward:.4f}")
            
            # 保存模型
            save_path = os.path.join(checkpoint_dir, 'ppo_scheduler_latest.pth')
            agent.save(save_path)

    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, 'ppo_scheduler_final.pth')
    agent.save(final_path)
    print(f"\nTraining completed. Model saved to {final_path}")

if __name__ == "__main__":
    run_phase2_training()