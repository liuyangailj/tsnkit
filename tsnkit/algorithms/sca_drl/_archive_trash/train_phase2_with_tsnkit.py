"""
Phase 2 Training Script with TSNKit CSV Data

Trains PPO agent for scheduling using TSNKit CSV format.
"""

import sys
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (即 runners/ 的上一级)
project_root = os.path.dirname(current_dir)
# 将根目录加入 python 查找路径 (insert at 0 to avoid conflicts)
sys.path.insert(0, project_root)

from phase2_scheduling.environment import TSNSchedulingEnv
from phase2_scheduling.scheduler_agent import PPOAgent
import importlib.util

# Load tsnkit_reader module explicitly to avoid import conflicts
tsnkit_reader_path = os.path.join(project_root, 'common', 'tsnkit_reader.py')
tsnkit_reader_spec = importlib.util.spec_from_file_location("tsnkit_reader", tsnkit_reader_path)
tsnkit_reader = importlib.util.module_from_spec(tsnkit_reader_spec)
tsnkit_reader_spec.loader.exec_module(tsnkit_reader)


def get_default_config():
    """Return default configuration for Phase 2 training."""
    return {
        'ksp': {'k': 5},
        'env_params': {
            'time_limit': 1000,
            'bandwidth': 1000,
            'max_steps': 100
        },
        'normalization': {
            'max_size': 1500.0,
            'max_period': 10000000.0,  # 10ms in microseconds
            'max_deadline': 10000000.0
        },
        'agent_params': {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'k_epochs': 4
        },
        'train_phase2': {
            'num_episodes': 100,
            'max_steps_per_episode': 100,
            'update_timestep': 2000
        }
    }


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description='Train Phase 2 PPO with TSNKit CSV data')
    parser.add_argument('--task', type=str, required=True, help='Path to TSNKit task CSV file')
    parser.add_argument('--topo', type=str, required=True, help='Path to TSNKit topology CSV file')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO clipping parameter')
    parser.add_argument('--k_epochs', type=int, default=4, help='PPO update epochs')
    parser.add_argument('--update_timestep', type=int, default=2000, help='Update frequency')
    parser.add_argument('--output', type=str, default=None, help='Model save path (default: models/phase2_ppo.pth)')
    parser.add_argument('--k', type=int, default=5, help='Number of K-shortest paths')

    args = parser.parse_args()

    print(f"Loading TSNKit data from {args.task} and {args.topo}...")

    # Load TSNKit CSV data
    flows, G = tsnkit_reader.read_tsnkit_data(args.task, args.topo, k=args.k)

    print(f"Loaded {len(flows)} flows on {len(G.nodes)} nodes")

    # Prepare config
    config = get_default_config()
    config['ksp']['k'] = args.k
    config['agent_params']['learning_rate'] = args.lr
    config['agent_params']['gamma'] = args.gamma
    config['agent_params']['eps_clip'] = args.eps_clip
    config['agent_params']['k_epochs'] = args.k_epochs
    config['train_phase2']['num_episodes'] = args.episodes
    config['train_phase2']['max_steps_per_episode'] = args.max_steps
    config['train_phase2']['update_timestep'] = args.update_timestep

    # Initialize environment
    print("Initializing environment...")
    device = get_device()
    print(f"Using device: {device}")

    env = TSNSchedulingEnv(flows, G, config)

    # Initialize agent
    print("Initializing PPO agent...")

    # Parse state dim
    state_dim = env.observation_space.shape[0] if hasattr(env, 'observation_space') else 64

    # Parse action dim (num_flows, num_paths)
    num_flows = len(flows)
    num_paths = args.k

    agent = PPOAgent(
        state_dim, num_flows, num_paths,
        lr=args.lr,
        gamma=args.gamma,
        eps_clip=args.eps_clip,
        k_epochs=args.k_epochs,
        device=str(device)
    )

    # Training loop
    print(f"Starting training for {args.episodes} episodes...")

    timestep = 0
    rewards_history = []
    update_timestep = args.update_timestep

    for episode in tqdm(range(args.episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0

        for t in range(args.max_steps):
            timestep += 1

            # Mask Generation
            mask = None
            if hasattr(env, 'get_mask'):
                mask = env.get_mask()

            if mask is None:
                # Fallback: all 1 mask (all flows available)
                mask = np.ones(num_flows, dtype=np.float32)

            # Select action
            action = agent.select_action(state, mask)

            # Environment step
            next_state, reward, done, info = env.step(action)

            # PPO save rewards and terminal states
            if hasattr(agent, 'buffer'):
                if isinstance(agent.buffer, dict):
                    agent.buffer['rewards'].append(reward)
                    agent.buffer['is_terminals'].append(done)
                else:
                    agent.buffer.rewards.append(reward)
                    agent.buffer.is_terminals.append(done)

            state = next_state
            episode_reward += reward

            # PPO update
            if timestep % update_timestep == 0:
                agent.update()

            if done:
                break

        rewards_history.append(episode_reward)

        # Periodic print and save
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            tqdm.write(f"Episode {episode+1} | Avg Reward: {avg_reward:.4f}")

    # Determine save path
    if args.output:
        save_path = args.output
    else:
        # Default save to models/ directory
        models_dir = os.path.join(project_root, 'models')
        os.makedirs(models_dir, exist_ok=True)
        save_path = os.path.join(models_dir, 'phase2_ppo.pth')

    # Save model
    agent.save(save_path)

    print(f"\n[SUCCESS] Phase 2 Training Complete! Model saved to: {save_path}")
    print(f"   - Episodes: {args.episodes}")
    print(f"   - Avg Reward (last 10): {np.mean(rewards_history[-10:]):.4f}")
    print(f"   - Flows: {len(flows)}")
    print(f"   - K-Shortest Paths: {args.k}")


if __name__ == "__main__":
    main()
