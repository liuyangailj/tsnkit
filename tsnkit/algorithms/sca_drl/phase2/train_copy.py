"""
Phase 2 训练入口，配置驱动，连接 Phase 1 Embedding 和分组。
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch

from sca_drl.common.utils import (
    set_seed, load_config, resolve_path, get_device, ensure_dir
)
from sca_drl.phase2.environment import TSNSchedulingEnv
from sca_drl.phase2.agent import PPOAgent


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


def main(config):
    train_cfg = config.get("training", {})
    agent_cfg = config.get("agent", {})

    device = get_device()
    phase1_emb = load_phase1_embeddings(config)

    env = TSNSchedulingEnv(config, phase1_embeddings=phase1_emb)
    print(f"Environment initialized: num_streams={len(env.streams)}, feature_dim={env.d_feature}")

    agent = PPOAgent(
        d_feature=env.d_feature,
        k_max=env.k_max,
        config=agent_cfg,
        device=device
    )

    num_episodes = train_cfg.get("num_episodes", 5000)
    log_interval = train_cfg.get("log_interval", 50)
    save_interval = train_cfg.get("save_interval", 500)

    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(obs, env.scheduled)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, log_prob, done)
            ep_reward += reward

        loss = agent.update()
        episode_rewards.append(ep_reward)

        if episode % log_interval == 0:
            avg_reward = sum(episode_rewards[-log_interval:]) / log_interval
            scheduled_ratio = env.scheduled.sum() / len(env.streams)
            print(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | "
                  f"Scheduled: {scheduled_ratio:.2%} | Loss: {loss:.4f}")

        if episode % save_interval == 0:
            ckpt_path = resolve_path(train_cfg.get("checkpoint_path", "models/phase2_ppo.pth"))
            ensure_dir(os.path.dirname(ckpt_path))
            agent.save(ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # 训练结束保存最终模型
    ckpt_path = resolve_path(train_cfg.get("checkpoint_path", "models/phase2_ppo.pth"))
    ensure_dir(os.path.dirname(ckpt_path))
    agent.save(ckpt_path)
    print(f"Training completed. Model saved to: {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 PPO Training")
    parser.add_argument("--config", default="configs/phase2.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    main(cfg)