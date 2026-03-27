"""SCA-DRL 公共工具：种子固定、配置加载、路径解析。"""

import os
import random
import yaml
import numpy as np
import torch


# 路径常量：基于本文件位置自动推导，不再需要各脚本重复计算
_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))       # common/
PROJECT_ROOT = os.path.dirname(_COMMON_DIR)                    # sca_drl/
ALGORITHM_DIR = os.path.dirname(os.path.dirname(PROJECT_ROOT)) # tsnkit/tsnkit/algorithms/
TSNKIT_ROOT = os.path.dirname(ALGORITHM_DIR)                   # tsnkit/tsnkit/
REPO_ROOT = os.path.dirname(TSNKIT_ROOT)                       # tsnkit/ (仓库根目录)


def set_seed(seed: int = 42):
    """固定全局随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件。

    Args:
        config_path: 配置文件路径，支持绝对路径或相对于 PROJECT_ROOT 的路径

    Returns:
        dict: 解析后的配置字典
    """
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str, base: str = "project") -> str:
    """将相对路径解析为绝对路径。

    Args:
        relative_path: 相对路径字符串
        base: 基准目录，可选 "project"(sca_drl/) 或 "repo"(仓库根目录)

    Returns:
        str: 绝对路径
    """
    if os.path.isabs(relative_path):
        return relative_path
    base_dir = PROJECT_ROOT if base == "project" else REPO_ROOT
    return os.path.join(base_dir, relative_path)


def get_device() -> torch.device:
    """自动选择可用的计算设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    """确保目录存在，不存在则递归创建。"""
    os.makedirs(path, exist_ok=True)