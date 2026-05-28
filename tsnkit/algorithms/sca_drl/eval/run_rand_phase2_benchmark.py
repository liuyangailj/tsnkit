"""
消融实验1：随机分组 + Phase2 推理

流程
----
1. 遍历 benchmark_v4/N*/*_task.csv
2. 每个实例生成 *_rand_c{n}_group.csv（随机等份分组，种子固定）
3. 复制 *_k5_d32_emb.pt  → *_rand_emb.pt  （Phase1 特征保持不变，只改分组）
4. 复制 *_k5_d32_paths.pkl → *_rand_paths.pkl（KSP 路径保持不变）
5. 以 emb_model_tag='rand' 运行 Phase2 benchmark 推理

消融目的：验证 Phase1 GNN 分组质量的贡献（分组策略 random vs GNN-guided）

用法
----
cd tsnkit/algorithms/sca_drl
python eval/run_rand_phase2_benchmark.py --ckpt models/phase2_ppo_best.pth
python eval/run_rand_phase2_benchmark.py --ckpt models/phase2_ppo_best.pth --force
"""

import sys
import os

_SCA_DRL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PHASE2_DIR   = os.path.join(_SCA_DRL_ROOT, "phase2")
_TSNKIT_ROOT  = os.path.abspath(os.path.join(_SCA_DRL_ROOT, "..", ".."))
sys.path.insert(0, _TSNKIT_ROOT)
sys.path.insert(0, _SCA_DRL_ROOT)
sys.path.insert(0, _PHASE2_DIR)

import glob
import csv
import time
import shutil
import random
import copy
import argparse
import numpy as np
import torch

import pandas as pd
from tsnkit import core as tsnkit_core
from environment import TSNEnv
from agent import PPOAgent
from infer import load_agent, _run_n_dirs, _read_n_groups
from sca_drl.common.utils import load_config, resolve_path, set_seed

RAND_TAG   = "rand"
SRC_TAG    = "k5_d32"
N_CLUSTERS = 4
SOLVER_TAG = "rand_Phase2"


# ─────────────────────────────────────────────────────────────────────────────
# 随机分组文件预生成
# ─────────────────────────────────────────────────────────────────────────────

def _gen_rand_group_csv(task_path: str, n_clusters: int, rng: random.Random) -> None:
    """从 Phase1 group.csv 读取流 ID，随机等份分组，写入 *_rand_c{n}_group.csv。"""
    src_grp = task_path.replace(".csv", f"_{SRC_TAG}_c{n_clusters}_group.csv")
    if not os.path.exists(src_grp):
        raise FileNotFoundError(f"Phase1 分组文件缺失（请先运行 batch_infer benchmark_v4）: {src_grp}")

    df = pd.read_csv(src_grp)
    stream_ids = df["stream_id"].tolist()

    ids = list(stream_ids)
    rng.shuffle(ids)
    size = len(ids) // n_clusters

    dst_grp = task_path.replace(".csv", f"_{RAND_TAG}_c{n_clusters}_group.csv")
    with open(dst_grp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stream_id", "group_id"])
        for gid in range(n_clusters):
            start = gid * size
            end   = start + size if gid < n_clusters - 1 else len(ids)
            for sid in ids[start:end]:
                writer.writerow([sid, gid])


def _prepare_rand_files(task_files: list, n_clusters: int,
                        seed: int = 42, force: bool = False) -> None:
    """为 task_files 中每个实例生成随机分组文件，并复制 emb/paths 文件。"""
    rng      = random.Random(seed)
    done     = 0
    skipped  = 0
    failed   = 0
    t0       = time.perf_counter()

    for i, task_path in enumerate(task_files, 1):
        dst_grp = task_path.replace(".csv", f"_{RAND_TAG}_c{n_clusters}_group.csv")
        dst_emb = task_path.replace(".csv", f"_{RAND_TAG}_emb.pt")
        dst_pkl = task_path.replace(".csv", f"_{RAND_TAG}_paths.pkl")

        if not force and os.path.exists(dst_grp) and os.path.exists(dst_emb) and os.path.exists(dst_pkl):
            skipped += 1
            # 保持 rng 步进一致：即使跳过，也消耗一次 shuffle 等价的随机状态
            rng.random()
            continue

        try:
            _gen_rand_group_csv(task_path, n_clusters, rng)

            src_emb = task_path.replace(".csv", f"_{SRC_TAG}_emb.pt")
            src_pkl = task_path.replace(".csv", f"_{SRC_TAG}_paths.pkl")
            if os.path.exists(src_emb):
                shutil.copy2(src_emb, dst_emb)
            if os.path.exists(src_pkl):
                shutil.copy2(src_pkl, dst_pkl)

            done += 1
        except Exception as e:
            print(f"\n   ⚠️ 预处理失败 {os.path.basename(task_path)}: {e}")
            failed += 1

        elapsed = time.perf_counter() - t0
        sys.stdout.write(
            f"\r   [{i:>4}/{len(task_files)}] {os.path.basename(task_path):<35}"
            f" | {elapsed:5.0f}s"
        )
        sys.stdout.flush()

    print(f"\r   预处理完成: 新生成 {done} 套，跳过(已有) {skipped} 套"
          + (f"，失败 {failed} 套" if failed else "") + " " * 20)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_rand_phase2_benchmark(config_path: str, ckpt_path: str,
                               benchmark_subdir: str, output_dir: str,
                               n_min: int = 0, n_max: int = 999999,
                               seed: int = 42, force_regen: bool = False) -> None:
    print("=" * 60)
    print(f"🎲 消融实验1：随机分组 + Phase2 推理 [{benchmark_subdir}]")
    print("=" * 60)

    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    dc        = load_config("configs/data_config.yaml")
    data_dir  = resolve_path(dc["data_dir"])
    topo_path = os.path.join(data_dir, "0_topo.csv")

    n_dirs = [
        d for d in sorted(glob.glob(os.path.join(data_dir, benchmark_subdir, "N*")))
        if n_min <= int(os.path.basename(d)[1:]) <= n_max
    ]
    if not n_dirs:
        print(f"❌ 未找到数据: {data_dir}/{benchmark_subdir}/N*/")
        return

    all_task_files = []
    for d in n_dirs:
        all_task_files.extend(sorted(glob.glob(os.path.join(d, "*_task.csv"))))
    print(f"📂 共 {len(n_dirs)} 档难度，{len(all_task_files)} 套任务\n")

    # ── 步骤1：预生成随机分组文件 ────────────────────────────────────────────
    print(f"🎲 步骤1/2：预生成随机分组（rand 种子={seed}，n_clusters={N_CLUSTERS}）")
    _prepare_rand_files(all_task_files, N_CLUSTERS, seed=seed, force=force_regen)

    # ── 步骤2：Phase2 推理（model_tag='rand'）────────────────────────────────
    print(f"\n🚀 步骤2/2：Phase2 推理（rand 分组）")

    cfg_rand = copy.deepcopy(cfg)
    cfg_rand["environment"]["emb_model_tag"] = RAND_TAG
    cfg_rand["environment"]["n_clusters"]    = N_CLUSTERS

    topo           = tsnkit_core.load_network(topo_path)
    init_task_path = sorted(glob.glob(os.path.join(n_dirs[0], "*_task.csv")))[0]
    init_task      = tsnkit_core.load_stream(init_task_path)

    env_config = {
        "environment": cfg_rand.get("environment", {}),
        "task":        init_task,
        "task_file":   init_task_path,
        "topo":        topo,
    }

    env   = TSNEnv(env_config)
    agent = load_agent(resolve_path(ckpt_path), env, cfg_rand)
    print(f"✅ 模型已加载: {ckpt_path}")
    print(f"   设备: {agent.device} | MAX_FLOWS: {env.MAX_FLOWS} | n_clusters: {N_CLUSTERS}\n")

    os.makedirs(output_dir, exist_ok=True)
    result_csv = os.path.join(output_dir, f"rand_phase2_{benchmark_subdir}_results.csv")
    _run_n_dirs(env, agent, n_dirs, result_csv, output_dir, save_schedule=False)


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="消融实验1：随机分组 + Phase2 推理")
    parser.add_argument("--config",           default="configs/phase2.yaml",
                        help="Phase2 配置文件路径")
    parser.add_argument("--ckpt",             required=True,
                        help="Phase2 模型权重路径（phase2_ppo_best.pth）")
    parser.add_argument("--benchmark_subdir", default="benchmark_v4")
    parser.add_argument("--output",           default="results/ablation",
                        help="输出目录（相对于 sca_drl/）")
    parser.add_argument("--n_min",            type=int, default=0)
    parser.add_argument("--n_max",            type=int, default=999999)
    parser.add_argument("--seed",             type=int, default=42,
                        help="随机分组种子（默认42，保证可复现）")
    parser.add_argument("--force",            action="store_true",
                        help="强制重新生成随机分组文件（默认：跳过已有）")
    parser.add_argument("--sleep",            action="store_true",
                        help="运行完成后自动休眠（仅 Windows）")
    args = parser.parse_args()

    run_rand_phase2_benchmark(
        args.config,
        args.ckpt,
        args.benchmark_subdir,
        resolve_path(args.output),
        args.n_min, args.n_max,
        args.seed, args.force,
    )

    if args.sleep:
        print("💤 即将休眠…")
        time.sleep(3)
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
