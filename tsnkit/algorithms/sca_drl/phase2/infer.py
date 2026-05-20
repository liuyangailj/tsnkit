"""Phase 2 端到端推理：加载已训练的 Transformer-PPO 模型，对任务实例进行确定性贪心调度。

两种模式
  single    — 对单个 task.csv 推理，可选输出 GCL/ROUTE/QUEUE/OFFSET 调度文件
  benchmark — 遍历 benchmark/N*/ 下全部实例，输出汇总统计 benchmark_results.csv

调度结果序列化
  DRL_PhysicsEngine 继承自 tsnkit ls，内部账本(_result/_paths/_offset/_delay)
  格式与 ls 完全一致，可直接调用 physics_engine.output().to_csv() 产出标准格式。
  所有时间量已由 tsnkit Config 类在构造时自动完成 slot → ns 的换算(×T_SLOT=1000)。

前置条件
  - Phase 1 batch_infer 已运行，每个 task.csv 旁存有 _emb.pt 和 _group.csv
  - 已有训练好的 phase2_ppo_best.pth（或 resume.pth）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import time
import glob
import csv
import argparse
import numpy as np
import torch

from tsnkit import core as utils_tsnkit
from environment import TSNEnv
from agent import PPOAgent
from sca_drl.common.utils import load_config, resolve_path, set_seed


# ─────────────────────────────────────────────────────────────────────────────
# 1. 模型加载
# ─────────────────────────────────────────────────────────────────────────────

def load_agent(ckpt_path: str, env: TSNEnv, config: dict) -> PPOAgent:
    """初始化 Agent 并加载已训练权重（推理模式，不需要 optimizer）。

    兼容两种 checkpoint 格式：
      best.pth   — 直接存 network.state_dict()
      resume.pth — 存嵌套字典 {'network_state_dict': ..., 'iteration': ..., ...}
    """
    agent = PPOAgent(env=env, config=config)
    raw = torch.load(ckpt_path, map_location=agent.device, weights_only=False)
    state_dict = raw.get("network_state_dict", raw) if isinstance(raw, dict) else raw
    agent.network.load_state_dict(state_dict)
    agent.network.eval()
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# 2. 单实例推理核心
# ─────────────────────────────────────────────────────────────────────────────

def run_one(env: TSNEnv, agent: PPOAgent, task_path: str) -> dict:
    """对单个 task.csv 运行确定性贪心推理（argmax，torch.no_grad）。

    返回字典
    --------
    n_flows        : 实际调度的流数（截断后）
    n_scheduled    : 成功分配时隙的流数
    schedulability : n_scheduled / n_flows
    time_s         : rollout 耗时（秒），不含数据加载
    failed_ids     : 未能调度的流 ID 列表
    physics        : env.physics_engine 引用（DRL_PhysicsEngine，ls 子类）
                     用于后续调用 .output().to_csv() 序列化调度结果
    """
    task = utils_tsnkit.load_stream(task_path)
    env.load_new_task(task, task_path)
    obs, _ = env.reset()

    t_start = time.perf_counter()

    with torch.no_grad():
        for _ in range(env.num_flows):
            flow_tokens = torch.FloatTensor(obs["flow_tokens"]).unsqueeze(0).to(agent.device)
            g_global    = torch.FloatTensor(obs["global_snapshot"]).unsqueeze(0).to(agent.device)
            mask        = torch.BoolTensor(obs["action_mask"]).unsqueeze(0).to(agent.device)

            logits, _ = agent.network(flow_tokens, g_global)
            logits = logits.masked_fill(~mask, -1e8)
            action = torch.argmax(logits, dim=-1).item()

            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break

    elapsed = time.perf_counter() - t_start

    # 采集结果：flow_states 键为 0-based 流索引，status: 0=成功 / -1=失败
    # Stream 继承自 int，int(stream) 直接返回 stream ID（即 CSV 中的 stream 列值）
    n_scheduled = sum(1 for s in env.flow_states.values() if s["status"] == 0)
    failed_ids  = [
        int(env.flows[i])
        for i, s in env.flow_states.items()
        if s["status"] == -1
    ]

    # 保存 physics 引用：load_new_task 每次重建 DRL_PhysicsEngine，
    # 此引用独立于 env.physics_engine，后续 serialize 调用不受影响
    physics_snapshot = env.physics_engine

    return {
        "n_flows":        env.num_flows,
        "n_scheduled":    n_scheduled,
        "schedulability": n_scheduled / max(env.num_flows, 1),
        "time_s":         elapsed,
        "failed_ids":     failed_ids,
        "physics":        physics_snapshot,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. 调度结果序列化
# ─────────────────────────────────────────────────────────────────────────────

def serialize_schedule(physics, task_stem: str, output_dir: str) -> None:
    """将 DRL_PhysicsEngine 的调度状态写为 tsnkit 标准 CSV。

    输出文件（在 output_dir/schedules/ 下）：
      {task_stem}-GCL.csv    — 门控列表，列: link, queue, start(ns), end(ns), cycle(ns)
      {task_stem}-OFFSET.csv — 发送偏移，列: stream, frame, offset(ns)
      {task_stem}-ROUTE.csv  — 路由决策，列: stream, link
      {task_stem}-QUEUE.csv  — 队列分配，列: stream, frame, link, queue
      {task_stem}-DELAY.csv  — 端到端延迟，列: stream, frame, delay(ns)
    """
    sched_dir = os.path.join(output_dir, "schedules")
    os.makedirs(sched_dir, exist_ok=True)
    try:
        config = physics.output()                        # ls 继承方法，返回 utils.Config
        config.to_csv(task_stem, sched_dir + os.sep)    # path+name 拼接为完整文件路径
        print(f"     ✓ 调度文件: {sched_dir}/{task_stem}-*.csv")
    except Exception as e:
        print(f"     ⚠ 序列化失败 ({task_stem}): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Benchmark 批量评估
# ─────────────────────────────────────────────────────────────────────────────

def _read_n_groups(task_path: str, tag: str = "k5_d32_c4") -> int:
    """从同目录的 _group.csv 读 unique cluster 数，文件不存在时返回 -1。"""
    group_csv = task_path.replace(".csv", f"_{tag}_group.csv")
    if not os.path.exists(group_csv):
        return -1
    import pandas as pd
    return pd.read_csv(group_csv)["group_id"].nunique()


def _run_n_dirs(env: TSNEnv, agent: PPOAgent, n_dirs: list,
                result_csv: str, output_dir: str, save_schedule: bool) -> None:
    """通用批量推理循环，被 benchmark 和 probe 共用。"""
    total = sum(len(glob.glob(os.path.join(d, "*_task.csv"))) for d in n_dirs)
    print(f"📊 共 {len(n_dirs)} 档难度，{total} 套任务，开始批量推理...\n")

    os.makedirs(output_dir, exist_ok=True)
    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "instance_id", "success",
                         "n_flows", "n_scheduled", "schedulability",
                         "inference_time", "n_groups", "task_file"])

        for n_dir in n_dirs:
            n = int(os.path.basename(n_dir)[1:])
            task_files = sorted(glob.glob(os.path.join(n_dir, "*_task.csv")))
            sched_list, time_list = [], []

            for task_path in task_files:
                stem        = os.path.basename(task_path).replace(".csv", "")
                instance_id = int(stem.split("_")[0])   # "0001_task" → 1
                sys.stdout.write(f"\r  N={n:3d} | {stem:<35s}")
                sys.stdout.flush()

                res     = run_one(env, agent, task_path)
                success = 1 if res["n_scheduled"] == res["n_flows"] else 0

                if res["failed_ids"]:
                    n_failed = len(res["failed_ids"])
                    print(f"\r  N={n:3d} | {stem} | fail={n_failed} "
                          f"ids={res['failed_ids'][:5]}{'...' if n_failed > 5 else ''}")

                if save_schedule and res["n_scheduled"] > 0:
                    serialize_schedule(res["physics"], stem, output_dir)

                n_groups = _read_n_groups(task_path)
                sched_list.append(res["schedulability"])
                time_list.append(res["time_s"])

                writer.writerow([
                    n,
                    instance_id,
                    success,
                    res["n_flows"],
                    res["n_scheduled"],
                    f"{res['schedulability']:.4f}",
                    f"{res['time_s']:.4f}",
                    n_groups,
                    os.path.basename(task_path),
                ])
                f.flush()

            avg_s = np.mean(sched_list) * 100
            avg_t = np.mean(time_list) * 1000
            print(f"\r  N={n:3d} | 均调度率: {avg_s:5.1f}% | 均推理时间: {avg_t:6.1f} ms"
                  f"  [{len(task_files)} 套]")

    print(f"\n✅ 汇总报告: {result_csv}")


def run_benchmark(env: TSNEnv, agent: PPOAgent, data_dir: str,
                  output_dir: str, save_schedule: bool,
                  benchmark_subdir: str = "benchmark_v2") -> None:
    """遍历 {benchmark_subdir}/N*/ 批量推理，输出 {benchmark_subdir}_results.csv。"""
    n_dirs = sorted(glob.glob(os.path.join(data_dir, benchmark_subdir, "N*")))
    if not n_dirs:
        print(f"❌ 未找到 benchmark 数据: {data_dir}/{benchmark_subdir}/N*/")
        return
    result_csv = os.path.join(output_dir, f"{benchmark_subdir}_results.csv")
    _run_n_dirs(env, agent, n_dirs, result_csv, output_dir, save_schedule)


def run_probe(env: TSNEnv, agent: PPOAgent, data_dir: str) -> None:
    """遍历 probe/N*/ 探针数据集批量推理，输出难度摸底报告。

    输出：data_dir/probe/probe_results.csv
    列名：N, sample_id, success_rate, success_count, total_flows
    打印：每档 mean / stdev / min / max
    """
    n_dirs = sorted(glob.glob(os.path.join(data_dir, "probe", "N*")))
    if not n_dirs:
        print(f"❌ 未找到探针数据: {data_dir}/probe/N*/  请先运行 batch_infer --probe")
        return

    result_csv = os.path.join(data_dir, "probe", "probe_results.csv")
    total = sum(len(glob.glob(os.path.join(d, "*_task.csv"))) for d in n_dirs)
    print(f"🔬 探针摸底推理: {len(n_dirs)} 档 × {total//len(n_dirs)} 套 = {total} 套\n")

    rows = []
    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "sample_id", "success_rate", "success_count", "total_flows"])

        for n_dir in n_dirs:
            n = int(os.path.basename(n_dir)[1:])
            task_files = sorted(glob.glob(os.path.join(n_dir, "*_task.csv")))
            n_success_rates = []

            for task_path in task_files:
                stem = os.path.basename(task_path).replace(".csv", "")
                # sample_id：从文件名 "0001_task" 提取数字
                sample_id = int(stem.split("_")[0])
                sys.stdout.write(f"\r  N={n:3d} | sample {sample_id:02d} ...")
                sys.stdout.flush()

                res = run_one(env, agent, task_path)
                sr = res["schedulability"]
                n_success_rates.append(sr)

                writer.writerow([n, sample_id,
                                  f"{sr:.4f}",
                                  res["n_scheduled"],
                                  res["n_flows"]])
                f.flush()
                rows.append((n, sample_id, sr, res["n_scheduled"], res["n_flows"]))

            rates = np.array(n_success_rates)
            print(f"\r  N={n:3d} | "
                  f"mean={rates.mean()*100:5.1f}% "
                  f"std={rates.std()*100:4.1f}% "
                  f"min={rates.min()*100:5.1f}% "
                  f"max={rates.max()*100:5.1f}%  [{len(task_files)} 套]")

    print(f"\n{'─'*55}")
    print(f"  {'N':>5}  {'mean':>7}  {'std':>6}  {'min':>7}  {'max':>7}")
    print(f"{'─'*55}")
    from collections import defaultdict
    n_rates = defaultdict(list)
    for n, _, sr, _, _ in rows:
        n_rates[n].append(sr)
    for n in sorted(n_rates):
        rates = np.array(n_rates[n])
        print(f"  N={n:<4d}  {rates.mean()*100:6.1f}%  {rates.std()*100:5.1f}%  "
              f"{rates.min()*100:6.1f}%  {rates.max()*100:6.1f}%")
    print(f"{'─'*55}")
    print(f"\n✅ 结果已保存: {result_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. 单实例模式
# ─────────────────────────────────────────────────────────────────────────────

def run_single(env: TSNEnv, agent: PPOAgent, task_path: str,
               output_dir: str, save_schedule: bool) -> None:
    """单实例推理，终端打印详细结果。"""
    stem = os.path.basename(task_path).replace(".csv", "")
    print(f"🔍 推理: {task_path}\n")

    res = run_one(env, agent, task_path)

    print(f"{'='*55}")
    print(f"  任务:     {stem}")
    print(f"  总流数:   {res['n_flows']}")
    print(f"  已调度:   {res['n_scheduled']}")
    print(f"  调度率:   {res['schedulability']*100:.1f}%")
    print(f"  推理时间: {res['time_s']*1000:.1f} ms")
    if res["failed_ids"]:
        print(f"  失败流:   {res['failed_ids']}")
    print(f"{'='*55}\n")

    if save_schedule and res["n_scheduled"] > 0:
        os.makedirs(output_dir, exist_ok=True)
        serialize_schedule(res["physics"], stem, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCA-DRL Phase 2 端到端推理")
    parser.add_argument("--config",        default="configs/phase2.yaml",
                        help="Phase 2 配置文件路径")
    parser.add_argument("--ckpt",          required=True,
                        help="模型权重路径（phase2_ppo_best.pth 或 resume.pth）")
    parser.add_argument("--mode",          choices=["single", "benchmark", "probe"],
                        default="benchmark")
    parser.add_argument("--task",          default=None,
                        help="[single 模式] 指定单个 task.csv 的路径")
    parser.add_argument("--output",        default="results/infer",
                        help="输出目录（汇总 CSV 与调度文件的根目录）")
    parser.add_argument("--save_schedule", action="store_true",
                        help="是否输出 GCL/ROUTE/QUEUE/OFFSET 调度文件（默认：仅统计不存档）")
    parser.add_argument("--benchmark_subdir", default="benchmark_v2",
                        help="benchmark 子目录名（benchmark / benchmark_v2 等）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    # ── 数据路径（单一事实来源：data_config.yaml） ────────────────────────────
    dc        = load_config("configs/data_config.yaml")
    data_dir  = resolve_path(dc["data_dir"])
    topo_path = os.path.join(data_dir, "0_topo.csv")
    topo      = utils_tsnkit.load_network(topo_path)

    # ── 确定初始化 env 用的第一个 task（W 在此时锁定） ───────────────────────
    if args.mode == "single":
        if not args.task:
            parser.error("--mode single 需要指定 --task 参数")
        init_task_path = args.task
    elif args.mode == "probe":
        first_n = sorted(glob.glob(os.path.join(data_dir, "probe", "N*")))[0]
        init_task_path = sorted(glob.glob(os.path.join(first_n, "*_task.csv")))[0]
    else:
        first_n = sorted(glob.glob(os.path.join(data_dir, args.benchmark_subdir, "N*")))[0]
        init_task_path = sorted(glob.glob(os.path.join(first_n, "*_task.csv")))[0]

    init_task  = utils_tsnkit.load_stream(init_task_path)
    env_config = {
        "environment": cfg.get("environment", {}),
        "task":        init_task,
        "task_file":   init_task_path,
        "topo":        topo,
    }

    print("="*60)
    print("🚀 SCA-DRL Phase 2 推理引擎启动")
    print("="*60)

    env   = TSNEnv(env_config)
    agent = load_agent(resolve_path(args.ckpt), env, cfg)
    print(f"✅ 模型已加载: {args.ckpt}")
    print(f"   设备: {agent.device} | MAX_FLOWS: {env.MAX_FLOWS}"
          f" | d_feature: {env.d_feature} | W: {env.W}\n")

    output_dir = resolve_path(args.output)

    if args.mode == "benchmark":
        run_benchmark(env, agent, data_dir, output_dir, args.save_schedule, args.benchmark_subdir)
    elif args.mode == "probe":
        run_probe(env, agent, data_dir)
    else:
        run_single(env, agent, args.task, output_dir, args.save_schedule)
