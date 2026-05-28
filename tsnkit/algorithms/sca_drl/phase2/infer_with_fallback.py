"""Phase 2 带双层 Fallback 的推理脚本。

与 infer.py 的区别
  run_one_with_fallback 替代 run_one，推理阶段加入两层 fallback：
  - Layer 1 (路径重试): agent 路径失败后在环境内部遍历剩余 K-1 条候选路径，
    第一条成功即采用，对 agent 完全透明，计入 fallback_path_rescued。
  - Layer 2 (延迟重试): K 条路径全失败则将流标记为 status=2 进入 deferred 队列，
    继续调度本组其他流；本组全部非 deferred 流处理完后再用全 K 条路径重试一次，
    计入 fallback_deferred_rescued。

状态机
  1  pending    等待 agent 选择
  0  scheduled  已写入 physics_engine._paths/_offset/_result
 -1  failed     全路径失败，无账本写入
  2  deferred   主 pass 全 K 路径失败，排队等待本组尾部重试

完整性校验（每个实例推理结束后执行）
  status==0 数量 == len(physics_engine._paths)
  status==2 数量 == 0

训练代码 (train.py / environment.py) 和原版 infer.py 均不改动。
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
from infer import load_agent, serialize_schedule, _read_n_groups


# ─────────────────────────────────────────────────────────────────────────────
# 1. 完整性校验
# ─────────────────────────────────────────────────────────────────────────────

def _verify_result_integrity(env: TSNEnv) -> bool:
    """调度结束后校验内部一致性，两项检查全通过返回 True。

    检查 1: status==0 流数量 == physics_engine._paths 写入数量
    检查 2: status==2 (未解决的 deferred) 数量 == 0
    """
    status_success     = sum(1 for s in env.flow_states.values() if s['status'] == 0)
    physics_count      = len(env.physics_engine._paths)
    deferred_remaining = sum(1 for s in env.flow_states.values() if s['status'] == 2)

    ok = True
    if status_success != physics_count:
        print(f"[INTEGRITY ERROR] status=0 count={status_success} "
              f"!= _paths count={physics_count}")
        ok = False
    if deferred_remaining != 0:
        print(f"[INTEGRITY ERROR] deferred_remaining={deferred_remaining} "
              f"(status=2 流未全部解决)")
        ok = False
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 2. 单实例推理核心（双层 Fallback）
# ─────────────────────────────────────────────────────────────────────────────

def run_one_with_fallback(env: TSNEnv, agent: PPOAgent, task_path: str) -> dict:
    """带双层 fallback 的确定性贪心推理（argmax，torch.no_grad）。

    返回字典
    --------
    n_flows                  : 实际调度的流数（截断后）
    n_scheduled              : 成功分配时隙的流数
    schedulability           : n_scheduled / n_flows
    time_s                   : rollout 耗时（秒），不含数据加载
    failed_ids               : 未能调度的流 ID 列表
    physics                  : env.physics_engine 引用，用于序列化
    fallback_path_rescued    : Layer 1 救活流数（路径重试）
    fallback_deferred_rescued: Layer 2 救活流数（延迟重试）
    fallback_total_rescued   : 两层合计
    integrity_ok             : 完整性校验（False 时不计入汇总统计）
    """
    task = utils_tsnkit.load_stream(task_path)
    env.load_new_task(task, task_path)
    env.reset()

    K_MAX = env.K_MAX
    fallback_path_count     = 0
    fallback_deferred_count = 0

    # flow index → group_id 映射
    groups = {
        i: env.flow_groups.get(str(int(env.flows[i])), 0)
        for i in range(env.num_flows)
    }
    unique_groups = sorted(set(groups.values()))

    # deferred 队列：group_id → [flow_idx, ...]
    deferred_per_group = {g: [] for g in unique_groups}

    t_start = time.perf_counter()

    for g in unique_groups:
        remaining = {i for i in range(env.num_flows) if groups[i] == g}

        # ── 主 pass：agent 贪心选择 + Layer 1 路径重试 ─────────────────────
        while remaining:
            obs = env._get_observation()

            with torch.no_grad():
                flow_tokens = torch.FloatTensor(
                    obs["flow_tokens"]).unsqueeze(0).to(agent.device)
                g_global    = torch.FloatTensor(
                    obs["global_snapshot"]).unsqueeze(0).to(agent.device)
                mask        = torch.BoolTensor(
                    obs["action_mask"]).unsqueeze(0).to(agent.device)

                logits, _ = agent.network(flow_tokens, g_global)
                logits = logits.masked_fill(~mask, -1e8)
                action = torch.argmax(logits, dim=-1).item()

            flow_idx = action // K_MAX
            path_idx = action % K_MAX

            # 防御：mask 失效时取 remaining 中第一个流强制标失败后继续
            if flow_idx not in remaining:
                print(f"[WARN] mask failure: agent selected flow {flow_idx} "
                      f"not in remaining, force-failing one flow")
                flow_idx = next(iter(remaining))
                env.flow_states[flow_idx]['status'] = -1
                remaining.discard(flow_idx)
                continue

            f      = env.flows[flow_idx]
            routes = env.physics_engine.task_routes[f][:K_MAX]

            # path_idx 越界保护（候选路径数可能 < K_MAX）
            eff_path_idx = path_idx if path_idx < len(routes) else 0

            # Layer 1: 先试 agent 选中路径，失败再遍历其余 K-1 条
            success = False
            ok, _ = env.physics_engine.try_allocate_agent_action(
                f, routes[eff_path_idx])
            if ok:
                success = True
            else:
                for k, path in enumerate(routes):
                    if k == eff_path_idx:
                        continue
                    ok, _ = env.physics_engine.try_allocate_agent_action(f, path)
                    if ok:
                        success = True
                        fallback_path_count += 1
                        break

            if success:
                env.flow_states[flow_idx]['status'] = 0
            else:
                # Layer 2: 全 K 路径失败，进入 deferred 队列，暂标 status=2
                deferred_per_group[g].append(flow_idx)
                env.flow_states[flow_idx]['status'] = 2

            remaining.discard(flow_idx)

        # ── deferred 重试 pass：本组非 deferred 流全部处理后执行 ────────────
        for flow_idx in deferred_per_group[g]:
            f      = env.flows[flow_idx]
            routes = env.physics_engine.task_routes[f][:K_MAX]
            success = False
            for path in routes:
                ok, _ = env.physics_engine.try_allocate_agent_action(f, path)
                if ok:
                    success = True
                    fallback_deferred_count += 1
                    env.flow_states[flow_idx]['status'] = 0
                    break
            if not success:
                env.flow_states[flow_idx]['status'] = -1  # 真正失败

    elapsed = time.perf_counter() - t_start

    integrity_ok = _verify_result_integrity(env)

    n_scheduled = sum(1 for s in env.flow_states.values() if s['status'] == 0)
    failed_ids  = [
        int(env.flows[i])
        for i, s in env.flow_states.items()
        if s['status'] == -1
    ]

    return {
        "n_flows":                   env.num_flows,
        "n_scheduled":               n_scheduled,
        "schedulability":            n_scheduled / max(env.num_flows, 1),
        "time_s":                    elapsed,
        "failed_ids":                failed_ids,
        "physics":                   env.physics_engine,
        "fallback_path_rescued":     fallback_path_count,
        "fallback_deferred_rescued": fallback_deferred_count,
        "fallback_total_rescued":    fallback_path_count + fallback_deferred_count,
        "integrity_ok":              integrity_ok,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Benchmark 批量评估（带 Fallback）
# ─────────────────────────────────────────────────────────────────────────────

def _run_n_dirs_with_fallback(env: TSNEnv, agent: PPOAgent, n_dirs: list,
                               result_csv: str, output_dir: str,
                               save_schedule: bool,
                               sample_per_n: int = 0, seed: int = 42,
                               stop_on_all_fail: bool = False) -> None:
    """带 fallback 的批量推理主循环，integrity_ok=False 的实例不计入均值统计。

    sample_per_n    : 每档随机抽取套数（0 = 全跑）
    stop_on_all_fail: True 时，若某档所有有效实例均失败则提前终止
    """
    import random as _random
    total = sum(len(glob.glob(os.path.join(d, "*_task.csv"))) for d in n_dirs)
    sample_note = f"，每档抽 {sample_per_n} 套" if sample_per_n > 0 else ""
    print(f"📊 共 {len(n_dirs)} 档难度，{total} 套任务{sample_note}，开始 fallback 推理...\n")

    os.makedirs(output_dir, exist_ok=True)
    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N", "instance_id", "success",
            "n_flows", "n_scheduled", "schedulability",
            "inference_time", "n_groups", "task_file",
            "fallback_path_rescued", "fallback_deferred_rescued",
            "fallback_total_rescued", "integrity_ok",
        ])

        for n_dir in n_dirs:
            n = int(os.path.basename(n_dir)[1:])
            task_files   = sorted(glob.glob(os.path.join(n_dir, "*_task.csv")))
            if sample_per_n > 0 and sample_per_n < len(task_files):
                task_files = sorted(_random.Random(seed).sample(task_files, sample_per_n))
            sched_list, time_list, success_list = [], [], []
            fb_path_sum, fb_defer_sum = 0, 0

            for task_path in task_files:
                stem        = os.path.basename(task_path).replace(".csv", "")
                instance_id = int(stem.split("_")[0])
                sys.stdout.write(f"\r  N={n:3d} | {stem:<35s}")
                sys.stdout.flush()

                res = run_one_with_fallback(env, agent, task_path)

                # integrity 失败：写占位行，不纳入均值统计
                if not res["integrity_ok"]:
                    print(f"\r  N={n:3d} | {stem} | [INTEGRITY FAIL] skipping stats")
                    writer.writerow([
                        n, instance_id, "",
                        res["n_flows"], "", "",
                        f"{res['time_s']:.4f}", "",
                        os.path.basename(task_path),
                        res["fallback_path_rescued"],
                        res["fallback_deferred_rescued"],
                        res["fallback_total_rescued"],
                        0,
                    ])
                    f.flush()
                    continue

                success = 1 if res["n_scheduled"] == res["n_flows"] else 0

                if res["failed_ids"]:
                    n_failed = len(res["failed_ids"])
                    print(
                        f"\r  N={n:3d} | {stem} | fail={n_failed} "
                        f"ids={res['failed_ids'][:5]}{'...' if n_failed > 5 else ''}"
                        f" | L1={res['fallback_path_rescued']}"
                        f" L2={res['fallback_deferred_rescued']}"
                    )

                if save_schedule and res["n_scheduled"] > 0:
                    serialize_schedule(res["physics"], stem, output_dir)

                n_groups = _read_n_groups(task_path)
                sched_list.append(res["schedulability"])
                time_list.append(res["time_s"])
                success_list.append(success)
                fb_path_sum  += res["fallback_path_rescued"]
                fb_defer_sum += res["fallback_deferred_rescued"]

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
                    res["fallback_path_rescued"],
                    res["fallback_deferred_rescued"],
                    res["fallback_total_rescued"],
                    1,
                ])
                f.flush()

            if sched_list:
                avg_s = np.mean(sched_list) * 100
                avg_t = np.mean(time_list) * 1000
                print(
                    f"\r  N={n:3d} | 均调度率: {avg_s:5.1f}%"
                    f" | 均推理时间: {avg_t:6.1f} ms"
                    f" | Fallback L1={fb_path_sum} L2={fb_defer_sum}"
                    f"  [{len(task_files)} 套]"
                )

                if stop_on_all_fail and success_list and sum(success_list) == 0:
                    print(f"  [早停] N={n} 全部 {len(success_list)} 套失败，终止推理。")
                    break

    print(f"\n✅ 汇总报告: {result_csv}")


def run_benchmark_with_fallback(env: TSNEnv, agent: PPOAgent,
                                 data_dir: str, output_dir: str,
                                 save_schedule: bool,
                                 benchmark_subdir: str = "benchmark_v2",
                                 sample_per_n: int = 0, seed: int = 42,
                                 stop_on_all_fail: bool = False) -> None:
    """遍历 {benchmark_subdir}/N*/ 批量 fallback 推理，输出 *_fallback_results.csv。"""
    n_dirs = sorted(glob.glob(os.path.join(data_dir, benchmark_subdir, "N*")))
    if not n_dirs:
        print(f"❌ 未找到 benchmark 数据: {data_dir}/{benchmark_subdir}/N*/")
        return
    result_csv = os.path.join(output_dir, f"{benchmark_subdir}_fallback_results.csv")
    _run_n_dirs_with_fallback(env, agent, n_dirs, result_csv, output_dir, save_schedule,
                               sample_per_n=sample_per_n, seed=seed,
                               stop_on_all_fail=stop_on_all_fail)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 单实例模式
# ─────────────────────────────────────────────────────────────────────────────

def run_single_with_fallback(env: TSNEnv, agent: PPOAgent,
                              task_path: str, output_dir: str,
                              save_schedule: bool) -> None:
    """单实例 fallback 推理，终端打印详细结果。"""
    stem = os.path.basename(task_path).replace(".csv", "")
    print(f"🔍 推理 (fallback): {task_path}\n")

    res = run_one_with_fallback(env, agent, task_path)

    print(f"{'='*60}")
    print(f"  任务:           {stem}")
    print(f"  总流数:         {res['n_flows']}")
    print(f"  已调度:         {res['n_scheduled']}")
    print(f"  调度率:         {res['schedulability']*100:.1f}%")
    print(f"  推理时间:       {res['time_s']*1000:.1f} ms")
    print(f"  L1 路径救活:    {res['fallback_path_rescued']} 流")
    print(f"  L2 延迟救活:    {res['fallback_deferred_rescued']} 流")
    print(f"  完整性校验:     {'✓ PASS' if res['integrity_ok'] else '✗ FAIL'}")
    if res["failed_ids"]:
        print(f"  失败流:         {res['failed_ids']}")
    print(f"{'='*60}\n")

    if save_schedule and res["n_scheduled"] > 0 and res["integrity_ok"]:
        os.makedirs(output_dir, exist_ok=True)
        serialize_schedule(res["physics"], stem, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SCA-DRL Phase 2 推理（双层 Fallback 版）")
    parser.add_argument("--config",  default="configs/phase2.yaml",
                        help="Phase 2 配置文件路径")
    parser.add_argument("--ckpt",    required=True,
                        help="模型权重路径（phase2_ppo_best.pth 或 resume.pth）")
    parser.add_argument("--mode",    choices=["single", "benchmark"],
                        default="benchmark")
    parser.add_argument("--task",    default=None,
                        help="[single 模式] 指定单个 task.csv 的路径")
    parser.add_argument("--output",  default="results/infer_fallback",
                        help="输出目录（汇总 CSV 与调度文件的根目录）")
    parser.add_argument("--save_schedule", action="store_true",
                        help="是否输出 GCL/ROUTE/QUEUE/OFFSET 调度文件")
    parser.add_argument("--benchmark_subdir", default="benchmark_v2",
                        help="benchmark 子目录名（benchmark / benchmark_v2 等）")
    parser.add_argument("--emb_model_tag", default="",
                        help="Phase1 文件命名标签（默认空=用 config 值，如 k5_d32_cng）")
    parser.add_argument("--cluster_tag", default="",
                        help="group.csv 版本标签（默认空=c{n_clusters}，均衡版传 c4bal）")
    parser.add_argument("--sample_per_n", type=int, default=0,
                        help="每档随机抽取套数（0=全跑，默认0）")
    parser.add_argument("--seed", type=int, default=42,
                        help="抽样随机种子（默认42）")
    parser.add_argument("--stop_on_all_fail", action="store_true",
                        help="某档位全部有效实例失败时提前终止（默认：跑完所有档位）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    dc        = load_config("configs/data_config.yaml")
    data_dir  = resolve_path(dc["data_dir"])
    topo_path = os.path.join(data_dir, "0_topo.csv")
    topo      = utils_tsnkit.load_network(topo_path)

    if args.mode == "single":
        if not args.task:
            parser.error("--mode single 需要指定 --task 参数")
        init_task_path = args.task
    else:
        first_n = sorted(glob.glob(
            os.path.join(data_dir, args.benchmark_subdir, "N*")))[0]
        init_task_path = sorted(glob.glob(
            os.path.join(first_n, "*_task.csv")))[0]

    init_task = utils_tsnkit.load_stream(init_task_path)
    env_cfg   = cfg.get("environment", {}).copy()
    if args.emb_model_tag:
        env_cfg["emb_model_tag"] = args.emb_model_tag
    if args.cluster_tag:
        env_cfg["cluster_tag"] = args.cluster_tag
    env_config = {
        "environment": env_cfg,
        "task":        init_task,
        "task_file":   init_task_path,
        "topo":        topo,
    }

    print("=" * 60)
    print("🚀 SCA-DRL Phase 2 Fallback 推理引擎启动")
    if args.cluster_tag:
        print(f"   cluster_tag: {args.cluster_tag}")
    print("=" * 60)

    env   = TSNEnv(env_config)
    agent = load_agent(resolve_path(args.ckpt), env, cfg)
    print(f"✅ 模型已加载: {args.ckpt}")
    print(f"   设备: {agent.device} | MAX_FLOWS: {env.MAX_FLOWS}"
          f" | d_feature: {env.d_feature} | W: {env.W}\n")

    output_dir = resolve_path(args.output)

    if args.mode == "benchmark":
        run_benchmark_with_fallback(
            env, agent, data_dir, output_dir,
            args.save_schedule, args.benchmark_subdir,
            sample_per_n=args.sample_per_n, seed=args.seed,
            stop_on_all_fail=args.stop_on_all_fail)
    else:
        run_single_with_fallback(
            env, agent, args.task, output_dir, args.save_schedule)
