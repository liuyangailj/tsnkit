"""LS 算法 benchmark 评测脚本（KSP-5 pkl路径 + 流ID顺序 + 并行）。

路径策略
--------
从 Phase1 batch_infer 生成的 *_k5_d32_paths.pkl 加载 KSP-5 路径，
保证与 Phase2 环境使用的路径完全一致。
pkl 缺失时自动 fallback 到按跳数截断的 KSP-5（并打印警告）。

调度顺序
--------
按流 ID 升序（int(stream) 即 CSV 中的 stream 列），与原版 ls.py 的
"最长路径跳数降序"不同，需在 solver.init() 后覆盖 task_order。

并行策略
--------
ProcessPoolExecutor：所有任务扁平化后并行提交，主进程收集结果。
workers <= 物理核数时，各任务的 inference_time / solve_time 基本不受并行影响。

列说明
------
inference_time      : init + prepare + solve 总挂钟时间（perf_counter）
solve_time          : 纯 solve() 挂钟时间（perf_counter）
n_paths_tried_total : 所有流候选路径数之和
avg_paths_per_flow  : n_paths_tried_total / n_flows
max_paths_per_flow  : 单条流最多候选路径数
n_scheduled         : 成功调度的流数（失败时 = 失败前已调度数）
n_flows_failed      : 卡住的流数（LS 遇第一个失败即停，故 0 或 1）
solver              : "LS_KSP5_idord"

用法
----
cd tsnkit/algorithms/sca_drl
python eval/run_ls_benchmark.py --benchmark_subdir benchmark_v4
python eval/run_ls_benchmark.py --benchmark_subdir benchmark_v4 --workers 8
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import glob
import csv
import time
import pickle
import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from tsnkit.algorithms import ls as ls_algo
from tsnkit.core import Result
from tsnkit.core._network import Path as TsnPath
from sca_drl.common.utils import load_config, resolve_path

SOLVER_TAG = "LS_KSP5_idord"
K_PATHS    = 5
PKL_TAG    = "_k5_d32"   # Phase1 模型标签，与 batch_infer 生成的文件名对应


# ─────────────────────────────────────────────────────────────────────
# 核心推理（模块级，供 ProcessPoolExecutor 序列化）
# ─────────────────────────────────────────────────────────────────────

def _run_one(task_path: str, topo_path: str):
    """单任务 LS 推理，返回供写入 CSV 的所有字段。

    路径来源（优先级）：
      1. 同目录的 *{PKL_TAG}_paths.pkl（与 Phase2 完全一致）
      2. fallback：按跳数升序截断的 KSP-5
    调度顺序：流 ID 升序（覆盖 ls.py 默认的最长路降序）。
    """
    solver = ls_algo.ls()
    solver.init(task_path, topo_path)

    # ── 路径加载 ──────────────────────────────────────────────────────
    pkl_path = task_path.replace(".csv", f"{PKL_TAG}_paths.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as pf:
            raw = pickle.load(pf)  # {str(sid): [[n1,n2,...], ...]}
        solver.task_routes = {
            s: [TsnPath(nodes, solver.net) for nodes in raw.get(str(int(s)), [])]
            for s in solver.task
        }
    else:
        # fallback：按跳数升序保留前 K_PATHS 条
        for s in solver.task_routes:
            solver.task_routes[s] = sorted(
                solver.task_routes[s], key=lambda p: len(p.links)
            )[:K_PATHS]
        print(f"⚠️  pkl 缺失，fallback KSP-5: {os.path.basename(pkl_path)}", flush=True)

    # ── 流调度顺序：按流 ID 升序 ──────────────────────────────────────
    solver.task_order = sorted(solver.task.streams, key=lambda s: int(s))

    # ── 路径统计（solve 前）──────────────────────────────────────────
    path_counts         = [len(solver.task_routes[s]) for s in solver.task]
    n_flows             = len(path_counts)
    n_paths_tried_total = sum(path_counts)
    avg_paths_per_flow  = n_paths_tried_total / n_flows if n_flows > 0 else 0.0
    max_paths_per_flow  = max(path_counts) if path_counts else 0

    solver.prepare()

    # ── 计时 & 求解 ───────────────────────────────────────────────────
    t_solve    = time.perf_counter()
    stat       = solver.solve()
    solve_time = time.perf_counter() - t_solve
    # inference_time（含 init）由 _worker 的外层 perf_counter 统一计量

    # ── 调度结果 ──────────────────────────────────────────────────────
    if stat.result == Result.schedulable:
        n_scheduled    = n_flows
        n_flows_failed = 0
    else:
        n_scheduled    = len(solver._paths)
        n_flows_failed = 1

    return dict(
        stat=stat,
        solve_time=solve_time,
        n_flows=n_flows,
        n_scheduled=n_scheduled,
        n_flows_failed=n_flows_failed,
        n_paths_tried_total=n_paths_tried_total,
        avg_paths_per_flow=avg_paths_per_flow,
        max_paths_per_flow=max_paths_per_flow,
    )


def _worker(args):
    """ProcessPoolExecutor worker：包一层 perf_counter 计 inference_time。"""
    n, instance_id, task_path, topo_path = args
    t0   = time.perf_counter()
    res  = _run_one(task_path, topo_path)
    res["inference_time"] = time.perf_counter() - t0
    res["n"]              = n
    res["instance_id"]    = instance_id
    res["task_file"]      = os.path.basename(task_path)
    return res


# ─────────────────────────────────────────────────────────────────────
# 格式化工具
# ─────────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h > 0 else f"{m:02d}m{s:02d}s"


# ─────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────

def run_ls_benchmark(benchmark_subdir: str, output_dir: str,
                     n_min: int = 0, n_max: int = 999999,
                     sample_per_n: int = 0, seed: int = 42,
                     workers: int = 4) -> None:
    dc        = load_config("configs/data_config.yaml")
    data_dir  = resolve_path(dc["data_dir"])
    topo_path = os.path.join(data_dir, "0_topo.csv")

    if not os.path.exists(topo_path):
        print(f"❌ 找不到拓扑文件: {topo_path}")
        return

    n_dirs = [d for d in sorted(glob.glob(os.path.join(data_dir, benchmark_subdir, "N*")))
              if n_min <= int(os.path.basename(d)[1:]) <= n_max]
    if not n_dirs:
        print(f"❌ 未找到数据: {data_dir}/{benchmark_subdir}/N*/")
        return

    # ── 收集全部任务 ──────────────────────────────────────────────────
    all_tasks = []
    for n_dir in n_dirs:
        n          = int(os.path.basename(n_dir)[1:])
        task_files = sorted(glob.glob(os.path.join(n_dir, "*_task.csv")))
        if sample_per_n > 0 and sample_per_n < len(task_files):
            task_files = sorted(random.Random(seed).sample(task_files, sample_per_n))
        for tf in task_files:
            stem        = os.path.basename(tf).replace(".csv", "")
            instance_id = int(stem.split("_")[0])
            all_tasks.append((n, instance_id, tf, topo_path))

    total = len(all_tasks)
    print(f"📊 {SOLVER_TAG} baseline — {benchmark_subdir}")
    print(f"   共 {len(n_dirs)} 档难度，{total} 套任务，workers={workers}\n")

    os.makedirs(output_dir, exist_ok=True)
    range_tag  = f"_N{n_min}-{n_max}" if (n_min > 0 or n_max < 999999) else ""
    sample_tag = f"_s{sample_per_n}"  if sample_per_n > 0               else ""
    result_csv = os.path.join(output_dir,
                              f"ls_{benchmark_subdir}{range_tag}{sample_tag}_results.csv")

    CSV_HEADER = [
        "N", "instance_id", "success",
        "n_flows", "n_scheduled", "schedulability",
        "inference_time", "solve_time",
        "n_paths_tried_total", "avg_paths_per_flow", "max_paths_per_flow",
        "n_flows_failed", "solver", "task_file",
    ]

    rows      = []
    done      = 0
    t_global  = time.perf_counter()

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker, args): args for args in all_tasks}

            for fut in as_completed(futures):
                res  = fut.result()
                done += 1

                # ── 进度显示（主进程，原地刷新）────────────────────────
                elapsed = time.perf_counter() - t_global
                eta     = elapsed / done * (total - done) if done > 0 else 0.0
                print(f"\r  [{done:3d}/{total:3d}] elapsed {_fmt_time(elapsed)} | ETA {_fmt_time(eta)}   ",
                      end="", flush=True)

                rows.append(res)

    except KeyboardInterrupt:
        print("\n⚠️  用户中断，写出已完成结果…")

    # ── 按 (N, instance_id) 排序后写 CSV ─────────────────────────────
    rows.sort(key=lambda r: (r["n"], r["instance_id"]))

    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for r in rows:
            stat           = r["stat"]
            success        = 1 if stat.result == Result.schedulable else 0
            schedulability = float(success)
            writer.writerow([
                r["n"],
                r["instance_id"],
                success,
                r["n_flows"],
                r["n_scheduled"],
                f"{schedulability:.4f}",
                f"{r['inference_time']:.4f}",
                f"{r['solve_time']:.4f}",
                r["n_paths_tried_total"],
                f"{r['avg_paths_per_flow']:.2f}",
                r["max_paths_per_flow"],
                r["n_flows_failed"],
                SOLVER_TAG,
                r["task_file"],
            ])

    # ── 汇总 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    from collections import defaultdict
    n_rows = defaultdict(list)
    for r in rows:
        n_rows[r["n"]].append(r)
    for n in sorted(n_rows):
        rlist  = n_rows[n]
        avg_s  = np.mean([1 if r["stat"].result == Result.schedulable else 0
                          for r in rlist]) * 100
        avg_t  = np.mean([r["inference_time"] for r in rlist])
        print(f"  N={n:3d} | 调度率: {avg_s:5.1f}% | 均耗时: {avg_t:6.3f}s  [{len(rlist)} 套]")

    total_t = time.perf_counter() - t_global
    print(f"{'─'*60}")
    print(f"  总用时: {_fmt_time(total_t)}")
    print(f"\n✅ {SOLVER_TAG} baseline 完成，结果: {result_csv}")


# ─────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LS KSP5-idord benchmark 评测")
    parser.add_argument("--benchmark_subdir", default="benchmark_v4",
                        help="benchmark 子目录名（benchmark_v4 / benchmark_v3 等）")
    parser.add_argument("--output", default="results/ls_baseline",
                        help="输出目录（相对于 sca_drl/）")
    parser.add_argument("--n_min", type=int, default=0,
                        help="只跑 N >= n_min 的档位（默认不限）")
    parser.add_argument("--n_max", type=int, default=999999,
                        help="只跑 N <= n_max 的档位（默认不限）")
    parser.add_argument("--sample", type=int, default=0,
                        help="每档随机抽取套数（0=全跑，默认0）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机抽样种子（默认42）")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行进程数（默认4，建议 <= 物理核数）")
    parser.add_argument("--sleep", action="store_true",
                        help="运行完成后自动休眠（仅 Windows）")
    args = parser.parse_args()

    run_ls_benchmark(
        args.benchmark_subdir,
        resolve_path(args.output),
        args.n_min, args.n_max,
        args.sample, args.seed,
        args.workers,
    )

    if args.sleep:
        print("💤 即将休眠…")
        time.sleep(3)
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
