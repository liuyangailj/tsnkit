"""
JRS-NW 算法 benchmark 评测脚本（横向对比基线，CBC 开源求解器）

算法简介
--------
jrs_nw (Joint Route and Schedule - No Wait) MILP 精确求解算法，
"No-Wait" 指帧在中间交换机不等待、直接流水转发（无缓存延迟）。
求解器已从 Gurobi 替换为开源 CBC（无变量数限制）。

CBC 状态映射
------------
OPTIMAL / FEASIBLE      → schedulable
INFEASIBLE / INT_INFEAS → unschedulable
NO_SOLUTION_FOUND 等    → unknown（时限内未找到任何解）

CSV 输出列
----------
N, instance_id, success, n_flows, n_scheduled, schedulability,
inference_time, solve_time, cbc_status, cbc_sol_count, solver, task_file

用法
----
cd tsnkit/algorithms/sca_drl
python eval/run_jrs_nw_benchmark.py --benchmark_subdir benchmark_v4
python eval/run_jrs_nw_benchmark.py --benchmark_subdir benchmark_v4 --n_max 150
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import glob
import csv
import time
import argparse
import random
import multiprocessing
from collections import defaultdict

import numpy as np

from tsnkit.core import Result
from sca_drl.common.utils import load_config, resolve_path
from jrs_nw_cbc import jrs_nw_cbc

SOLVER_TAG = "JRS-NW_CBC"


# ─────────────────────────────────────────────────────────────────────────────
# 核心推理
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_jrs(task_path: str, topo_path: str, time_limit_s: float) -> dict:
    """单任务 JRS-NW(CBC) 推理：init → prepare → solve(max_seconds)。"""
    solver = jrs_nw_cbc()
    solver.init(task_path, topo_path)
    n_flows = len(solver.task)

    solver.prepare()
    res = solver.solve(max_seconds=time_limit_s)

    n_scheduled = n_flows if res["result"] == Result.schedulable else 0

    return dict(
        result=res["result"],
        cbc_status=res["cbc_status"],
        sol_count=res["sol_count"],
        solve_time=res["solve_time"],
        n_flows=n_flows,
        n_scheduled=n_scheduled,
    )


def _subprocess_worker(queue, task_path, topo_path, time_limit_s):
    """子进程入口：把结果放入 Queue，供父进程读取。"""
    try:
        result = _run_one_jrs(task_path, topo_path, time_limit_s)
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("err", str(e)))


def _run_with_hard_timeout(task_path: str, topo_path: str,
                            time_limit_s: float) -> dict:
    """在独立子进程中运行推理，挂钟超时后强制 terminate。

    挂钟上限 = time_limit_s（整个 init+prepare+solve 的总时间上限）。
    CBC 内部的 max_seconds 是第二道防线；进程 kill 是第一道硬性保障。
    """
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_subprocess_worker,
        args=(q, task_path, topo_path, time_limit_s),
        daemon=True,
    )
    p.start()
    p.join(timeout=time_limit_s)

    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
        return dict(
            result=Result.unknown,
            cbc_status="WALL_TIMEOUT",
            sol_count=0,
            solve_time=time_limit_s,
            n_flows=0,
            n_scheduled=0,
        )

    if not q.empty():
        tag, data = q.get_nowait()
        if tag == "ok":
            return data
        # tag == "err"
        raise RuntimeError(data)

    # 进程正常退出但 queue 为空（异常退出）
    return dict(
        result=Result.unknown,
        cbc_status="PROC_ERROR",
        sol_count=0,
        solve_time=0.0,
        n_flows=0,
        n_scheduled=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 格式化工具
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h > 0 else f"{m:02d}m{s:02d}s"


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_jrs_nw_benchmark(benchmark_subdir: str, output_dir: str,
                          n_min: int = 0, n_max: int = 999999,
                          sample_per_n: int = 5, seed: int = 42,
                          time_limit_s: float = 600.0) -> None:
    print("=" * 60)
    print(f"🔬 JRS-NW (Gurobi MILP) benchmark [{benchmark_subdir}]")
    print(f"   每实例时限: {time_limit_s:.0f}s | 顺序执行")
    print("=" * 60)

    dc        = load_config("configs/data_config.yaml")
    data_dir  = resolve_path(dc["data_dir"])
    topo_path = os.path.join(data_dir, "0_topo.csv")

    if not os.path.exists(topo_path):
        print(f"❌ 找不到拓扑文件: {topo_path}")
        return

    n_dirs = [
        d for d in sorted(glob.glob(os.path.join(data_dir, benchmark_subdir, "N*")),
                          key=lambda d: int(os.path.basename(d)[1:]))
        if n_min <= int(os.path.basename(d)[1:]) <= n_max
    ]
    if not n_dirs:
        print(f"❌ 未找到数据: {data_dir}/{benchmark_subdir}/N*/")
        return

    # ── 收集任务 ──────────────────────────────────────────────────────────
    all_tasks = []
    for n_dir in n_dirs:
        n          = int(os.path.basename(n_dir)[1:])
        task_files = sorted(glob.glob(os.path.join(n_dir, "*_task.csv")))
        if sample_per_n > 0 and sample_per_n < len(task_files):
            task_files = task_files[:sample_per_n]   # 取前 N 套（文件名有序，即 0001~000N）
        for tf in task_files:
            stem        = os.path.basename(tf).replace(".csv", "")
            instance_id = int(stem.split("_")[0])
            all_tasks.append((n, instance_id, tf))

    total = len(all_tasks)
    print(f"   共 {len(n_dirs)} 档难度，{total} 套任务（顺序执行）\n")

    os.makedirs(output_dir, exist_ok=True)
    range_tag  = f"_N{n_min}-{n_max}" if (n_min > 0 or n_max < 999999) else ""
    sample_tag = f"_s{sample_per_n}"  if sample_per_n > 0               else ""
    tl_tag     = f"_tl{int(time_limit_s)}"
    result_csv = os.path.join(
        output_dir,
        f"jrs_nw_{benchmark_subdir}{range_tag}{sample_tag}{tl_tag}_results.csv"
    )

    CSV_HEADER = [
        "N", "instance_id", "success",
        "n_flows", "n_scheduled", "schedulability",
        "inference_time", "solve_time",
        "cbc_status", "cbc_sol_count",
        "solver", "task_file",
    ]

    rows            = []
    done            = 0
    consec_timeout  = 0          # 连续无解 timeout 计数
    MAX_CONSEC_TL   = 3          # 触发提前终止的阈值
    t_global        = time.perf_counter()

    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for n, instance_id, task_path in all_tasks:
            stem = os.path.basename(task_path).replace(".csv", "")
            sys.stdout.write(
                f"\r  [{done+1:3d}/{total:3d}] N={n:3d} | {stem:<35s}"
            )
            sys.stdout.flush()

            t0 = time.perf_counter()
            try:
                res = _run_with_hard_timeout(task_path, topo_path, time_limit_s)
                inference_time = time.perf_counter() - t0
            except Exception as e:
                print(f"\n   ⚠️ 异常 {stem}: {e}")
                inference_time = time.perf_counter() - t0
                res = dict(
                    result=Result.unknown,
                    cbc_status="ERROR",
                    sol_count=0,
                    solve_time=inference_time,
                    n_flows=0,
                    n_scheduled=0,
                )

            done    += 1
            success  = 1 if res["result"] == Result.schedulable else 0
            n_flows  = res["n_flows"]
            n_sched  = res["n_scheduled"]
            sched_r  = n_sched / max(n_flows, 1)

            # 进度行：有结果的行换行打印，保持可读性
            elapsed = time.perf_counter() - t_global
            eta     = elapsed / done * (total - done) if done > 0 else 0.0
            status_str = res["cbc_status"][:8]
            print(
                f"\r  [{done:3d}/{total:3d}] N={n:3d} | {stem:<30s}"
                f" | {'✓' if success else '✗'} {status_str:<6s}"
                f" | {inference_time:6.1f}s"
                f" | elapsed {_fmt_time(elapsed)} ETA {_fmt_time(eta)}"
            )

            row = dict(
                n=n, instance_id=instance_id,
                success=success,
                n_flows=n_flows,
                n_scheduled=n_sched,
                schedulability=sched_r,
                inference_time=inference_time,
                solve_time=res["solve_time"],
                cbc_status=res["cbc_status"],
                sol_count=res["sol_count"],
                task_file=os.path.basename(task_path),
            )
            rows.append(row)

            writer.writerow([
                n, instance_id, success,
                n_flows, n_sched,
                f"{sched_r:.4f}",
                f"{inference_time:.4f}",
                f"{res['solve_time']:.4f}",
                res["cbc_status"],
                res["sol_count"],
                SOLVER_TAG,
                os.path.basename(task_path),
            ])
            f.flush()

            # ── 连续 timeout 检测 ─────────────────────────────────────────
            if res["cbc_status"] in ("NO_SOLUTION_FOUND", "WALL_TIMEOUT") and res["sol_count"] == 0:
                consec_timeout += 1
            else:
                consec_timeout = 0
            if consec_timeout >= MAX_CONSEC_TL:
                print(f"\n⏹️  连续 {MAX_CONSEC_TL} 次 timeout（无解），提前终止")
                break

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    n_rows = defaultdict(list)
    for r in rows:
        n_rows[r["n"]].append(r)
    for n in sorted(n_rows):
        rlist  = n_rows[n]
        avg_s  = np.mean([r["success"] for r in rlist]) * 100
        avg_t  = np.mean([r["inference_time"] for r in rlist])
        n_tl   = sum(1 for r in rlist if "NO_SOLUTION" in r["cbc_status"] or "FEASIBLE" == r["cbc_status"])
        n_opt  = sum(1 for r in rlist if r["success"])
        print(
            f"  N={n:3d} | 调度率: {avg_s:5.1f}%"
            f" | 均耗时: {avg_t:6.1f}s"
            f" | OPT/TL/total: {n_opt}/{n_tl}/{len(rlist)}"
        )

    total_t = time.perf_counter() - t_global
    print(f"{'─'*60}")
    print(f"  总用时: {_fmt_time(total_t)}")
    print(f"\n✅ JRS-NW 评测完成，结果: {result_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.freeze_support()   # Windows spawn 模式必须
    parser = argparse.ArgumentParser(description="JRS-NW CBC MILP benchmark 评测")
    parser.add_argument("--benchmark_subdir", default="benchmark_v4")
    parser.add_argument("--output",     default="results/ablation",
                        help="输出目录（相对于 sca_drl/）")
    parser.add_argument("--n_min",      type=int, default=0)
    parser.add_argument("--n_max",      type=int, default=999999)
    parser.add_argument("--sample",     type=int, default=5,
                        help="每档取前 N 套（0=全跑，默认5）")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--time_limit", type=float, default=600.0,
                        help="每实例 Gurobi 时限（秒，默认600）")
    parser.add_argument("--sleep",      action="store_true",
                        help="运行完成后自动休眠（仅 Windows）")
    args = parser.parse_args()

    run_jrs_nw_benchmark(
        args.benchmark_subdir,
        resolve_path(args.output),
        args.n_min, args.n_max,
        args.sample, args.seed,
        args.time_limit,
    )

    if args.sleep:
        print("💤 即将休眠…")
        time.sleep(3)
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
