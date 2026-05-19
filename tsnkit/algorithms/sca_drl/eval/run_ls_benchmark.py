"""LS 算法 benchmark 评测脚本（KSP-5 路径策略）。

对指定 benchmark 子目录下的全部实例运行 tsnkit 原生 LS 算法，
输出与 phase2/infer.py benchmark 模式格式对齐的 CSV，方便直接 join 对比。

路径策略
--------
与 Phase 1 保持一致：对每条流取跳数最短的 K=5 条简单路径（KSP-5）。
实现方式：调用 ls.init() 获取全部路径后，按 len(path.links) 排序取前 5 条，
不修改 ls.py 本身。

列说明
------
n_scheduled : LS 成功时 = n_flows；失败时 = -1（LS 遇第一个失败即返回，中间状态未知）
schedulability : 二值（1.0 / 0.0），不同于 DRL 的连续值
n_groups    : -1（LS 不使用 Phase 1 分组）

用法
----
cd tsnkit/algorithms/sca_drl
python eval/run_ls_benchmark.py --benchmark_subdir benchmark_v3
python eval/run_ls_benchmark.py --benchmark_subdir benchmark_v2
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import glob
import csv
import time
import argparse
import numpy as np
import pandas as pd

from tsnkit.algorithms import ls as ls_algo
from tsnkit.core import Result
from sca_drl.common.utils import load_config, resolve_path

K_PATHS = 5


def _run_ls_ksp(task_path: str, topo_path: str, k: int = K_PATHS):
    """KSP-k 路径策略的 LS 推理：init() 后截断 task_routes，再 solve()。

    不修改 ls.py，通过在 init() 和 solve() 之间替换 task_routes 实现。
    """
    solver = ls_algo.ls()
    solver.init(task_path, topo_path)

    # 按跳数升序排序，只保留前 k 条（与 Phase 1 KSP 策略对齐）
    for s in solver.task_routes:
        solver.task_routes[s] = sorted(
            solver.task_routes[s], key=lambda p: len(p.links)
        )[:k]

    solver.prepare()
    return solver.solve()


def run_ls_benchmark(benchmark_subdir: str, output_dir: str) -> None:
    dc        = load_config("configs/data_config.yaml")
    data_dir  = resolve_path(dc["data_dir"])
    topo_path = os.path.join(data_dir, "0_topo.csv")

    if not os.path.exists(topo_path):
        print(f"❌ 找不到拓扑文件: {topo_path}")
        return

    n_dirs = [d for d in sorted(glob.glob(os.path.join(data_dir, benchmark_subdir, "N*")))
              if int(os.path.basename(d)[1:]) >= 200]
    if not n_dirs:
        print(f"❌ 未找到数据: {data_dir}/{benchmark_subdir}/N*/")
        return

    total = sum(len(glob.glob(os.path.join(d, "*_task.csv"))) for d in n_dirs)
    print(f"📊 LS-KSP{K_PATHS} baseline — {benchmark_subdir}")
    print(f"   共 {len(n_dirs)} 档难度，{total} 套任务\n")
    print(f"  {'N':>4} | {'instance':<12} | {'result':<12} | {'time(s)':>10}")

    os.makedirs(output_dir, exist_ok=True)
    result_csv = os.path.join(output_dir, f"ls_{benchmark_subdir}_results.csv")

    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "instance_id", "success",
                         "n_flows", "n_scheduled", "schedulability",
                         "inference_time", "n_groups", "task_file"])

        for n_dir in n_dirs:
            n = int(os.path.basename(n_dir)[1:])
            task_files = sorted(glob.glob(os.path.join(n_dir, "*_task.csv")))[:3]
            sched_list, time_list = [], []

            for task_path in task_files:
                stem        = os.path.basename(task_path).replace(".csv", "")
                instance_id = int(stem.split("_")[0])
                n_flows = len(pd.read_csv(task_path))

                t0      = time.perf_counter()
                stat    = _run_ls_ksp(task_path, topo_path, k=K_PATHS)
                elapsed = time.perf_counter() - t0

                success        = 1 if stat.result == Result.schedulable else 0
                schedulability = 1.0 if success else 0.0
                # LS 失败时立即返回，无法得知已调度流数，用 -1 标记
                n_scheduled    = n_flows if success else -1

                result_str = "schedulable" if success else stat.result.name
                print(f"  N={n:3d} | {stem:<12} | {result_str:<12} | {elapsed:>10.3f} s")

                sched_list.append(schedulability)
                time_list.append(elapsed)

                writer.writerow([
                    n,
                    instance_id,
                    success,
                    n_flows,
                    n_scheduled,
                    f"{schedulability:.4f}",
                    f"{elapsed:.4f}",
                    -1,
                    os.path.basename(task_path),
                ])
                f.flush()

            avg_s = np.mean(sched_list) * 100
            avg_t = np.mean(time_list)
            print(f"  N={n:3d} | 调度率: {avg_s:5.1f}% | 均求解时间: {avg_t:6.3f} s"
                  f"  [{len(task_files)} 套]")

    print(f"\n✅ LS-KSP{K_PATHS} baseline 完成，结果: {result_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LS-KSP 算法 benchmark 评测")
    parser.add_argument("--benchmark_subdir", default="benchmark_v3",
                        help="benchmark 子目录名（benchmark_v2 / benchmark_v3 等）")
    parser.add_argument("--output", default="results/ls_baseline",
                        help="输出目录（相对于 sca_drl/）")
    args = parser.parse_args()

    try:
        run_ls_benchmark(args.benchmark_subdir, resolve_path(args.output))
    except KeyboardInterrupt:
        print("\n⚠️  用户中断，已退出。")
        sys.exit(0)
