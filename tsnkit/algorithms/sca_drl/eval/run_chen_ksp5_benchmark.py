"""Chen-KSP5 adapted benchmark for SCA-DRL comparisons.

This script mirrors eval/run_ls_benchmark.py, but replaces the LS policy with
an adapted implementation of Chen et al.'s incremental TT scheduling heuristic:

  * KSP-5 candidate paths are loaded from Phase1 *_k5_d32_paths.pkl files.
  * Streams are ordered by Chen-style urgency: shorter period first, then
    larger frame, then tighter deadline.
  * For each stream, candidate paths are scored by earliest feasible injection
    time under the same no-wait/link-conflict checks used by tsnkit ls.py.
  * On failure, an optional Chen-style fallback moves one already scheduled
    stream with the most similar path and retries the failed stream.

Usage:
    cd tsnkit/algorithms/sca_drl
    python eval/run_chen_ksp5_benchmark.py --benchmark_subdir benchmark_v4
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ALGORITHMS_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
for path in (REPO_ROOT, ALGORITHMS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from tsnkit.algorithms import ls as ls_algo
from tsnkit.core import Result
from tsnkit.core._network import Path as TsnPath
from sca_drl.common.utils import load_config, resolve_path


SOLVER_TAG = "Chen_KSP5_prune_fb"
SOLVER_TAG_NO_FB = "Chen_KSP5_prune"
K_PATHS = 5
PKL_TAG = "_k5_d32"


class ChenKSP5Adapted(ls_algo.ls):
    """Chen-style KSP5 heuristic adapted to the existing tsnkit LS engine."""

    def __init__(
        self,
        workers: int = 1,
        enable_fallback: bool = True,
        fallback_top_m: int = 5,
        fallback_max_shifts: int = 64,
    ) -> None:
        super().__init__(workers)
        self.enable_fallback = enable_fallback
        self.fallback_top_m = fallback_top_m
        self.fallback_max_shifts = fallback_max_shifts
        self.fallback_attempts = 0
        self.fallback_rescued = 0
        self.fallback_moved = 0

    @staticmethod
    def chen_priority(stream) -> Tuple[int, int, int, int]:
        return (stream.period, -stream.size, stream.deadline, int(stream))

    def prepare(self) -> None:
        self.task_order = sorted(self.task.streams, key=self.chen_priority)

    def solve(self):
        start_time = time.perf_counter()
        for stream in self.task_order:
            path, inject_time = self.select_path_by_chen_pruning(stream)
            if path is not None and inject_time is not None:
                self.commit_flow(stream, path, inject_time)
            elif self.enable_fallback and self.try_chen_fallback(stream):
                pass
            else:
                end_time = time.perf_counter()
                return ls_algo.utils.Statistics(
                    "-", ls_algo.utils.Result.unschedulable, end_time - start_time
                )

            if time.perf_counter() - start_time > ls_algo.utils.T_LIMIT:
                end_time = time.perf_counter()
                return ls_algo.utils.Statistics(
                    "-", ls_algo.utils.Result.unknown, end_time - start_time
                )

        end_time = time.perf_counter()
        return ls_algo.utils.Statistics(
            "-", ls_algo.utils.Result.schedulable, end_time - start_time
        )

    def select_path_by_chen_pruning(self, stream):
        best = None
        for path_idx, path in enumerate(self.task_routes.get(stream, [])[:K_PATHS]):
            delay = self.get_nw_delay(stream, path)
            if delay > stream.deadline:
                continue
            inject_time = self.find_inject_time(stream, path)
            if inject_time == -1:
                continue
            max_util, total_util = self.path_utilization(path)
            score = (
                inject_time,
                max_util,
                total_util,
                len(path.links),
                delay,
                path_idx,
            )
            if best is None or score < best[0]:
                best = (score, path, inject_time)
        if best is None:
            return None, None
        return best[1], best[2]

    def path_utilization(self, path) -> Tuple[float, float]:
        utils = []
        for link in path.links:
            occupied = sum(end - start for start, end, _queue in self._result[link])
            util = occupied / self.task.lcm if self.task.lcm else 0.0
            utils.append(util)
        if not utils:
            return 0.0, 0.0
        return max(utils), sum(utils)

    def is_offset_feasible(self, stream, path, offset: int) -> bool:
        delay = self.get_nw_delay(stream, path)
        if delay > stream.deadline:
            return False
        if offset < 0 or offset > stream.period - delay:
            return False

        prev_end = offset
        for link in path.links:
            start = prev_end
            end = start + stream.get_t_trans(link)
            prev_end = start + link.t_proc + stream.get_t_trans(link)

            for frame_idx in stream.get_frame_indexes(self.task.lcm):
                abs_start = start + frame_idx * stream.period
                abs_end = end + frame_idx * stream.period

                match_start = self.match_time(abs_start, self._result[link])
                match_end = self.match_time(abs_end, self._result[link])

                if (
                    match_start == -1
                    and self._result[link]
                    and abs_end > self._result[link][0][0]
                ):
                    return False
                if (
                    match_start == -2
                    and self._result[link]
                    and abs_start < self._result[link][-1][1]
                ):
                    return False
                if match_start >= 0 and (
                    match_start != match_end
                    or (
                        self._result[link]
                        and self._result[link][match_start][1] > abs_start
                    )
                ):
                    return False
        return True

    def commit_flow(self, stream, path, inject_time: int) -> None:
        self._delay[stream] = self.get_nw_delay(stream, path)
        self._paths[stream] = path
        self._offset[stream] = inject_time

        prev_end = inject_time
        for link in path.links:
            start = prev_end
            end = start + stream.get_t_trans(link)
            prev_end = start + link.t_proc + stream.get_t_trans(link)
            for frame_idx in stream.get_frame_indexes(self.task.lcm):
                self._result[link].append(
                    (start + frame_idx * stream.period, end + frame_idx * stream.period, 0)
                )
            self._result[link].sort(key=lambda entry: entry[0], reverse=False)

    def remove_flow(self, stream):
        if stream not in self._paths:
            return None

        record = {
            "path": self._paths[stream],
            "offset": self._offset[stream],
            "delay": self._delay.get(stream),
        }
        path = record["path"]
        offset = record["offset"]

        prev_end = offset
        for link in path.links:
            start = prev_end
            end = start + stream.get_t_trans(link)
            prev_end = start + link.t_proc + stream.get_t_trans(link)
            for frame_idx in stream.get_frame_indexes(self.task.lcm):
                entry = (start + frame_idx * stream.period, end + frame_idx * stream.period, 0)
                try:
                    self._result[link].remove(entry)
                except ValueError:
                    # If the exact tuple is already absent, keep the rollback path
                    # conservative by rebuilding the sorted list unchanged.
                    pass
            self._result[link].sort(key=lambda item: item[0], reverse=False)

        self._paths.pop(stream, None)
        self._offset.pop(stream, None)
        self._delay.pop(stream, None)
        return record

    def restore_flow(self, stream, record) -> None:
        self.commit_flow(stream, record["path"], record["offset"])
        self._delay[stream] = record["delay"]

    def shared_link_count(self, path_a, path_b) -> int:
        return len(set(path_a.links).intersection(path_b.links))

    def similar_scheduled_streams(self, current_stream, current_path) -> List:
        candidates = []
        for scheduled_stream, scheduled_path in self._paths.items():
            shared = self.shared_link_count(current_path, scheduled_path)
            if shared <= 0:
                continue
            candidates.append(
                (
                    -shared,
                    abs(scheduled_stream.period - current_stream.period),
                    -scheduled_stream.size,
                    int(scheduled_stream),
                    scheduled_stream,
                )
            )
        candidates.sort()
        return [item[-1] for item in candidates[: self.fallback_top_m]]

    def shifted_offsets(self, stream, record) -> Iterable[int]:
        path = record["path"]
        delay = self.get_nw_delay(stream, path)
        latest = stream.period - delay
        if latest < 0:
            return []

        first_link = path.links[0]
        shift_step = max(1, stream.get_t_trans(first_link))
        offsets = []
        for shift_idx in range(1, self.fallback_max_shifts + 1):
            offset = record["offset"] + shift_idx * shift_step
            if offset > latest:
                break
            offsets.append(offset)
        return offsets

    def try_chen_fallback(self, stream) -> bool:
        for current_path in self.task_routes.get(stream, [])[:K_PATHS]:
            if self.get_nw_delay(stream, current_path) > stream.deadline:
                continue
            for moved_stream in self.similar_scheduled_streams(stream, current_path):
                old_record = self.remove_flow(moved_stream)
                if old_record is None:
                    continue

                self.fallback_attempts += 1
                moved_success = False

                for new_offset in self.shifted_offsets(moved_stream, old_record):
                    moved_path = old_record["path"]
                    if not self.is_offset_feasible(moved_stream, moved_path, new_offset):
                        continue

                    self.commit_flow(moved_stream, moved_path, new_offset)
                    inject_time = self.find_inject_time(stream, current_path)
                    if inject_time != -1:
                        self.commit_flow(stream, current_path, inject_time)
                        self.fallback_rescued += 1
                        self.fallback_moved += 1
                        moved_success = True
                        break

                    self.remove_flow(moved_stream)

                if moved_success:
                    return True

                self.restore_flow(moved_stream, old_record)
        return False


def _load_ksp5_routes(solver: ChenKSP5Adapted, task_path: str, pkl_tag: str = "_k5_d32") -> None:
    pkl_path = task_path.replace(".csv", f"{pkl_tag}_paths.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as pf:
            raw = pickle.load(pf)
        solver.task_routes = {
            stream: [TsnPath(nodes, solver.net) for nodes in raw.get(str(int(stream)), [])][
                :K_PATHS
            ]
            for stream in solver.task
        }
    else:
        for stream in solver.task_routes:
            solver.task_routes[stream] = sorted(
                solver.task_routes[stream], key=lambda path: len(path.links)
            )[:K_PATHS]
        print(
            f"Warning: paths pkl missing, fallback to hop-sorted KSP5: {os.path.basename(pkl_path)}",
            flush=True,
        )


def _run_one(
    task_path: str,
    topo_path: str,
    enable_fallback: bool,
    fallback_top_m: int,
    fallback_max_shifts: int,
    pkl_tag: str = "_k5_d32",
) -> dict:
    solver = ChenKSP5Adapted(
        enable_fallback=enable_fallback,
        fallback_top_m=fallback_top_m,
        fallback_max_shifts=fallback_max_shifts,
    )
    solver.init(task_path, topo_path)
    _load_ksp5_routes(solver, task_path, pkl_tag)

    path_counts = [len(solver.task_routes[stream]) for stream in solver.task]
    n_flows = len(path_counts)
    n_paths_tried_total = sum(path_counts)
    avg_paths_per_flow = n_paths_tried_total / n_flows if n_flows else 0.0
    max_paths_per_flow = max(path_counts) if path_counts else 0

    solver.prepare()

    t_solve = time.perf_counter()
    stat = solver.solve()
    solve_time = time.perf_counter() - t_solve

    if stat.result == Result.schedulable:
        n_scheduled = n_flows
        n_flows_failed = 0
    else:
        n_scheduled = len(solver._paths)
        n_flows_failed = 1

    return {
        "stat": stat,
        "solve_time": solve_time,
        "n_flows": n_flows,
        "n_scheduled": n_scheduled,
        "n_flows_failed": n_flows_failed,
        "n_paths_tried_total": n_paths_tried_total,
        "avg_paths_per_flow": avg_paths_per_flow,
        "max_paths_per_flow": max_paths_per_flow,
        "fallback_attempts": solver.fallback_attempts,
        "fallback_rescued": solver.fallback_rescued,
        "fallback_moved": solver.fallback_moved,
    }


def _worker(args):
    (
        n,
        instance_id,
        task_path,
        topo_path,
        enable_fallback,
        fallback_top_m,
        fallback_max_shifts,
        pkl_tag,
    ) = args
    t0 = time.perf_counter()
    res = _run_one(
        task_path,
        topo_path,
        enable_fallback,
        fallback_top_m,
        fallback_max_shifts,
        pkl_tag,
    )
    res["inference_time"] = time.perf_counter() - t0
    res["n"] = n
    res["instance_id"] = instance_id
    res["task_file"] = os.path.basename(task_path)
    return res


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


def run_chen_ksp5_benchmark(
    benchmark_subdir: str,
    output_dir: str,
    n_min: int = 0,
    n_max: int = 999999,
    sample_per_n: int = 0,
    seed: int = 42,
    workers: int = 4,
    enable_fallback: bool = True,
    fallback_top_m: int = 5,
    fallback_max_shifts: int = 64,
    pkl_tag: str = "_k5_d32",
) -> None:
    dc = load_config("configs/data_config.yaml")
    data_dir = resolve_path(dc["data_dir"])
    topo_path = os.path.join(data_dir, "0_topo.csv")

    if not os.path.exists(topo_path):
        print(f"Cannot find topology file: {topo_path}")
        return

    n_dirs = [
        directory
        for directory in sorted(glob.glob(os.path.join(data_dir, benchmark_subdir, "N*")))
        if n_min <= int(os.path.basename(directory)[1:]) <= n_max
    ]
    if not n_dirs:
        print(f"Cannot find benchmark data under {data_dir}/{benchmark_subdir}/N*/")
        return

    all_tasks = []
    for n_dir in n_dirs:
        n = int(os.path.basename(n_dir)[1:])
        task_files = sorted(glob.glob(os.path.join(n_dir, "*_task.csv")))
        if sample_per_n > 0 and sample_per_n < len(task_files):
            task_files = sorted(random.Random(seed).sample(task_files, sample_per_n))
        for task_file in task_files:
            stem = os.path.basename(task_file).replace(".csv", "")
            instance_id = int(stem.split("_")[0])
            all_tasks.append(
                (
                    n,
                    instance_id,
                    task_file,
                    topo_path,
                    enable_fallback,
                    fallback_top_m,
                    fallback_max_shifts,
                    pkl_tag,
                )
            )

    solver_tag = SOLVER_TAG if enable_fallback else SOLVER_TAG_NO_FB
    print(f"Chen-KSP5 benchmark - {benchmark_subdir}")
    print(
        f"  tasks={len(all_tasks)}, n_dirs={len(n_dirs)}, workers={workers}, "
        f"fallback={enable_fallback}, top_m={fallback_top_m}, max_shifts={fallback_max_shifts}"
    )

    os.makedirs(output_dir, exist_ok=True)
    range_tag = f"_N{n_min}-{n_max}" if (n_min > 0 or n_max < 999999) else ""
    sample_tag = f"_s{sample_per_n}" if sample_per_n > 0 else ""
    fb_tag = "" if enable_fallback else "_nofb"
    result_csv = os.path.join(
        output_dir,
        f"chen_ksp5_{benchmark_subdir}{range_tag}{sample_tag}{fb_tag}_results.csv",
    )

    csv_header = [
        "N",
        "instance_id",
        "success",
        "n_flows",
        "n_scheduled",
        "schedulability",
        "inference_time",
        "solve_time",
        "n_paths_tried_total",
        "avg_paths_per_flow",
        "max_paths_per_flow",
        "n_flows_failed",
        "fallback_attempts",
        "fallback_rescued",
        "fallback_moved",
        "solver",
        "task_file",
    ]

    rows = []
    done = 0
    t_global = time.perf_counter()

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker, args): args for args in all_tasks}
            for future in as_completed(futures):
                res = future.result()
                done += 1
                elapsed = time.perf_counter() - t_global
                eta = elapsed / done * (len(all_tasks) - done) if done else 0.0
                print(
                    f"\r  [{done:3d}/{len(all_tasks):3d}] elapsed {_fmt_time(elapsed)} | ETA {_fmt_time(eta)}   ",
                    end="",
                    flush=True,
                )
                rows.append(res)
    except KeyboardInterrupt:
        print("\nInterrupted; writing completed rows.")

    rows.sort(key=lambda item: (item["n"], item["instance_id"]))
    with open(result_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        for row in rows:
            stat = row["stat"]
            success = 1 if stat.result == Result.schedulable else 0
            writer.writerow(
                [
                    row["n"],
                    row["instance_id"],
                    success,
                    row["n_flows"],
                    row["n_scheduled"],
                    f"{float(success):.4f}",
                    f"{row['inference_time']:.4f}",
                    f"{row['solve_time']:.4f}",
                    row["n_paths_tried_total"],
                    f"{row['avg_paths_per_flow']:.2f}",
                    row["max_paths_per_flow"],
                    row["n_flows_failed"],
                    row["fallback_attempts"],
                    row["fallback_rescued"],
                    row["fallback_moved"],
                    solver_tag,
                    row["task_file"],
                ]
            )

    print()
    print("-" * 60)
    from collections import defaultdict

    n_rows: Dict[int, list] = defaultdict(list)
    for row in rows:
        n_rows[row["n"]].append(row)
    for n in sorted(n_rows):
        group = n_rows[n]
        avg_success = np.mean(
            [1 if row["stat"].result == Result.schedulable else 0 for row in group]
        ) * 100
        avg_time = np.mean([row["inference_time"] for row in group])
        rescued = sum(row["fallback_rescued"] for row in group)
        print(
            f"  N={n:3d} | schedulability={avg_success:5.1f}% | "
            f"avg_time={avg_time:7.3f}s | fallback_rescued={rescued}"
        )
    print("-" * 60)
    print(f"  total time: {_fmt_time(time.perf_counter() - t_global)}")
    print(f"Done: {result_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chen-KSP5 adapted benchmark")
    parser.add_argument("--benchmark_subdir", default="benchmark_v4")
    parser.add_argument("--output", default="results/chen_ksp5_baseline")
    parser.add_argument("--n_min", type=int, default=0)
    parser.add_argument("--n_max", type=int, default=999999)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fallback_top_m", type=int, default=5)
    parser.add_argument("--fallback_max_shifts", type=int, default=64)
    parser.add_argument("--no_fallback", action="store_true")
    parser.add_argument("--pkl_tag", default="_k5_d32",
                        help="Phase1 路径文件标签（默认 _k5_d32，benchmark_v2 传 _k5_d32_cng）")
    args = parser.parse_args()

    run_chen_ksp5_benchmark(
        args.benchmark_subdir,
        resolve_path(args.output),
        args.n_min,
        args.n_max,
        args.sample,
        args.seed,
        args.workers,
        enable_fallback=not args.no_fallback,
        fallback_top_m=args.fallback_top_m,
        fallback_max_shifts=args.fallback_max_shifts,
        pkl_tag=args.pkl_tag,
    )
