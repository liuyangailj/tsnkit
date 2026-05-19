"""v4 数据集生成：均匀随机采样，一次生成 train_v4 + val_v4 + benchmark_v4。

与 v3 benchmark 相同的采样策略（无核心节点偏置，全 ES 均匀随机），
用于修正 Phase1/Phase2 KSP 路径不一致后的完整重训。

种子隔离（单一事实来源，见 data_config.yaml v4 节）：
  train_v4    : 2000~2359  (6档×60=360套, N=280/320/360/400/440/480)
  val_v4      : 6000~6059  (3档×20=60套,  N=320/400/480, 扁平目录)
  benchmark_v4: 9000~9359  (12档×30=360套)

运行：
  cd tsnkit/algorithms/sca_drl
  python data/generate_v4.py              # 生成全部三组
  python data/generate_v4.py --only train # 只生成 train_v4
  python data/generate_v4.py --only val   # 只生成 val_v4
  python data/generate_v4.py --only bench # 只生成 benchmark_v4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import random
import argparse
import pandas as pd

from sca_drl.common.utils import load_config, resolve_path
from sca_drl.data.data_generater import _ES_MOUNT, _ES_TO_SW


def _gen_one_file(flow_count, seed, out_path, task_cfg, es_nodes):
    """均匀随机生成一个 task.csv（与 benchmark_v3 相同逻辑）。"""
    random.seed(seed)
    period_pool    = task_cfg["period_pool"]
    frame_size_min = task_cfg["frame_size_min"]
    frame_size_max = task_cfg["frame_size_max"]

    rows = []
    for stream_id in range(flow_count):
        src    = random.choice(es_nodes)
        src_sw = _ES_TO_SW.get(src)
        dst    = random.choice([n for n in es_nodes if n != src and _ES_TO_SW.get(n) != src_sw])
        period = random.choice(period_pool)
        size   = random.randint(frame_size_min, frame_size_max)
        rows.append({
            "stream":   stream_id,
            "src":      src,
            "dst":      f"[{dst}]",
            "size":     size,
            "period":   period,
            "deadline": period,
            "jitter":   period,
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)


def gen_train(data_dir, cfg, task_cfg, es_nodes):
    """生成 train_v4/N{n}/ — 档位和套数由 data_config.yaml v4.train 决定。"""
    v4      = cfg["v4"]["train"]
    subdir  = cfg["train_subdir_v4"]
    n_pool  = v4["n_pool"]
    per_n   = v4["per_n"]
    seed0   = v4["seed_start"]

    print(f"\n📚 train_v4: {len(n_pool)}档 × {per_n}套 = {len(n_pool)*per_n}套")
    total = 0
    for n_idx, n in enumerate(n_pool):
        seed_base = seed0 + n_idx * per_n
        out_dir   = os.path.join(data_dir, subdir, f"N{n}")
        os.makedirs(out_dir, exist_ok=True)
        for i in range(per_n):
            _gen_one_file(n, seed_base + i, os.path.join(out_dir, f"{i+1:04d}_task.csv"),
                          task_cfg, es_nodes)
        total += per_n
        print(f"   ✅ {subdir}/N{n}/: {per_n}套  (seed {seed_base}~{seed_base+per_n-1})")
    print(f"   合计 {total} 套")


def gen_val(data_dir, cfg, task_cfg, es_nodes):
    """生成 val_v4/N{n}/ 多档验证集（与 train_v4 相同的 N 子目录结构）。"""
    v4     = cfg["v4"]["val"]
    subdir = cfg["val_subdir_v4"]
    n_pool = v4["n_pool"]
    per_n  = v4["per_n"]
    seed0  = v4["seed_start"]

    print(f"\n🧪 val_v4: {len(n_pool)}档 × {per_n}套 = {len(n_pool)*per_n}套")
    total = 0
    for n_idx, n in enumerate(n_pool):
        seed_base = seed0 + n_idx * per_n
        out_dir   = os.path.join(data_dir, subdir, f"N{n}")
        os.makedirs(out_dir, exist_ok=True)
        for i in range(per_n):
            _gen_one_file(n, seed_base + i, os.path.join(out_dir, f"{i+1:04d}_task.csv"),
                          task_cfg, es_nodes)
        total += per_n
        print(f"   ✅ {subdir}/N{n}/: {per_n}套  (seed {seed_base}~{seed_base+per_n-1})")
    print(f"   合计 {total} 套")


def gen_benchmark(data_dir, cfg, task_cfg, es_nodes):
    """生成 benchmark_v4/N{n}/ 12档×30=360套，供评测用。"""
    v4     = cfg["v4"]["benchmark"]
    subdir = v4["subdir"]
    n_pool = v4["n_pool"]
    per_n  = v4["per_n"]
    seed0  = v4["seed_start"]

    print(f"\n📊 benchmark_v4: {len(n_pool)}档 × {per_n}套 = {len(n_pool)*per_n}套")
    total = 0
    for n_idx, n in enumerate(n_pool):
        seed_base = seed0 + n_idx * per_n
        out_dir   = os.path.join(data_dir, subdir, f"N{n}")
        os.makedirs(out_dir, exist_ok=True)
        for i in range(per_n):
            _gen_one_file(n, seed_base + i, os.path.join(out_dir, f"{i+1:04d}_task.csv"),
                          task_cfg, es_nodes)
        total += per_n
        print(f"   ✅ {subdir}/N{n}/: {per_n}套  (seed {seed_base}~{seed_base+per_n-1})")
    print(f"   合计 {total} 套")


def main(only=None):
    cfg      = load_config("configs/data_config.yaml")
    data_dir = resolve_path(cfg["data_dir"])
    task_cfg = cfg["task"]
    es_nodes = [es for es_list in _ES_MOUNT.values() for es in es_list]

    print(f"{'='*60}")
    print(f"v4 数据生成  |  数据目录: {data_dir}")
    print(f"{'='*60}")

    if only is None or only == "train":
        gen_train(data_dir, cfg, task_cfg, es_nodes)
    if only is None or only == "val":
        gen_val(data_dir, cfg, task_cfg, es_nodes)
    if only is None or only == "bench":
        gen_benchmark(data_dir, cfg, task_cfg, es_nodes)

    print(f"\n{'='*60}")
    print(f"✅ v4 数据生成完毕，数据目录: {data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v4 数据生成（均匀随机，Phase1+Phase2 共用）")
    parser.add_argument("--only", choices=["train", "val", "bench"], default=None,
                        help="只生成某一组；省略则全部生成")
    args = parser.parse_args()
    main(args.only)
