"""benchmark_v3 数据集生成：纯随机均匀采样 src/dst，贴近文献通用 benchmark 标准。

与 generate_benchmark_v2（核心节点偏置采样）的区别：
- 采样规则：全 31 个 ES 均匀随机，唯一约束 src ≠ dst 且不同 SW
- seed 段：每档独立，8000~8359（与 train_v2/val_v2/benchmark_v2 严格隔离）
- 档位/套数：与 v2 benchmark 一致（12档 × 30套 = 360套），便于对照
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import random
import pandas as pd

from sca_drl.common.utils import load_config, resolve_path
from sca_drl.data.data_generater import _ES_MOUNT, _ES_TO_SW


def _gen_one_file(flow_count, seed, out_path, task_cfg, es_nodes):
    random.seed(seed)
    period_pool    = task_cfg["period_pool"]
    frame_size_min = task_cfg["frame_size_min"]
    frame_size_max = task_cfg["frame_size_max"]

    task_data = []
    for stream_id in range(flow_count):
        src        = random.choice(es_nodes)
        src_sw     = _ES_TO_SW.get(src)
        dst_pool   = [n for n in es_nodes if n != src and _ES_TO_SW.get(n) != src_sw]
        dst        = random.choice(dst_pool)

        period = random.choice(period_pool)
        size   = random.randint(frame_size_min, frame_size_max)

        task_data.append({
            'stream':   stream_id,
            'src':      src,
            'dst':      f"[{dst}]",
            'size':     size,
            'period':   period,
            'deadline': period,
            'jitter':   period,
        })

    pd.DataFrame(task_data).to_csv(out_path, index=False)


def generate_benchmark_v3(config_path="configs/data_config.yaml"):
    cfg      = load_config(config_path)
    data_dir = resolve_path(cfg["data_dir"])
    task_cfg = cfg["task"]
    bv3_cfg  = cfg["benchmark_v3"]

    es_nodes    = [es for es_list in _ES_MOUNT.values() for es in es_list]
    subdir      = bv3_cfg["subdir"]
    per_n       = bv3_cfg["per_n"]
    n_pool      = bv3_cfg["n_pool"]
    seed_start  = bv3_cfg["seed_start"]

    print(f"📦 生成 {subdir}（纯随机均匀采样，共 {len(n_pool)} 档 × {per_n} 套）\n")

    generated = []
    for n_idx, n in enumerate(n_pool):
        seed_base = seed_start + n_idx * per_n
        out_dir   = os.path.join(data_dir, subdir, f"N{n}")
        os.makedirs(out_dir, exist_ok=True)
        for i in range(per_n):
            seed     = seed_base + i
            filename = f"{i + 1:04d}_task.csv"
            out_path = os.path.join(out_dir, filename)
            _gen_one_file(n, seed, out_path, task_cfg, es_nodes)
            generated.append(out_path)
        print(f"   ✅ {subdir}/N{n}/: {per_n} 套  (seed {seed_base}~{seed_base + per_n - 1})")

    print(f"\n{'─' * 60}")
    print(f"✅ {subdir} 生成完毕，共 {len(generated)} 套")
    print(f"📂 数据目录: {os.path.join(data_dir, subdir)}")


if __name__ == "__main__":
    generate_benchmark_v3()
