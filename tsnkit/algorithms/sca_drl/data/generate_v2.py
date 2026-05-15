"""v2 数据集生成：大流量重训用 train_v2 + val_v2。

Train: N∈{280,320,360,400,440,480}，60套/档，共360套
       种子按档连续分配：N=280→1000-1059，N=320→1060-1119，依次类推
Val:   N∈{320,400,480}，20套/档，共60套
       种子从5000起：N=320→5000-5019，N=400→5020-5039，N=480→5040-5059
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import random
import pandas as pd

from sca_drl.common.utils import load_config, resolve_path
from sca_drl.data.data_generater import (
    _SW_EDGES, _ES_MOUNT, _ES_TO_SW, _CORE_ES,
)

# ─── v2 数据集规格 ─────────────────────────────────────────────────────────────
TRAIN_N_POOL   = [280, 320, 360, 400, 440, 480]
TRAIN_PER_N    = 60
TRAIN_SEED_START = 1000   # N=280: 1000-1059, N=320: 1060-1119, ...

VAL_N_POOL     = [320, 400, 480]
VAL_PER_N      = 20
VAL_SEED_START = 5000     # N=320: 5000-5019, N=400: 5020-5039, N=480: 5040-5059


def _gen_one_file(flow_count, seed, out_path, task_cfg, es_nodes, edge_es_nodes):
    """用指定 seed 生成单个 task.csv。"""
    random.seed(seed)
    period_pool    = task_cfg["period_pool"]
    frame_size_min = task_cfg["frame_size_min"]
    frame_size_max = task_cfg["frame_size_max"]

    task_data = []
    for stream_id in range(flow_count):
        if random.random() < 0.8:
            if random.random() < 0.5:
                src = random.choice(edge_es_nodes)
                dst = random.choice(_CORE_ES)
            else:
                src = random.choice(_CORE_ES)
                dst = random.choice(edge_es_nodes)
        else:
            src = random.choice(edge_es_nodes)
            dst = random.choice(edge_es_nodes)

        while _ES_TO_SW.get(src) == _ES_TO_SW.get(dst) or src == dst:
            dst = random.choice(es_nodes)

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


def generate_v2(config_path="configs/data_config.yaml"):
    cfg      = load_config(config_path)
    data_dir = resolve_path(cfg["data_dir"])
    task_cfg = cfg["task"]

    es_nodes      = [es for es_list in _ES_MOUNT.values() for es in es_list]
    edge_es_nodes = [n for n in es_nodes if n not in _CORE_ES]

    generated = []

    # ── Train v2 ──────────────────────────────────────────────────────────────
    print("🏋️  生成 train_v2 数据集")
    for n_idx, n in enumerate(TRAIN_N_POOL):
        seed_base = TRAIN_SEED_START + n_idx * TRAIN_PER_N
        out_dir   = os.path.join(data_dir, "train_v2", f"N{n}")
        os.makedirs(out_dir, exist_ok=True)
        for i in range(TRAIN_PER_N):
            seed     = seed_base + i
            filename = f"{i + 1:04d}_task.csv"
            out_path = os.path.join(out_dir, filename)
            _gen_one_file(n, seed, out_path, task_cfg, es_nodes, edge_es_nodes)
            generated.append(out_path)
        print(f"   ✅ train_v2/N{n}/ : {TRAIN_PER_N} 套  (seed {seed_base}~{seed_base+TRAIN_PER_N-1})")

    # ── Val v2 ────────────────────────────────────────────────────────────────
    print("\n🧪  生成 val_v2 数据集")
    for n_idx, n in enumerate(VAL_N_POOL):
        seed_base = VAL_SEED_START + n_idx * VAL_PER_N
        out_dir   = os.path.join(data_dir, "val_v2", f"N{n}")
        os.makedirs(out_dir, exist_ok=True)
        for i in range(VAL_PER_N):
            seed     = seed_base + i
            filename = f"{i + 1:04d}_task.csv"
            out_path = os.path.join(out_dir, filename)
            _gen_one_file(n, seed, out_path, task_cfg, es_nodes, edge_es_nodes)
            generated.append(out_path)
        print(f"   ✅ val_v2/N{n}/   : {VAL_PER_N} 套  (seed {seed_base}~{seed_base+VAL_PER_N-1})")

    # ── 验收打印 ──────────────────────────────────────────────────────────────
    total_train = len(TRAIN_N_POOL) * TRAIN_PER_N
    total_val   = len(VAL_N_POOL)   * VAL_PER_N
    print(f"\n{'─'*60}")
    print(f"📋 生成文件清单（共 {len(generated)} 套）：")
    print(f"   train_v2: {total_train} 套  ({len(TRAIN_N_POOL)} 档 × {TRAIN_PER_N})")
    print(f"   val_v2  : {total_val}  套  ({len(VAL_N_POOL)} 档 × {VAL_PER_N})")
    print(f"\n📂 目录结构：")
    for split, n_pool in [("train_v2", TRAIN_N_POOL), ("val_v2", VAL_N_POOL)]:
        for n in n_pool:
            d = os.path.join(data_dir, split, f"N{n}")
            count = len([f for f in os.listdir(d) if f.endswith(".csv")])
            print(f"   {split}/N{n}/  →  {count} 个文件")
    print(f"\n✅ v2 数据集生成完毕！共 {len(generated)} 套。")


if __name__ == "__main__":
    generate_v2()
