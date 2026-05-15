"""探针数据集生成：用于 Phase2 难度摸底测试。

N = {200, 250, 300, 350, 400, 500}，每档 10 套，共 60 套。
输出目录：data/fig10/probe/N{n}/
随机种子：每档文件 i 使用 seed = 9000 + i，与 train/val 不冲突。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import random
import pandas as pd

from sca_drl.common.utils import load_config, resolve_path

# 复用现有数据生成器的拓扑常量（保证与 train 数据同分布）
from sca_drl.data.data_generater import (
    _SW_EDGES, _ES_MOUNT, _ES_TO_SW, _CORE_ES,
)

# ─── 探针配置 ──────────────────────────────────────────────────────────────────
PROBE_N_POOL  = [200, 250, 300, 350, 400, 500]
PROBE_PER_N   = 10
SEED_START    = 9000   # 文件 i 使用 seed = SEED_START + i


def generate_probe_dataset(config_path="configs/data_config.yaml"):
    cfg      = load_config(config_path)
    data_dir = resolve_path(cfg["data_dir"])
    task_cfg = cfg["task"]

    period_pool    = task_cfg["period_pool"]
    frame_size_min = task_cfg["frame_size_min"]
    frame_size_max = task_cfg["frame_size_max"]

    # 所有 ES 节点列表
    es_nodes      = [es for es_list in _ES_MOUNT.values() for es in es_list]
    edge_es_nodes = [n for n in es_nodes if n not in _CORE_ES]

    generated = []

    print(f"🔬 探针数据集生成开始")
    print(f"   N 档位: {PROBE_N_POOL}")
    print(f"   每档: {PROBE_PER_N} 套，种子: {SEED_START} ~ {SEED_START + PROBE_PER_N - 1}")
    print(f"   输出根目录: {os.path.join(data_dir, 'probe')}\n")

    for n in PROBE_N_POOL:
        out_dir = os.path.join(data_dir, "probe", f"N{n}")
        os.makedirs(out_dir, exist_ok=True)

        for i in range(PROBE_PER_N):
            # 每个文件独立种子，确保可复现且与 train/val 不冲突
            random.seed(SEED_START + i)

            task_data = []
            for stream_id in range(n):
                # 80/20 法则（与 train 数据同分布）
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

            filename = f"{i + 1:04d}_task.csv"
            filepath = os.path.join(out_dir, filename)
            pd.DataFrame(task_data).to_csv(filepath, index=False)
            generated.append(filepath)

        print(f"   ✅ probe/N{n}/ : {PROBE_PER_N} 套 (seed {SEED_START}~{SEED_START + PROBE_PER_N - 1})")

    print(f"\n{'─'*60}")
    print(f"📋 生成文件清单（共 {len(generated)} 个）：")
    for fp in generated:
        rel = os.path.relpath(fp, resolve_path("."))
        print(f"   {rel}")

    print(f"\n✅ 探针数据集生成完毕！共 {len(generated)} 套考卷。")


if __name__ == "__main__":
    generate_probe_dataset()
