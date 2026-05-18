"""benchmark_v2 数据集生成

小规模: N∈{40,60,80,100,150}，seed 7000~7029，30套/档
大规模: N∈{200,250,300,350,400,450,500}，seed 7100~7129，30套/档
总计: 12档 × 30套 = 360套

文件命名对齐 v2 风格：{i+1:04d}_task.csv（0001~0030）
输出目录：data/data_storm/fig10/benchmark_v2/N{n}/
流量参数继承 data_config.yaml 的 task.period_pool / frame_size 设置。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from sca_drl.common.utils import load_config, resolve_path
from sca_drl.data.generate_v2 import _gen_one_file
from sca_drl.data.data_generater import _ES_MOUNT, _CORE_ES


def generate_benchmark_v2(config_path="configs/data_config.yaml"):
    cfg      = load_config(config_path)
    data_dir = resolve_path(cfg["data_dir"])
    task_cfg = cfg["task"]
    bv2_cfg  = cfg["benchmark_v2"]

    es_nodes      = [es for es_list in _ES_MOUNT.values() for es in es_list]
    edge_es_nodes = [n for n in es_nodes if n not in _CORE_ES]

    subdir     = bv2_cfg["subdir"]
    per_n      = bv2_cfg["per_n"]
    small_cfg  = bv2_cfg["small"]
    large_cfg  = bv2_cfg["large"]

    generated = []

    groups = [
        ("小规模", small_cfg["n_pool"], small_cfg["seed_start"]),
        ("大规模", large_cfg["n_pool"], large_cfg["seed_start"]),
    ]

    for label, n_pool, seed_start in groups:
        print(f"\n📊  生成 {label} {subdir}")
        for n in n_pool:
            out_dir = os.path.join(data_dir, subdir, f"N{n}")
            os.makedirs(out_dir, exist_ok=True)
            for i in range(per_n):
                seed     = seed_start + i
                filename = f"{i + 1:04d}_task.csv"
                out_path = os.path.join(out_dir, filename)
                _gen_one_file(n, seed, out_path, task_cfg, es_nodes, edge_es_nodes)
                generated.append(out_path)
            print(f"   ✅ {subdir}/N{n}/: {per_n} 套  (seed {seed_start}~{seed_start + per_n - 1})")

    total_small = len(small_cfg["n_pool"]) * per_n
    total_large = len(large_cfg["n_pool"]) * per_n
    print(f"\n{'─' * 60}")
    print(f"📋 生成汇总（共 {len(generated)} 套）：")
    print(f"   小规模 {small_cfg['n_pool']}: {total_small} 套")
    print(f"   大规模 {large_cfg['n_pool']}: {total_large} 套")
    print(f"✅ benchmark_v2 数据集生成完毕！共 {len(generated)} 套")
    print(f"📂 数据目录: {os.path.join(data_dir, subdir)}")


if __name__ == "__main__":
    generate_benchmark_v2()
