import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import pandas as pd
import networkx as nx
import random

from sca_drl.common.utils import load_config, resolve_path

# ─── Fig.10 拓扑常量 ──────────────────────────────────────────────────────────
# 节点编号：SW1~SW15 → 0~14，ES1~ES31 → 15~45
# SW4(node 3) 和 SW8(node 7) 无挂载 ES

# SW-SW 骨干链路（0-based，原 1-based 编号已减 1）
_SW_EDGES = [
    (0, 2), (0, 12), (1, 2),  (1, 12),
    (2, 3), (2, 4),  (2, 13), (2, 14),
    (3, 4), (3, 11), (3, 12),
    (4, 5), (4, 7),  (4, 8),
    (5, 6), (6, 7),
    (7, 9), (7, 11), (8, 11),
    (9, 10),(10, 11),(11, 12),
    (12, 13),(12, 14),
]

# ES 挂载表：SW node → [ES node, ...]
_ES_MOUNT = {
    0:  [15, 16],
    1:  [17, 18, 19],
    2:  [20],
    4:  [21, 22, 23],
    5:  [24, 25],
    6:  [26, 27],
    8:  [28, 29],
    9:  [30, 31],
    10: [32, 33],
    11: [34, 35, 36],
    12: [37, 38],
    13: [39, 40],
    14: [41, 42, 43, 44, 45],
}

# 反向映射：ES node → 其所属 SW node
_ES_TO_SW = {es: sw for sw, es_list in _ES_MOUNT.items() for es in es_list}

# 高度节点 SW3(2)、SW5(4)、SW12(11) 挂载的 ES，作为"核心"流量集中区
_CORE_ES = [20, 21, 22, 23, 34, 35, 36]


def generate_industrial_dataset(config_path="configs/data_config.yaml"):
    cfg = load_config(config_path)

    data_dir = resolve_path(cfg["data_dir"])
    os.makedirs(data_dir, exist_ok=True)

    task_cfg        = cfg["task"]
    train_flow_pool = task_cfg["train_flow_pool"]
    train_per_n     = task_cfg["train_per_n"]
    val_flow_count  = task_cfg["val_flow_count"]
    val_count       = task_cfg["val_count"]
    bench_flow_pool = task_cfg["benchmark_flow_pool"]
    bench_per_n     = task_cfg["benchmark_per_n"]
    period_pool     = task_cfg["period_pool"]
    frame_size_min  = task_cfg["frame_size_min"]
    frame_size_max  = task_cfg["frame_size_max"]

    # =========================================================
    # 1. 生成 Fig.10 拓扑 (15 SW + 31 ES，共 46 节点)
    # =========================================================
    G = nx.Graph()
    G.add_edges_from(_SW_EDGES)

    es_nodes = []
    for sw_id, es_list in _ES_MOUNT.items():
        for es_id in es_list:
            G.add_edge(es_id, sw_id)
            es_nodes.append(es_id)

    G_dir = G.to_directed()

    topo_data = []
    for u, v in G_dir.edges():
        topo_data.append({
            'link':   f"({u}, {v})",
            'q_num':  8,
            'rate':   1,
            't_proc': 2000,
            't_prop': 0,
        })

    topo_path = os.path.join(data_dir, "0_topo.csv")
    pd.DataFrame(topo_data).to_csv(topo_path, index=False)
    print(f"✅ Fig.10 拓扑: 46 节点 (15 SW + 31 ES), {G_dir.number_of_edges()} 条有向边 → {topo_path}")

    edge_es_nodes = [n for n in es_nodes if n not in _CORE_ES]

    # =========================================================
    # 2. 内部辅助：生成固定流数的 n_files 份考卷到 out_dir
    # =========================================================
    def gen_tasks(flow_count, n_files, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for i in range(1, n_files + 1):
            task_data = []
            for stream_id in range(flow_count):
                # 80/20 法则：向核心节点聚集，制造空间拥塞
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

                # 过滤：同一 SW 下的 ES 不生成流（也过滤 src==dst）
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

            pd.DataFrame(task_data).to_csv(
                os.path.join(out_dir, f"{i:04d}_task.csv"), index=False
            )

    # =========================================================
    # 3. 训练集：11 档 × train_per_n 套/档
    # =========================================================
    print(f"\n🏋️  生成训练集: {len(train_flow_pool)} 档难度 × {train_per_n} 套/档 ...")
    for n in train_flow_pool:
        out = os.path.join(data_dir, "train", f"N{n}")
        gen_tasks(n, train_per_n, out)
        print(f"   ✅ train/N{n}/ : {train_per_n} 份")

    # =========================================================
    # 4. 验证集：固定 N=val_flow_count
    # =========================================================
    print(f"\n🧪  生成验证集: N={val_flow_count}, {val_count} 套 ...")
    gen_tasks(val_flow_count, val_count, os.path.join(data_dir, "val"))
    print(f"   ✅ val/ : {val_count} 份")

    # =========================================================
    # 5. Benchmark：7 档（含分布外档位）× bench_per_n 套
    # =========================================================
    print(f"\n📊  生成 benchmark: {len(bench_flow_pool)} 档 × {bench_per_n} 套/档 ...")
    for n in bench_flow_pool:
        out = os.path.join(data_dir, "benchmark", f"N{n}")
        gen_tasks(n, bench_per_n, out)
        print(f"   ✅ benchmark/N{n}/ : {bench_per_n} 份")

    total = len(train_flow_pool) * train_per_n + val_count + len(bench_flow_pool) * bench_per_n
    print(f"\n✅ 数据生成完毕！共 {total} 份考卷。")
    print(f"📂 数据目录: {data_dir}")


if __name__ == "__main__":
    generate_industrial_dataset()
