import sys
import os
import glob
import csv

import torch
import numpy as np
from sklearn.cluster import SpectralClustering

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from sca_drl.common.utils import load_config, resolve_path, get_device
from sca_drl.phase1.dataset import TSNPhase1Dataset, MultiInstanceDataset
from sca_drl.phase1.model import GNNPartitionModel
from sca_drl.phase1.infer import compute_affinity_matrix


def _collect_files(data_dir):
    """收集4组文件。

    train 按 N 值拆成两段：
      train_large (N>=150)：与 Phase1 训练集一致，可命中 train_300_k5.pt
      train_small (N<150) ：Phase2 课程学习需要，无现成缓存
    """
    train_small, train_large = [], []
    for n_dir in sorted(glob.glob(os.path.join(data_dir, "train", "N*"))):
        n_val = int(os.path.basename(n_dir)[1:])
        files = sorted(glob.glob(os.path.join(n_dir, "*_task.csv")))
        if n_val >= 150:
            train_large.extend(files)
        else:
            train_small.extend(files)

    val_files = sorted(glob.glob(os.path.join(data_dir, "val", "*_task.csv")))

    bench_files = []
    for n_dir in sorted(glob.glob(os.path.join(data_dir, "benchmark", "N*"))):
        bench_files.extend(sorted(glob.glob(os.path.join(n_dir, "*_task.csv"))))

    return train_large, train_small, val_files, bench_files


def _load_group(files, topo_path, k_paths, cache_name):
    """用 MultiInstanceDataset 加载一组文件（命中缓存则秒读）。"""
    if not files:
        return []
    ds = MultiInstanceDataset(files, topo_path, k_paths, cache_name=cache_name)
    return ds._graphs


def _already_done(task_path, model_tag):
    base    = os.path.basename(task_path).replace(".csv", "")
    dir_    = os.path.dirname(task_path)
    emb     = os.path.join(dir_, f"{base}_{model_tag}_emb.pt")
    grp     = os.path.join(dir_, f"{base}_{model_tag}_group.csv")
    return os.path.exists(emb) and os.path.exists(grp)


def _infer_and_save(data, task_path, model, n_clusters, model_tag, device):
    """对单张图执行 GNN 推理 + 谱聚类，结果立即写盘。"""
    base   = os.path.basename(task_path).replace(".csv", "")
    dir_   = os.path.dirname(task_path)
    out_emb   = os.path.join(dir_, f"{base}_{model_tag}_emb.pt")
    out_grp   = os.path.join(dir_, f"{base}_{model_tag}_group.csv")

    data = data.to(device)

    with torch.no_grad():
        embeddings = model(data).cpu().numpy()

    W      = compute_affinity_matrix(embeddings)
    sc     = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42)
    labels = sc.fit_predict(W)

    emb_dict = {sid: torch.tensor(embeddings[idx]) for idx, sid in enumerate(data.stream_ids)}
    torch.save(emb_dict, out_emb)

    with open(out_grp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stream_id", "group_id"])
        for sid, gid in zip(data.stream_ids, labels):
            writer.writerow([sid, int(gid)])


def batch_inference(config_path="configs/phase1.yaml"):
    print("=" * 60)
    print("🌉 Phase 1 -> Phase 2: 全量数据桥接工程启动")
    print("=" * 60)

    cfg       = load_config(config_path)
    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    infer_cfg = cfg["inference"]

    dc        = load_config("configs/data_config.yaml")
    data_dir  = resolve_path(dc["data_dir"])
    topo_path = os.path.join(data_dir, "0_topo.csv")

    device     = get_device()
    k_paths    = data_cfg.get("k_paths", 3)
    n_clusters = infer_cfg.get("n_clusters", 8)
    model_tag  = f"k{k_paths}_d{model_cfg['output_dim']}"
    print(f"📦 模型版本标签: {model_tag}")

    model = GNNPartitionModel(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
        heads=model_cfg["heads"],
    ).to(device)

    ckpt_path = resolve_path(cfg["training"]["checkpoint_path"]).replace(".pth", "_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 找不到最优模型 {ckpt_path}，请确认 Phase 1 已训练完成！")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"✅ GNN 模型已加载: {os.path.basename(ckpt_path)}\n")

    # ── 收集4组文件 ────────────────────────────────────────────────────────
    train_large, train_small, val_files, bench_files = _collect_files(data_dir)
    print(f"📂 文件统计:")
    print(f"   train_large (N>=150): {len(train_large)} 套  ← 命中缓存，秒读")
    print(f"   val               : {len(val_files):>4} 套  ← 命中缓存，秒读")
    print(f"   train_small (N<150): {len(train_small)} 套  ← 逐文件流式")
    print(f"   benchmark         : {len(bench_files):>4} 套  ← 逐文件流式")
    total = len(train_large) + len(val_files) + len(train_small) + len(bench_files)
    print(f"   合计: {total} 套\n")

    # ── 循环 A：有缓存的组（train_large + val），秒读后逐文件推理 ───────────
    print("🔨 步骤 1/3：加载有缓存的图数据（train_large + val）")
    large_graphs = _load_group(train_large, topo_path, k_paths, "train")
    val_graphs   = _load_group(val_files,   topo_path, k_paths, "val")

    cached_files  = train_large + val_files
    cached_graphs = large_graphs + val_graphs

    print(f"\n🚀 步骤 2/3：推理（有缓存组，共 {len(cached_files)} 套）")
    done_a, skip_a = 0, 0
    for i, (task_path, graph) in enumerate(zip(cached_files, cached_graphs)):
        if _already_done(task_path, model_tag):
            skip_a += 1
            continue
        sys.stdout.write(f"\r   [{i+1}/{len(cached_files)}] {os.path.basename(task_path)} ...")
        sys.stdout.flush()
        _infer_and_save(graph, task_path, model, n_clusters, model_tag, device)
        done_a += 1
    print(f"\r   完成 {done_a} 套，跳过(已有) {skip_a} 套" + " " * 30)

    # ── 循环 B：无缓存的组（train_small + bench），逐文件建图+推理+写盘 ─────
    stream_files = train_small + bench_files
    print(f"\n🚀 步骤 3/3：推理（流式组，共 {len(stream_files)} 套）— 逐文件建图+写盘")
    done_b, skip_b = 0, 0
    for i, task_path in enumerate(stream_files):
        if _already_done(task_path, model_tag):
            skip_b += 1
            continue
        sys.stdout.write(f"\r   [{i+1}/{len(stream_files)}] {os.path.basename(task_path)} ...")
        sys.stdout.flush()
        ds   = TSNPhase1Dataset(task_path, topo_path, k_paths=k_paths)
        _infer_and_save(ds[0], task_path, model, n_clusters, model_tag, device)
        done_b += 1
    print(f"\r   完成 {done_b} 套，跳过(已有) {skip_b} 套" + " " * 30)

    total_done = done_a + done_b
    total_skip = skip_a + skip_b
    print(f"\n🎉 全部完成！新生成 {total_done} 套，跳过(已有) {total_skip} 套")
    print("=" * 60)


if __name__ == "__main__":
    batch_inference()
