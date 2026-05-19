import sys
import os
import glob
import csv
import time
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np
from sklearn.cluster import SpectralClustering

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from sca_drl.common.utils import load_config, resolve_path, get_device
from sca_drl.phase1.dataset import TSNPhase1Dataset, MultiInstanceDataset
from sca_drl.phase1.model import GNNPartitionModel
from sca_drl.phase1.infer import compute_affinity_matrix


# ── 必须是 top-level 函数，Windows spawn 模式下才能 pickle ──────────────────
def _build_graph_worker(args):
    """Worker: 构建 conflict graph 并序列化为 *_conflict.pkl。
    返回 (task_path, pkl_path)；失败时 pkl_path=None。"""
    task_path, topo_path, k_paths = args
    pkl_path = task_path.replace(".csv", "_conflict.pkl")
    # 静默子进程输出，防止与主进程进度条交错
    _orig_out, _orig_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(os.devnull, "w")
    try:
        ds = TSNPhase1Dataset(task_path, topo_path, k_paths=k_paths)
        with open(pkl_path, "wb") as f:
            pickle.dump(ds[0], f)
        return task_path, pkl_path
    except Exception:
        return task_path, None
    finally:
        sys.stdout.close()
        sys.stdout, sys.stderr = _orig_out, _orig_err


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


def _already_done(task_path, model_tag, n_clusters):
    base  = os.path.basename(task_path).replace(".csv", "")
    dir_  = os.path.dirname(task_path)
    emb   = os.path.join(dir_, f"{base}_{model_tag}_emb.pt")
    grp   = os.path.join(dir_, f"{base}_{model_tag}_c{n_clusters}_group.csv")
    paths = os.path.join(dir_, f"{base}_{model_tag}_paths.pkl")
    return os.path.exists(emb) and os.path.exists(grp) and os.path.exists(paths)


def _infer_and_save(data, task_path, model, n_clusters, model_tag, device):
    """对单张图执行 GNN 推理 + 谱聚类，结果立即写盘。"""
    base   = os.path.basename(task_path).replace(".csv", "")
    dir_   = os.path.dirname(task_path)
    out_emb   = os.path.join(dir_, f"{base}_{model_tag}_emb.pt")
    out_grp   = os.path.join(dir_, f"{base}_{model_tag}_c{n_clusters}_group.csv")

    data = data.to(device)

    with torch.no_grad():
        embeddings = model(data).cpu().numpy()

    W      = compute_affinity_matrix(embeddings)
    W      = (W + W.T) / 2   # 浮点误差可能导致微小非对称，显式对称化以消除 sklearn 警告
    sc     = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42)
    labels = sc.fit_predict(W)

    emb_dict = {str(sid): torch.tensor(embeddings[idx]) for idx, sid in enumerate(data.stream_ids)}
    torch.save(emb_dict, out_emb)

    with open(out_grp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stream_id", "group_id"])
        for sid, gid in zip(data.stream_ids, labels):
            writer.writerow([sid, int(gid)])

    # 同步写出 KSP 路径（节点序列），供 Phase2 env 直接加载，保证路径一致性
    out_paths = os.path.join(dir_, f"{base}_{model_tag}_paths.pkl")
    if hasattr(data, 'raw_paths') and data.raw_paths:
        with open(out_paths, "wb") as pf:
            pickle.dump(data.raw_paths, pf)
    else:
        print(f"\n   ⚠️ {os.path.basename(task_path)} 无 raw_paths（旧缓存？），跳过 paths.pkl")


def _collect_probe_files(data_dir):
    """收集 probe/N*/ 下的全部 task.csv（逐文件流式处理）。"""
    files = []
    for n_dir in sorted(glob.glob(os.path.join(data_dir, "probe", "N*"))):
        files.extend(sorted(glob.glob(os.path.join(n_dir, "*_task.csv"))))
    return files


def _collect_v2_files(data_dir, dc):
    """收集 train_v2/N*/ + val_v2/N*/ 下的全部 task.csv。"""
    train_subdir = dc.get("train_subdir_v2", "train_v2")
    val_subdir   = dc.get("val_subdir_v2",   "val_v2")
    files = []
    for n_dir in sorted(glob.glob(os.path.join(data_dir, train_subdir, "N*"))):
        files.extend(sorted(glob.glob(os.path.join(n_dir, "*_task.csv"))))
    for n_dir in sorted(glob.glob(os.path.join(data_dir, val_subdir, "N*"))):
        files.extend(sorted(glob.glob(os.path.join(n_dir, "*_task.csv"))))
    return files


def _collect_benchmark_files(data_dir, dc, key):
    """收集指定 benchmark 子目录（key: benchmark_v2 / benchmark_v3 / benchmark_v4）下的全部 task.csv。"""
    subdir = dc.get(key, {}).get("subdir", key)
    # v4 benchmark 配置嵌套在 v4.benchmark.subdir
    if not subdir or subdir == key:
        subdir = dc.get("v4", {}).get("benchmark", {}).get("subdir", key) if "v4" in key else subdir
    files = []
    for n_dir in sorted(glob.glob(os.path.join(data_dir, subdir, "N*"))):
        files.extend(sorted(glob.glob(os.path.join(n_dir, "*_task.csv"))))
    return files


def _collect_v4_files(data_dir, dc):
    """收集 train_v4/N*/ + val_v4/N*/ 下的全部 task.csv（Phase1 batch_infer 专用）。"""
    train_subdir = dc.get("train_subdir_v4", "train_v4")
    val_subdir   = dc.get("val_subdir_v4",   "val_v4")
    files = []
    for n_dir in sorted(glob.glob(os.path.join(data_dir, train_subdir, "N*"))):
        files.extend(sorted(glob.glob(os.path.join(n_dir, "*_task.csv"))))
    for n_dir in sorted(glob.glob(os.path.join(data_dir, val_subdir, "N*"))):
        files.extend(sorted(glob.glob(os.path.join(n_dir, "*_task.csv"))))
    return files


def _build_graphs_parallel(pending_files, topo_path, k_paths, workers):
    """阶段1：并行构建 conflict graph，返回 {task_path: pkl_path}。
    pkl_path=None 表示该文件构建失败。"""
    # 已有 pkl 的直接复用（上次中断留下的）
    to_build, result = [], {}
    for f in pending_files:
        pkl = f.replace(".csv", "_conflict.pkl")
        if os.path.exists(pkl):
            result[f] = pkl
        else:
            to_build.append(f)

    if len(result) > 0:
        print(f"   [阶段1] 复用 {len(result)} 套上次中断的 pkl")

    if not to_build:
        return result

    print(f"   [阶段1] 并行建图 ({workers} 进程) — 共 {len(to_build)} 套待处理")
    t0 = time.time()

    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(_build_graph_worker, (f, topo_path, k_paths)): f
                   for f in to_build}
        for i, future in enumerate(as_completed(futures), 1):
            task_path, pkl_path = future.result()
            result[task_path] = pkl_path
            elapsed = time.time() - t0
            eta = elapsed / i * (len(to_build) - i) if i < len(to_build) else 0
            sys.stdout.write(
                f"\r   [{i:>4}/{len(to_build)}] {os.path.basename(task_path):<35}"
                f" | {elapsed:5.0f}s | ETA {eta:5.0f}s"
            )
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n   [中断] 收到中断，正在停止子进程...")
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=False)

    print()
    return result


def _try_load_from_pt_cache(all_files, k_paths, cache_name):
    """尝试从 MultiInstanceDataset .pt 缓存直接加载图数据。
    返回 {task_path: Data} 或 None（缓存不存在/数量不匹配）。"""
    if not all_files or not cache_name:
        return None
    cache_path = os.path.join(
        os.path.dirname(all_files[0]),
        f"{cache_name}_{len(all_files)}_k{k_paths}.pt"
    )
    if not os.path.exists(cache_path):
        return None
    print(f"   ⚡ 命中 .pt 缓存: {os.path.basename(cache_path)}，直接加载...")
    graphs = torch.load(cache_path, weights_only=False)
    if len(graphs) != len(all_files):
        print(f"   ⚠️ 数量不匹配（缓存 {len(graphs)} != 文件 {len(all_files)}），忽略缓存")
        return None
    return {f: g for f, g in zip(all_files, graphs)}


def _run_two_stage(files, topo_path, k_paths, model, n_clusters, model_tag,
                   device, workers, label="", cache_name=None):
    """两阶段处理：①优先读 .pt 缓存（或并行建图）②串行 GAT+聚类+写盘。"""
    pending = [f for f in files if not _already_done(f, model_tag, n_clusters)]
    skipped = len(files) - len(pending)
    if skipped:
        print(f"   跳过(已有结果) {skipped} 套")
    if not pending:
        print(f"   {label}全部已完成，无需处理。")
        return 0, skipped

    # 阶段1：优先读 .pt 缓存，否则并行建 pkl
    graph_map = None  # {task_path: Data}
    pkl_map   = None  # {task_path: pkl_path}
    if cache_name:
        full_cache = _try_load_from_pt_cache(files, k_paths, cache_name)
        if full_cache is not None:
            graph_map = {f: full_cache[f] for f in pending if f in full_cache}
    if graph_map is None:
        pkl_map = _build_graphs_parallel(pending, topo_path, k_paths, workers)

    # 阶段2：串行 GAT 推理 + 谱聚类
    src = "来自 .pt 缓存" if graph_map is not None else "来自并行建图"
    print(f"   [阶段2] 串行 GAT+聚类（{src}）— 共 {len(pending)} 套")
    done, failed = 0, 0
    t0 = time.time()
    for i, task_path in enumerate(pending, 1):
        # 从缓存或 pkl 获取图
        if graph_map is not None:
            graph = graph_map.get(task_path)
        else:
            pkl_path = pkl_map.get(task_path)
            if pkl_path is None:
                failed += 1
                continue
            try:
                with open(pkl_path, "rb") as pf:
                    graph = pickle.load(pf)
                os.remove(pkl_path)
            except Exception as e:
                print(f"\n   ⚠️ 读取 pkl 失败 {os.path.basename(task_path)}: {e}")
                failed += 1
                continue

        if graph is None:
            failed += 1
            continue

        sys.stdout.write(
            f"\r   [{i:>4}/{len(pending)}] {os.path.basename(task_path):<35}"
            f" | {time.time()-t0:5.0f}s"
        )
        sys.stdout.flush()
        try:
            _infer_and_save(graph, task_path, model, n_clusters, model_tag, device)
            done += 1
        except Exception as e:
            print(f"\n   ⚠️ 推理失败 {os.path.basename(task_path)}: {e}")
            failed += 1

    print(f"\n   {label}完成 {done} 套，跳过(已有) {skipped} 套" +
          (f"，失败 {failed} 套" if failed else ""))
    return done, skipped


def batch_inference(config_path="configs/phase1.yaml", probe=False, version="v1", workers=2):
    print("=" * 60)
    if probe:
        print("🔬 Phase 1 -> Phase 2: 探针数据集桥接启动")
    elif version == "v2":
        print(f"🌉 Phase 1 -> Phase 2: v2 数据集桥接启动 (train_v2 + val_v2) [{workers} 进程]")
    elif version == "v4":
        print(f"🌉 Phase 1 -> Phase 2: v4 数据集桥接启动 (train_v4 + val_v4) [{workers} 进程]")
    elif version.startswith("benchmark_"):
        print(f"📊 Phase 1 -> Phase 2: {version} 桥接启动 [{workers} 进程]")
    else:
        print(f"🌉 Phase 1 -> Phase 2: 全量数据桥接工程启动 [{workers} 进程]")
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
    print(f"📦 模型版本标签: {model_tag}  n_clusters={n_clusters}")

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

    def _run_stream_files(files, label):
        """串行逐文件建图 + 推理 + 写盘（探针/小规模专用）。"""
        done, skip = 0, 0
        for i, task_path in enumerate(files):
            if _already_done(task_path, model_tag, n_clusters):
                skip += 1
                continue
            sys.stdout.write(f"\r   [{i+1}/{len(files)}] {os.path.basename(task_path)} ...")
            sys.stdout.flush()
            ds = TSNPhase1Dataset(task_path, topo_path, k_paths=k_paths)
            _infer_and_save(ds[0], task_path, model, n_clusters, model_tag, device)
            done += 1
        print(f"\r   {label}完成 {done} 套，跳过(已有) {skip} 套" + " " * 30)
        return done, skip

    if probe:
        # ── 探针模式：只处理 probe/N*/ ────────────────────────────────────────
        probe_files = _collect_probe_files(data_dir)
        print(f"📂 探针文件统计: {len(probe_files)} 套（probe/N*/）\n")
        print(f"🚀 推理（探针，共 {len(probe_files)} 套）— 串行逐文件")
        done, skip = _run_stream_files(probe_files, "探针")
        print(f"\n🎉 探针推理完成！新生成 {done} 套，跳过(已有) {skip} 套")
        print("=" * 60)
        return

    if version == "v2":
        # ── v2 模式：两阶段并行处理 train_v2/N*/ + val_v2/N*/ ───────────────
        v2_files = _collect_v2_files(data_dir, dc)
        print(f"📂 v2 文件统计: {len(v2_files)} 套（train_v2 + val_v2）\n")
        t_start = time.time()
        _run_two_stage(v2_files, topo_path, k_paths, model, n_clusters,
                       model_tag, device, workers, label="v2")
        total_elapsed = time.time() - t_start
        print(f"\n🎉 v2 推理完成！总耗时 {total_elapsed/60:.1f} 分钟")
        print("=" * 60)
        return

    if version in ("benchmark_v2", "benchmark_v3", "benchmark_v4"):
        # ── benchmark_v2 / v3 / v4 模式：两阶段并行处理对应子目录 ─────────────
        bfiles = _collect_benchmark_files(data_dir, dc, version)
        if not bfiles:
            print(f"❌ 未找到 {version} 数据，请先运行 data/generate_v4.py --only bench")
            return
        print(f"📂 {version} 文件统计: {len(bfiles)} 套\n")
        t_start = time.time()
        _run_two_stage(bfiles, topo_path, k_paths, model, n_clusters,
                       model_tag, device, workers, label=version)
        total_elapsed = time.time() - t_start
        print(f"\n🎉 {version} 推理完成！总耗时 {total_elapsed/60:.1f} 分钟")
        print("=" * 60)
        return

    if version == "v4":
        # ── v4 模式：train_v4/val_v4 分组处理，优先读 .pt 缓存 ──────────────────
        train_subdir = dc.get("train_subdir_v4", "train_v4")
        val_subdir   = dc.get("val_subdir_v4",   "val_v4")
        train_files_v4 = []
        for n_dir in sorted(glob.glob(os.path.join(data_dir, train_subdir, "N*"))):
            train_files_v4.extend(sorted(glob.glob(os.path.join(n_dir, "*_task.csv"))))
        val_files_v4 = sorted(glob.glob(os.path.join(data_dir, val_subdir, "*_task.csv")))
        if not train_files_v4 and not val_files_v4:
            print("❌ 未找到 v4 数据，请先运行: python data/generate_v4.py")
            return
        print(f"📂 v4 文件统计: train {len(train_files_v4)} 套 + val {len(val_files_v4)} 套 "
              f"= {len(train_files_v4)+len(val_files_v4)} 套\n")
        t_start = time.time()
        _run_two_stage(train_files_v4, topo_path, k_paths, model, n_clusters,
                       model_tag, device, workers, label="train_v4", cache_name="train_v4")
        _run_two_stage(val_files_v4, topo_path, k_paths, model, n_clusters,
                       model_tag, device, workers, label="val_v4", cache_name="val_v4")
        total_elapsed = time.time() - t_start
        print(f"\n🎉 v4 推理完成！总耗时 {total_elapsed/60:.1f} 分钟")
        print("=" * 60)
        return

    # ── 常规 v1 模式：收集4组文件 ────────────────────────────────────────────
    train_large, train_small, val_files, bench_files = _collect_files(data_dir)
    print(f"📂 文件统计:")
    print(f"   train_large (N>=150): {len(train_large)} 套  ← 命中缓存，秒读")
    print(f"   val               : {len(val_files):>4} 套  ← 命中缓存，秒读")
    print(f"   train_small (N<150): {len(train_small)} 套  ← 两阶段并行")
    print(f"   benchmark         : {len(bench_files):>4} 套  ← 两阶段并行")
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
        if _already_done(task_path, model_tag, n_clusters):
            skip_a += 1
            continue
        sys.stdout.write(f"\r   [{i+1}/{len(cached_files)}] {os.path.basename(task_path)} ...")
        sys.stdout.flush()
        _infer_and_save(graph, task_path, model, n_clusters, model_tag, device)
        done_a += 1
    print(f"\r   完成 {done_a} 套，跳过(已有) {skip_a} 套" + " " * 30)

    # ── 循环 B：无缓存的组（train_small + bench），两阶段并行 ────────────────
    stream_files = train_small + bench_files
    print(f"\n🚀 步骤 3/3：推理（流式组，共 {len(stream_files)} 套）— 两阶段并行")
    done_b, skip_b = _run_two_stage(stream_files, topo_path, k_paths, model,
                                     n_clusters, model_tag, device, workers, label="")

    total_done = done_a + done_b
    total_skip = skip_a + skip_b
    print(f"\n🎉 全部完成！新生成 {total_done} 套，跳过(已有) {total_skip} 套")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1 批量推理")
    parser.add_argument("--config",  default="configs/phase1.yaml")
    parser.add_argument("--probe",   action="store_true",
                        help="只推理 probe/N*/ 探针数据集（串行）")
    parser.add_argument("--version", default="v1",
                        choices=["v1", "v2", "v4",
                                 "benchmark_v2", "benchmark_v3", "benchmark_v4"],
                        help="v4: train_v4+val_v4（推荐）；benchmark_v4: 评测子目录")
    parser.add_argument("--workers", type=int, default=2,
                        help="阶段1并行建图进程数（默认2，probe模式无效）")
    args = parser.parse_args()
    batch_inference(args.config, probe=args.probe, version=args.version, workers=args.workers)
