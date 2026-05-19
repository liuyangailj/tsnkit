"""Phase 1 GAT 自监督训练：多实例 batch 训练。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import glob
import time
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter # 用于训练日志记录

from sca_drl.common.utils import (
    set_seed, load_config, resolve_path, get_device, ensure_dir
)
from sca_drl.phase1.dataset import TSNPhase1Dataset, MultiInstanceDataset
from sca_drl.phase1.model import GNNPartitionModel

def compute_loss(embeddings, data, neg_sample_ratio=0.5):
    """计算单张图的自监督拓扑重构损失。

    Args:
        embeddings: 模型输出 [N, D]
        data: PyG Data（可能是 batch 中的一张图）
        neg_sample_ratio: 负采样比例

    Returns:
        torch.Tensor: 标量损失
    """
    device = embeddings.device
    num_nodes = embeddings.size(0)

    if data.edge_index.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    src, dst = data.edge_index
    e_ij = data.edge_attr.squeeze()

    # 正样本损失
    sim_pos = (F.cosine_similarity(embeddings[src], embeddings[dst]) + 1) / 2
    sim_pos = torch.clamp(sim_pos, 1e-7, 1 - 1e-7)
    loss_pos = F.binary_cross_entropy(sim_pos, e_ij)

    # 负采样损失
    num_neg = max(1, int(src.size(0) * neg_sample_ratio))
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=device)
    sim_neg = (F.cosine_similarity(embeddings[neg_src], embeddings[neg_dst]) + 1) / 2
    sim_neg = torch.clamp(sim_neg, 1e-7, 1 - 1e-7)
    loss_neg = F.binary_cross_entropy(sim_neg, torch.zeros_like(sim_neg))

    return loss_pos + loss_neg

def train_epoch(model, loader, optimizer, neg_sample_ratio, device):
    """一个 epoch：遍历 DataLoader 中所有 batch。

    Returns:
        float: epoch 平均损失
    """
    model.train()
    total_loss = 0.0
    num_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        embeddings = model(batch)

        # PyG batch 把多张图拼成一张大图，用 batch.batch 区分
        # 需要按图拆分计算损失
        loss = torch.tensor(0.0, device=device)
        graph_ids = batch.batch.unique()

        for gid in graph_ids:
            mask = batch.batch == gid
            node_emb = embeddings[mask]

            # 提取该子图的边
            node_indices = mask.nonzero(as_tuple=True)[0]
            node_min = node_indices.min()
            edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
            sub_edge_index = batch.edge_index[:, edge_mask] - node_min
            sub_edge_attr = batch.edge_attr[edge_mask]

            sub_data = type(batch)(
                edge_index=sub_edge_index,
                edge_attr=sub_edge_attr,
            )
            loss = loss + compute_loss(node_emb, sub_data, neg_sample_ratio)

        loss = loss / len(graph_ids)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(graph_ids)
        num_graphs += len(graph_ids)

    return total_loss / max(num_graphs, 1)

def resolve_task_files(data_cfg, key="task_pattern"):
    """从配置解析 task 文件列表，兼容单文件和多文件模式。"""
    if "task_file" in data_cfg and key == "task_pattern":
        return [resolve_path(data_cfg["task_file"])]

    task_dir = resolve_path(data_cfg["task_dir"])
    pattern = data_cfg.get("task_pattern", "*_task.csv")
    files = sorted(glob.glob(os.path.join(task_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {task_dir}")
    return files

def evaluate(model, loader, neg_sample_ratio, device):
    """验证集评估，不更新梯度。"""
    model.eval()
    total_loss = 0.0
    num_graphs = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embeddings = model(batch)
            graph_ids = batch.batch.unique()

            for gid in graph_ids:
                mask = batch.batch == gid
                node_emb = embeddings[mask]
                node_indices = mask.nonzero(as_tuple=True)[0]
                node_min = node_indices.min()
                edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
                sub_edge_index = batch.edge_index[:, edge_mask] - node_min
                sub_edge_attr = batch.edge_attr[edge_mask]

                sub_data = type(batch)(
                    edge_index=sub_edge_index,
                    edge_attr=sub_edge_attr,
                )
                total_loss += compute_loss(node_emb, sub_data, neg_sample_ratio).item()
                num_graphs += 1

    return total_loss / max(num_graphs, 1)

# ── 必须是 top-level 函数，Windows spawn 模式下才能 pickle ──────────────────
def _build_graph_worker(args):
    """构建单个冲突图并序列化为 *_conflict.pkl；失败时返回 (path, None)。"""
    task_path, topo_path, k_paths = args
    pkl_path = task_path.replace(".csv", "_conflict.pkl")
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


def _get_cache_path(files, cache_name, k_paths):
    """与 MultiInstanceDataset 保持一致的缓存路径计算逻辑。"""
    if not files:
        return None
    return os.path.join(os.path.dirname(files[0]), f"{cache_name}_{len(files)}_k{k_paths}.pt")


def _build_parallel_and_cache(files, topo_path, k_paths, cache_path, workers, label):
    """并行构建冲突图，结果组装为 MultiInstanceDataset 格式的 .pt 缓存文件。"""
    to_build, pkl_map = [], {}
    for f in files:
        pkl = f.replace(".csv", "_conflict.pkl")
        if os.path.exists(pkl):
            pkl_map[f] = pkl
        else:
            to_build.append(f)
    if pkl_map:
        print(f"   [{label}] 复用 {len(pkl_map)} 套上次中断的缓存")

    if to_build:
        print(f"   [{label}] 并行建图 ({workers} 进程) — {len(to_build)} 套待处理")
        t0 = time.time()
        total = len(to_build)
        executor = ProcessPoolExecutor(max_workers=workers)
        try:
            futures = {executor.submit(_build_graph_worker, (f, topo_path, k_paths)): f
                       for f in to_build}
            for i, future in enumerate(as_completed(futures), 1):
                task_path, pkl_path = future.result()
                pkl_map[task_path] = pkl_path
                elapsed = time.time() - t0
                eta = elapsed / i * (total - i) if i < total else 0
                eta_end = datetime.fromtimestamp(time.time() + eta)
                sys.stdout.write(
                    f"\r   [{i:>4}/{total}] {os.path.basename(task_path):<35}"
                    f" | {elapsed:6.1f}s | ETA {eta:6.1f}s ({eta_end.strftime('%H:%M:%S')})"
                )
                sys.stdout.flush()
        except KeyboardInterrupt:
            print(f"\n   [{label}] 收到中断，正在停止子进程...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            executor.shutdown(wait=False)
        print()

    # 按原始文件顺序组装图列表 → 写入缓存
    print(f"   [{label}] 组装 {len(files)} 张图 → {os.path.basename(cache_path)}")
    graphs, failed = [], 0
    for f in files:
        pkl_path = pkl_map.get(f)
        if pkl_path and os.path.exists(pkl_path):
            with open(pkl_path, "rb") as pf:
                graphs.append(pickle.load(pf))
            os.remove(pkl_path)
        else:
            failed += 1
            print(f"\n   ⚠️ 跳过失败文件: {os.path.basename(f)}")
    torch.save(graphs, cache_path)
    print(f"   [{label}] 缓存写入: {os.path.basename(cache_path)}"
          + (f"  (失败 {failed} 套)" if failed else ""))


def _make_loader(files, topo_path, k_paths, cache_name, batch_size, shuffle, workers, label):
    """构建 DataLoader：缓存缺失且 workers>1 时先并行建图，否则直接创建。"""
    if not files:
        return None, None
    cache_path = _get_cache_path(files, cache_name, k_paths)
    if cache_path and not os.path.exists(cache_path) and workers > 1:
        build_start = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{label}] 缓存缺失，启动并行建图 ...")
        _build_parallel_and_cache(files, topo_path, k_paths, cache_path, workers, label)
        print(f"   [{label}] 建图总耗时 {time.time() - build_start:.1f}s")
    t0 = time.time()
    dataset = MultiInstanceDataset(files, topo_path, k_paths, cache_name=cache_name)
    elapsed = time.time() - t0
    print(f"   [{label}] 数据集就绪: {len(dataset)} 张图，加载耗时 {elapsed:.1f}s")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        exclude_keys=["raw_paths"])
    return dataset, loader


def main(config: dict, workers: int = 1):
    """多实例 batch 训练主流程(含验证)。"""
    # 从 data_config.yaml 覆盖数据路径（单一事实来源）
    dc = load_config("configs/data_config.yaml")
    data_dir = resolve_path(dc["data_dir"])
    config["data"]["task_dir"] = data_dir
    config["data"]["topo_file"] = os.path.join(data_dir, "0_topo.csv")

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    topo_path = resolve_path(data_cfg["topo_file"])
    k_paths = data_cfg["k_paths"]

    # 训练集：train_v4/N*/ — 档位和套数由 data_config.yaml v4.train 决定
    train_subdir = dc.get("train_subdir_v4", "train_v4")
    train_files = []
    for n_dir in sorted(glob.glob(os.path.join(data_dir, train_subdir, "N*"))):
        train_files.extend(sorted(glob.glob(os.path.join(n_dir, "*_task.csv"))))
    if not train_files:
        raise FileNotFoundError(
            f"No training files found under {train_subdir}/N*/ in {data_dir}\n"
            f"请先运行: python data/generate_v4.py --only train"
        )
    batch_size = train_cfg.get("batch_size", 8)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 构建冲突图 (workers={workers}) ...")
    print(f"   训练集文件: {len(train_files)} 套 ({train_subdir}/N*/)")
    _, train_loader = _make_loader(
        train_files, topo_path, k_paths, "train_v4", batch_size, True, workers, "train"
    )

    # 验证集：val_v4/（扁平目录，与 generate_v4.py 输出结构一致）
    val_subdir = dc.get("val_subdir_v4", "val_v4")
    val_loader = None
    val_files = sorted(glob.glob(os.path.join(data_dir, val_subdir, "*_task.csv")))
    if val_files:
        print(f"   验证集文件: {len(val_files)} 套 ({val_subdir}/)")
        _, val_loader = _make_loader(
            val_files, topo_path, k_paths, "val_v4", batch_size, False, workers, "val"
        )
    else:
        print(f"⚠️  未找到验证集 {val_subdir}/，将跳过验证（Early Stop 不生效）")

    # 模型 & 优化器
    device = get_device()
    model = GNNPartitionModel(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
        heads=model_cfg["heads"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    n_epochs = train_cfg["n_epochs"]
    neg_ratio = train_cfg.get("neg_sample_ratio", 0.5)
    val_interval = train_cfg.get("val_interval", 10)
    patience = train_cfg.get("early_stopping_patience", 10)

    # ── 断点续训：检查 _resume.pth ──────────────────────────────────────────
    resume_path = resolve_path(train_cfg["checkpoint_path"].replace(".pth", "_resume.pth"))
    best_path   = resolve_path(train_cfg["checkpoint_path"].replace(".pth", "_best.pth"))
    start_epoch     = 1
    best_val_loss   = float("inf")
    patience_counter = 0

    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch      = ckpt["epoch"] + 1
        best_val_loss    = ckpt["best_val_loss"]
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"▶ 续训：从 epoch {start_epoch} 继续，best_val_loss={best_val_loss:.4f}，"
              f"patience={patience_counter}/{patience}")
    else:
        print("▶ 全新训练（无续训 checkpoint）")

    # 🌟 TensorBoard 记录器
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    run_dir = resolve_path(f"./runs/phase1/phase1_gnn_{current_time}")
    ensure_dir(run_dir)
    writer = SummaryWriter(log_dir=run_dir)
    print(f"📊 TensorBoard 日志已开启: tensorboard --logdir={run_dir}")

    print(f"Training: {n_epochs} epochs, {len(train_files)} graphs, "
          f"batch_size={batch_size}, val_interval={val_interval}")
    print("-" * 60)
    
    train_start_time = time.time()

    for epoch in range(start_epoch, n_epochs + 1):
        
        epoch_start_time = time.time()
        epoch_start_dt = datetime.now()
        
        train_loss = train_epoch(model, train_loader, optimizer, neg_ratio, device)
        
        epoch_elapsed = time.time() - epoch_start_time
        
        total_elapsed = time.time() - train_start_time
        finished_epochs = epoch - start_epoch + 1
        remaining_epochs = n_epochs - epoch
        avg_epoch_time = total_elapsed / max(finished_epochs, 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_end_time = datetime.fromtimestamp(time.time() + eta_seconds)

        writer.add_scalar("Loss/1_Train_Loss", train_loss, epoch)

        log = (
            f"[{epoch_start_dt.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Epoch {epoch:03d}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Epoch Time: {epoch_elapsed:.1f}s | "
            f"Elapsed: {total_elapsed / 60:.1f}min | "
            f"ETA: {eta_seconds / 60:.1f}min | "
            f"End: {eta_end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        stop_early = False
        if val_loader and epoch % val_interval == 0:
            val_start_time = time.time()
            val_loss = evaluate(model, val_loader, neg_ratio, device)
            val_elapsed = time.time() - val_start_time
            writer.add_scalar("Loss/2_Val_Loss", val_loss, epoch)
            log += f" | Val Loss: {val_loss:.4f} | Val Time: {val_elapsed:.1f}s"
            

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ensure_dir(os.path.dirname(best_path))
                # _best.pth 只存 model weights，供 batch_infer 加载
                torch.save(model.state_dict(), best_path)
                log += " ★ [Best Saved]"
            else:
                patience_counter += 1
                log += f" (patience {patience_counter}/{patience})"
                if patience_counter >= patience:
                    log += " → Early Stop!"
                    stop_early = True

        # 每个 epoch 保存完整 resume checkpoint（覆盖上一个），支持断点续训
        ensure_dir(os.path.dirname(resume_path))
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss":        best_val_loss,
            "patience_counter":     patience_counter,
        }, resume_path)

        print(log)
        if stop_early:
            break

    # 保存最终模型（model weights only，供推理）
    ckpt_path = resolve_path(train_cfg["checkpoint_path"])
    ensure_dir(os.path.dirname(ckpt_path))
    torch.save(model.state_dict(), ckpt_path)
    writer.close()

    print("-" * 65)
    print(f"Final model saved: {ckpt_path}")
    if val_loader:
        print(f"Best model (val_loss={best_val_loss:.4f}): {best_path}")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1 GAT Training")
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行建图进程数（默认1=串行；>1 时在缓存缺失时启用并行）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    main(cfg, workers=args.workers)