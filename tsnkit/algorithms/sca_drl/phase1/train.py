"""Phase 1 GAT 自监督训练：多实例 batch 训练。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import glob
import time
from datetime import datetime # 用于日志时间戳

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter # 用于训练日志记录

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

def main(config: dict):
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
    k_paths=data_cfg["k_paths"]
    
    # 训练集：train/N150/ ~ N200/ (6 档 × 50 套)
    _train_n = [150, 160, 170, 180, 190, 200]
    train_files = []
    for n in _train_n:
        train_files.extend(sorted(glob.glob(os.path.join(data_dir, "train", f"N{n}", "*_task.csv"))))
    if not train_files:
        raise FileNotFoundError(f"No training files found under train/N150~N200 in {data_dir}")
    train_dataset = MultiInstanceDataset(train_files, topo_path, k_paths, cache_name="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
    )
    print(f"Training set: {len(train_files)} graphs (N150~N200)")

    # 验证集：val/
    val_loader = None
    val_files = sorted(glob.glob(os.path.join(data_dir, "val", "*_task.csv")))
    if val_files:
        val_dataset = MultiInstanceDataset(val_files, topo_path, k_paths, cache_name="val")
        val_loader = DataLoader(val_dataset, batch_size=train_cfg.get("batch_size", 8))
        print(f"Validation set: {len(val_files)} graphs")

    # 模型 & 优化器
    device = get_device()
    model = GNNPartitionModel(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
        heads=model_cfg["heads"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    
    # 🌟 3. 初始化 TensorBoard MLOps 记录器
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    run_dir = resolve_path(f"./runs/phase1/phase1_gnn_{current_time}")
    ensure_dir(run_dir)
    writer = SummaryWriter(log_dir=run_dir)
    print(f"📊 TensorBoard 日志已开启: tensorboard --logdir={run_dir}")

    n_epochs = train_cfg["n_epochs"]
    neg_ratio = train_cfg.get("neg_sample_ratio", 0.5)
    val_interval = train_cfg.get("val_interval", 10)
    patience = train_cfg.get("early_stopping_patience", 10)
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Training: {n_epochs} epochs, {len(train_dataset)} graphs, "
          f"batch_size={train_cfg.get('batch_size', 8)}, val_interval={val_interval}")
    print("-" * 60)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, neg_ratio, device)
        elapsed = time.time() - t0

        writer.add_scalar("Loss/1_Train_Loss", train_loss, epoch)
        log = f"Epoch {epoch:03d}/{n_epochs} | Train Loss: {train_loss:.4f} | {elapsed:.1f}s"

        stop_early = False
        if val_loader and epoch % val_interval == 0:
            val_loss = evaluate(model, val_loader, neg_ratio, device)
            writer.add_scalar("Loss/2_Val_Loss", val_loss, epoch)
            log += f" | Val Loss: {val_loss:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_path = resolve_path(
                    train_cfg["checkpoint_path"].replace(".pth", "_best.pth")
                )
                ensure_dir(os.path.dirname(best_path))
                torch.save(model.state_dict(), best_path)
                log += " ★ [Best Saved]"
            else:
                patience_counter += 1
                log += f" (patience {patience_counter}/{patience})"
                if patience_counter >= patience:
                    log += " → Early Stop!"
                    stop_early = True

        print(log)
        if stop_early:
            break
        
    # 保存最终模型
    ckpt_path = resolve_path(train_cfg["checkpoint_path"])
    ensure_dir(os.path.dirname(ckpt_path))
    torch.save(model.state_dict(), ckpt_path)
    writer.close() # 关闭 TensorBoard 记录器
    
    print("-" * 65)
    print(f"Final model saved: {ckpt_path}")
    if val_loader:
        print(f"Best model (val_loss={best_val_loss:.4f}): {best_path}")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1 GAT Training")
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    main(cfg)