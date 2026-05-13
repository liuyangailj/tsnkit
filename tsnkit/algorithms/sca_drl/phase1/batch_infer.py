import sys
import os
import glob
import time
import csv

import torch
import numpy as np
from sklearn.cluster import SpectralClustering

# 确保能找到项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from sca_drl.common.utils import load_config, resolve_path, get_device
from sca_drl.phase1.dataset import TSNPhase1Dataset
from sca_drl.phase1.model import GNNPartitionModel
from sca_drl.phase1.infer import compute_affinity_matrix

def batch_inference(config_path="configs/phase1.yaml"):
    print("=" * 60)
    print("🌉 Phase 1 -> Phase 2: 全量数据桥接工程启动")
    print("=" * 60)

    # 1. 加载配置与模型
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    infer_cfg = cfg["inference"]

    # 单一事实来源：从 data_config.yaml 覆盖数据路径
    from sca_drl.common.utils import load_config as _lc
    dc = _lc("configs/data_config.yaml")
    data_dir  = resolve_path(dc["data_dir"])
    topo_path = os.path.join(data_dir, "0_topo.csv")

    device = get_device()
    k_paths = data_cfg.get("k_paths", 3)
    n_clusters = infer_cfg.get("n_clusters", 8)
    model_tag = f"k{k_paths}_d{model_cfg['output_dim']}"
    print(f"📦 模型版本标签: {model_tag}  (文件后缀: _{model_tag}_emb.pt / _{model_tag}_group.csv)")

    # 初始化并加载最优模型
    model = GNNPartitionModel(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
        heads=model_cfg["heads"],
    ).to(device)
    
    # 🌟 自动寻找 best 模型
    ckpt_path = resolve_path(cfg["training"]["checkpoint_path"]).replace(".pth", "_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 找不到最优模型 {ckpt_path}，请确认 Phase 1 已经训练并保存！")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"✅ GNN 最优大脑已挂载: {os.path.basename(ckpt_path)}")

    # 2. 搜集所有的考卷 (train / val / benchmark)
    all_task_files = []
    for n_dir in sorted(glob.glob(os.path.join(data_dir, "train", "N*"))):
        all_task_files.extend(sorted(glob.glob(os.path.join(n_dir, "*_task.csv"))))
    all_task_files.extend(sorted(glob.glob(os.path.join(data_dir, "val", "*_task.csv"))))
    for n_dir in sorted(glob.glob(os.path.join(data_dir, "benchmark", "N*"))):
        all_task_files.extend(sorted(glob.glob(os.path.join(n_dir, "*_task.csv"))))
    
    print(f"📂 共发现 {len(all_task_files)} 份考卷需要打标，准备大批量前向推理...")
    time.sleep(1)

    # 3. 流水线作业开始
    for i, task_path in enumerate(all_task_files):
        base_name = os.path.basename(task_path).replace(".csv", "") # e.g., "1_task" or "val_1_task"
        dir_name = os.path.dirname(task_path)
        
        out_emb_path = os.path.join(dir_name, f"{base_name}_{model_tag}_emb.pt")
        out_group_path = os.path.join(dir_name, f"{base_name}_{model_tag}_group.csv")

        # 同版本已生成则跳过（不同 model_tag 生成不同文件名，互不干扰）
        if os.path.exists(out_emb_path) and os.path.exists(out_group_path):
            continue

        sys.stdout.write(f"\r⏳ 正在处理 [{i+1}/{len(all_task_files)}]: {base_name} ...")
        sys.stdout.flush()

        # [步骤 A] 构建图数据 (单张实时构建)
        # 注意：这里会产生寻路开销，但这只是离线生成一次，稍微等几分钟是值得的
        dataset = TSNPhase1Dataset(task_path, topo_path, k_paths=k_paths)
        data = dataset[0].to(device)

        # [步骤 B] GNN 瞬间算出高维 Embedding
        with torch.no_grad():
            embeddings = model(data).cpu().numpy()

        # [步骤 C] 谱聚类，计算分组
        W = compute_affinity_matrix(embeddings)
        sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42)
        labels = sc.fit_predict(W)

        # [步骤 D] 存盘 (写入硬盘，供 Phase 2 读取)
        # 1. 保存 Embedding (转成字典格式 sid -> tensor)
        emb_dict = {sid: embeddings[idx] for idx, sid in enumerate(data.stream_ids)}
        emb_dict_torch = {k: torch.tensor(v) for k, v in emb_dict.items()}
        torch.save(emb_dict_torch, out_emb_path)

        # 2. 保存分组 Group CSV
        with open(out_group_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["stream_id", "group_id"])
            for sid, gid in zip(data.stream_ids, labels):
                writer.writerow([sid, int(gid)])

    print("\n🎉 批量桥接数据生成完毕！你的 Phase 2 粮草已经全部堆满仓库！")
    print("=" * 60)

if __name__ == "__main__":
    batch_inference()