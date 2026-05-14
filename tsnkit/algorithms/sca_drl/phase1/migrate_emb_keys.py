"""一次性迁移脚本：将所有 _emb.pt 文件的 key 从 int 转为 str。

背景
----
batch_infer 之前保存 emb_dict 时 key 为 int（如 {0: tensor, 1: tensor}），
environment 查找时使用 str(int(f))（如 "0", "1"），类型不匹配导致 embedding 全部 miss。
修复后 batch_infer 改为 str key，但已有文件需要迁移。

运行方式
--------
cd tsnkit/algorithms/sca_drl
python phase1/migrate_emb_keys.py
"""

import sys
import os
import glob

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from sca_drl.common.utils import load_config, resolve_path


def migrate(data_dir: str, model_tag: str = "k5_d32") -> None:
    pattern = os.path.join(data_dir, "**", f"*_{model_tag}_emb.pt")
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        print(f"未找到任何 *_{model_tag}_emb.pt 文件，路径: {data_dir}")
        return

    print(f"共找到 {len(files)} 个 emb.pt 文件，开始迁移...\n")

    already_ok, converted, failed = 0, 0, 0

    for i, fpath in enumerate(files):
        try:
            d = torch.load(fpath, map_location="cpu", weights_only=False)

            if not d:
                already_ok += 1
                continue

            first_key = next(iter(d.keys()))

            # key 已经是 str，无需迁移
            if isinstance(first_key, str):
                already_ok += 1
                sys.stdout.write(f"\r  [{i+1}/{len(files)}] 已是 str key，跳过: {os.path.basename(fpath)}")
                sys.stdout.flush()
                continue

            # key 是 int，转换后覆盖写回
            d_new = {str(k): v for k, v in d.items()}
            torch.save(d_new, fpath)
            converted += 1
            sys.stdout.write(f"\r  [{i+1}/{len(files)}] 已转换: {os.path.basename(fpath)}")
            sys.stdout.flush()

        except Exception as e:
            print(f"\n  ⚠️ 失败: {fpath}\n     原因: {e}")
            failed += 1

    print(f"\n\n✅ 迁移完成")
    print(f"   已转换 (int→str): {converted}")
    print(f"   已跳过 (已是str): {already_ok}")
    if failed:
        print(f"   失败: {failed}")


if __name__ == "__main__":
    dc       = load_config("configs/data_config.yaml")
    data_dir = resolve_path(dc["data_dir"])
    print(f"数据目录: {data_dir}\n")
    migrate(data_dir)
