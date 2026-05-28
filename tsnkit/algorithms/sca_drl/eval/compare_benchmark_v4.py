"""benchmark_v4 四方法对比图。

对比方法
  1. LS                  — ls_baseline/ls_benchmark_v4_results.csv        (solve_time)
  2. SCA-DRL c4          — benchmark_v4/benchmark_v4_results.csv           (inference_time)
  3. SCA-DRL c4+Fallback — benchmark_v4/benchmark_v4_fallback_results.csv  (inference_time)
  4. SCA-DRL c4bal       — benchmark_v4_bal/benchmark_v4_results.csv       (inference_time)
  5. LS v2               — ls_baseline/ls_benchmark_v2_results.csv         (solve_time)
  6. SCA-DRL v2          — benchmark_v2/benchmark_v2_results.csv            (inference_time)
  7. Chen KSP5 v2        — chen_ksp5_baseline/chen_ksp5_benchmark_v2_nofb_results.csv (solve_time)
  8. Phase1+LS           — ablation/phase1_ls_benchmark_v4_results.csv      (solve_time)
  9. Rand+Phase2         — ablation/rand_phase2_benchmark_v4_results.csv    (inference_time)

对齐策略（P1）
  c4bal 只有每档 10 套；其余三方法有 30 套。
  从 c4bal 结果中读取实际采样的 instance_id，
  其余三方法筛选相同 instance_id，保证比较的是完全相同的实例集。
  v2 结果来自 benchmark_v2 数据集，按全量 30 套/档统计，仅作为额外参考曲线。

成功率定义
  该档位中 success == 1 的实例数 / 有效实例数（fallback 排除 integrity_ok==0 的行）。

用法
  cd tsnkit/algorithms/sca_drl
  python eval/compare_benchmark_v4.py
  python eval/compare_benchmark_v4.py --output results/figures/my_compare.png
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from sca_drl.common.utils import resolve_path


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载 & 对齐
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    p = resolve_path(path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"找不到结果文件: {p}")
    return pd.read_csv(p)


def _align_to_bal(df: pd.DataFrame, bal: pd.DataFrame) -> pd.DataFrame:
    """筛选 df 中与 bal 相同的 (N, instance_id) 行。"""
    keys = bal[["N", "instance_id"]].drop_duplicates()
    return df.merge(keys, on=["N", "instance_id"], how="inner")


def _success_rate(df: pd.DataFrame) -> pd.Series:
    """按 N 计算成功率，fallback 中 integrity_ok==0 的行先排除。"""
    d = df.copy()
    if "integrity_ok" in d.columns:
        d = d[d["integrity_ok"] == 1]
    d["success"] = pd.to_numeric(d["success"], errors="coerce")
    d = d.dropna(subset=["success"])
    return d.groupby("N")["success"].mean() * 100   # → %


def _avg_time(df: pd.DataFrame, col: str) -> pd.Series:
    """按 N 计算平均时间（秒）。"""
    d = df.copy()
    if "integrity_ok" in d.columns:
        d = d[d["integrity_ok"] == 1]
    d[col] = pd.to_numeric(d[col], errors="coerce")
    return d.groupby("N")[col].mean()


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main(output_path: str) -> None:
    # ── 读取原始数据 ──────────────────────────────────────────────────────────
    df_ls   = _load("results/ls_baseline/ls_benchmark_v4_results.csv")
    df_chen = _load("results/chen_ksp5_baseline/chen_ksp5_benchmark_v4_nofb_results.csv")
    df_c4   = _load("results/benchmark_v4/benchmark_v4_results.csv")
    df_fb   = _load("results/benchmark_v4/benchmark_v4_fallback_results.csv")
    df_bal  = _load("results/benchmark_v4_bal/benchmark_v4_results.csv")
    df_ls_v2 = _load("results/ls_baseline/ls_benchmark_v2_results.csv")
    df_v2    = _load("results/benchmark_v2/benchmark_v2_results.csv")
    df_chen_v2 = _load("results/chen_ksp5_baseline/chen_ksp5_benchmark_v2_nofb_results.csv")
    df_p1_ls = _load("results/ablation/phase1_ls_benchmark_v4_results.csv")
    df_rand_p2 = _load("results/ablation/rand_phase2_benchmark_v4_results.csv")

    # ── P1：对齐到 bal 的实例集 ───────────────────────────────────────────────
    df_ls_a   = _align_to_bal(df_ls,   df_bal)
    df_chen_a = _align_to_bal(df_chen, df_bal)
    df_c4_a   = _align_to_bal(df_c4,   df_bal)
    df_fb_a   = _align_to_bal(df_fb,   df_bal)
    df_p1_ls_a = _align_to_bal(df_p1_ls, df_bal)
    df_rand_p2_a = _align_to_bal(df_rand_p2, df_bal)
    # df_bal 本身已经是 10 套

    n_per_n = df_bal.groupby("N").size().iloc[0]
    n_total = df_bal["N"].nunique()
    print(f"对齐后：{n_total} 档 × {n_per_n} 套/档")
    print(f"各档 instance_id（示例 N={df_bal['N'].min()}）: "
          f"{sorted(df_bal[df_bal['N']==df_bal['N'].min()]['instance_id'].tolist())}")

    # ── 统计 ─────────────────────────────────────────────────────────────────
    sr_ls   = _success_rate(df_ls_a)
    sr_chen = _success_rate(df_chen_a)
    sr_c4   = _success_rate(df_c4_a)
    sr_fb   = _success_rate(df_fb_a)
    sr_bal  = _success_rate(df_bal)
    sr_ls_v2 = _success_rate(df_ls_v2)
    sr_v2    = _success_rate(df_v2)
    sr_chen_v2 = _success_rate(df_chen_v2)
    sr_p1_ls = _success_rate(df_p1_ls_a)
    sr_rand_p2 = _success_rate(df_rand_p2_a)

    t_ls   = _avg_time(df_ls_a,   "solve_time")
    t_chen = _avg_time(df_chen_a, "solve_time")
    t_c4   = _avg_time(df_c4_a,   "inference_time")
    t_fb   = _avg_time(df_fb_a,   "inference_time")
    t_bal  = _avg_time(df_bal,    "inference_time")
    t_ls_v2 = _avg_time(df_ls_v2, "solve_time")
    t_v2    = _avg_time(df_v2,    "inference_time")
    t_chen_v2 = _avg_time(df_chen_v2, "solve_time")
    t_p1_ls = _avg_time(df_p1_ls_a, "solve_time")
    t_rand_p2 = _avg_time(df_rand_p2_a, "inference_time")

    # ── 打印数字汇总 ──────────────────────────────────────────────────────────
    all_n = sorted(set(df_bal["N"].unique()) | set(df_ls_v2["N"].unique()) | set(df_v2["N"].unique()))
    header = (
        f"{'N':>5}  {'LS':>8}  {'Chen':>8}  {'c4':>8}  {'c4+FB':>8}  "
        f"{'c4bal':>8}  {'LS v2':>8}  {'v2':>8}  {'ChenV2':>8}  "
        f"{'P1+LS':>8}  {'RandP2':>8}"
    )
    print("\n-- Success Rate (%) " + "-" * 46)
    print(header)
    for n in all_n:
        print(f"{n:>5}  "
              f"{sr_ls.get(n, float('nan')):>7.1f}%  "
              f"{sr_chen.get(n, float('nan')):>7.1f}%  "
              f"{sr_c4.get(n, float('nan')):>7.1f}%  "
              f"{sr_fb.get(n, float('nan')):>7.1f}%  "
              f"{sr_bal.get(n, float('nan')):>7.1f}%  "
              f"{sr_ls_v2.get(n, float('nan')):>7.1f}%  "
              f"{sr_v2.get(n, float('nan')):>7.1f}%  "
              f"{sr_chen_v2.get(n, float('nan')):>7.1f}%  "
              f"{sr_p1_ls.get(n, float('nan')):>7.1f}%  "
              f"{sr_rand_p2.get(n, float('nan')):>7.1f}%")

    print("\n-- Avg Scheduling Time (s) " + "-" * 39)
    print(header)
    for n in all_n:
        print(f"{n:>5}  "
              f"{t_ls.get(n, float('nan')):>8.3f}  "
              f"{t_chen.get(n, float('nan')):>8.3f}  "
              f"{t_c4.get(n, float('nan')):>8.3f}  "
              f"{t_fb.get(n, float('nan')):>8.3f}  "
              f"{t_bal.get(n, float('nan')):>8.3f}  "
              f"{t_ls_v2.get(n, float('nan')):>8.3f}  "
              f"{t_v2.get(n, float('nan')):>8.3f}  "
              f"{t_chen_v2.get(n, float('nan')):>8.3f}  "
              f"{t_p1_ls.get(n, float('nan')):>8.3f}  "
              f"{t_rand_p2.get(n, float('nan')):>8.3f}")

    # ── 画图 ─────────────────────────────────────────────────────────────────
    METHODS = [
        ("LS",               sr_ls,   t_ls,   "dimgray",  "--",  "s"),
        ("Chen KSP5",        sr_chen, t_chen, "#9C27B0",  "--",  "P"),
        ("SCA-DRL c4",       sr_c4,   t_c4,   "#2196F3",  "-",   "o"),
        ("SCA-DRL c4+FB",    sr_fb,   t_fb,   "#4CAF50",  "-",   "^"),
        ("SCA-DRL c4bal",    sr_bal,  t_bal,  "#FF9800",  "-",   "D"),
        ("LS v2",            sr_ls_v2, t_ls_v2, "#795548", ":",   "v"),
        ("SCA-DRL v2",       sr_v2,   t_v2,   "#E91E63",  ":",   "X"),
        ("Chen KSP5 v2",     sr_chen_v2, t_chen_v2, "#673AB7", ":", "h"),
        ("Phase1+LS",        sr_p1_ls, t_p1_ls, "#607D8B", "-.",  "<"),
        ("Rand+Phase2",      sr_rand_p2, t_rand_p2, "#009688", "-.", ">"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.4))
    fig.suptitle(
        f"benchmark_v4 Method Comparison (v4 aligned n={n_per_n}/N; v2 full n=30/N)",
        fontsize=13, fontweight="bold"
    )

    # Left: success rate
    for label, sr, _, color, ls, marker in METHODS:
        xs = sorted(sr.index)
        ys = [sr.get(x, np.nan) for x in xs]
        ax1.plot(xs, ys, color=color, linestyle=ls,
                 marker=marker, markersize=6, linewidth=1.8, label=label)

    ax1.set_xlabel("N (# flows)")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Scheduling Success Rate")
    ax1.set_ylim(-5, 105)
    ax1.set_xticks(all_n)
    ax1.set_xticklabels([str(n) for n in all_n], rotation=45, fontsize=8)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Right: scheduling time (log)
    for label, _, t, color, ls, marker in METHODS:
        xs = sorted(t.index)
        ys = [t.get(x, np.nan) for x in xs]
        ax2.plot(xs, ys, color=color, linestyle=ls,
                 marker=marker, markersize=6, linewidth=1.8, label=label)

    ax2.set_xlabel("N (# flows)")
    ax2.set_ylabel("Avg Scheduling Time (s)")
    ax2.set_title("Scheduling Time (log scale)")
    ax2.set_yscale("log")
    ax2.set_xticks(all_n)
    ax2.set_xticklabels([str(n) for n in all_n], rotation=45, fontsize=8)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y:.3f}s" if y < 1 else f"{y:.1f}s"
    ))
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = resolve_path(output_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark_v4 四方法对比图")
    parser.add_argument("--output", default="results/figures/benchmark_v4_comparison.png")
    args = parser.parse_args()
    main(args.output)
