#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def require_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")


def safe_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def to_markdown_table(df: pd.DataFrame, out_md: str, max_rows: int = 200):
    # 简单导出 markdown（避免依赖 tabulate）
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)

    with open(out_md, "w", encoding="utf-8") as f:
        # header
        f.write("| " + " | ".join(df2.columns) + " |\n")
        f.write("|" + "|".join(["---"] * len(df2.columns)) + "|\n")
        # rows
        for _, row in df2.iterrows():
            vals = []
            for v in row.tolist():
                if isinstance(v, float):
                    if math.isnan(v):
                        vals.append("")
                    else:
                        # 保守格式化
                        vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")


def plot_series(df: pd.DataFrame, x: str, y: str, out_png: str, title: str):
    if x not in df.columns or y not in df.columns:
        raise KeyError(f"Missing column(s) for plot: x={x}, y={y}")

    # 只画有效值
    d = df[[x, y]].dropna()
    if len(d) == 0:
        raise ValueError(f"No valid data for plot {y} vs {x}")

    plt.figure(figsize=(10, 3.2))
    plt.plot(d[x].to_numpy(), d[y].to_numpy())
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def add_phase_shading(ax, df: pd.DataFrame, x: str, phase_col: str = "phase"):
    if phase_col not in df.columns or x not in df.columns:
        return
    d = df[[x, phase_col]].dropna().copy()
    if len(d) == 0:
        return
    d = d.sort_values(x)

    def norm_phase(p):
        return str(p).strip().lower()

    phases = d[phase_col].map(norm_phase).tolist()
    xs = d[x].to_numpy()

    start = xs[0]
    cur = phases[0]
    for i in range(1, len(xs)):
        if phases[i] != cur:
            if cur == "low":
                ax.axvspan(start, xs[i], color="#e0e0e0", alpha=0.5, zorder=0)
            # high -> no shading (white)
            start = xs[i]
            cur = phases[i]
    # last segment
    if cur == "low":
        ax.axvspan(start, xs[-1], color="#e0e0e0", alpha=0.5, zorder=0)


def plot_series_with_phase(df: pd.DataFrame, x: str, y: str, out_png: str, title: str):
    if x not in df.columns or y not in df.columns:
        raise KeyError(f"Missing column(s) for plot: x={x}, y={y}")

    d = df[[x, y, "phase"]].dropna(subset=[x, y])
    if len(d) == 0:
        raise ValueError(f"No valid data for plot {y} vs {x}")

    fig, ax = plt.subplots(figsize=(10, 3.2))
    add_phase_shading(ax, d, x, "phase")
    ax.plot(d[x].to_numpy(), d[y].to_numpy(), zorder=2)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 mot2_make_artifacts.py <RUN_DIR>")
        print("Example: python3 mot2_make_artifacts.py logs/mot2/20260128_202336_low0.60_high0.85_poisson-poisson_b4")
        sys.exit(1)

    run_dir = sys.argv[1].rstrip("/")
    if not os.path.isdir(run_dir):
        raise NotADirectoryError(run_dir)

    # 输入文件（你的三个文件）
    f_all = os.path.join(run_dir, "metrics_timeseries_all.csv")
    f_cycle = os.path.join(run_dir, "proof2_cycle_summary.csv")
    f_phase = os.path.join(run_dir, "proof2_phase_summary.csv")

    require_file(f_all)
    require_file(f_cycle)
    require_file(f_phase)

    out_dir = os.path.join(run_dir, "artifacts_proof2")
    os.makedirs(out_dir, exist_ok=True)

    # ========== 1) 画图 ==========
    ts = pd.read_csv(f_all)
    # 期待列：t_rel_s, preempt_rate_per_s, req_waiting, gpu_cache_usage_perc（可选）
    ts = safe_numeric(ts, ["t_rel_s", "preempt_rate_per_s", "preempt_total",
                           "req_waiting", "req_running", "gpu_cache_usage_perc"])

    # 证明 Mot2 的两张核心图
    plot_series_with_phase(
        ts, "t_rel_s", "preempt_rate_per_s",
        os.path.join(out_dir, "fig_preempt_rate_vs_time.png"),
        "Mot2: preempt_rate_per_s vs t_rel_s"
    )
    plot_series_with_phase(
        ts, "t_rel_s", "req_waiting",
        os.path.join(out_dir, "fig_waiting_vs_time.png"),
        "Mot2: req_waiting vs t_rel_s"
    )

    # 可选图：缓存压力解释性图
    if "gpu_cache_usage_perc" in ts.columns:
        # 如果全是 0 或全 NaN，就跳过
        g = ts["gpu_cache_usage_perc"].dropna()
        if len(g) > 0 and (g.abs().max() > 1e-12):
            plot_series(
                ts, "t_rel_s", "gpu_cache_usage_perc",
                os.path.join(out_dir, "fig_gpu_cache_vs_time.png"),
                "Mot2: gpu_cache_usage_perc vs t_rel_s"
            )

    # ========== 2) 导出表格 ==========
    cyc = pd.read_csv(f_cycle)
    phs = pd.read_csv(f_phase)

    # 将你最常用的列放前面（不改变原始文件，只输出“用于写作”的表）
    cycle_cols = [c for c in [
        "cycle",
        "low_lambda_rps", "high_lambda_rps",
        "low_preempt_delta", "high_preempt_delta",
        "low_preempt_rate_avg", "high_preempt_rate_avg",
        "low_preempt_rate_max", "high_preempt_rate_max",
        "low_waiting_avg", "high_waiting_avg",
        "low_waiting_max", "high_waiting_max",
        "separation_ok"
    ] if c in cyc.columns]
    cyc_out = cyc[cycle_cols].copy()

    phase_cols = [c for c in [
        "cond_dir", "cycle", "phase", "mode", "lambda_rps", "T_s", "rows",
        "preempt_min", "preempt_max", "preempt_delta",
        "preempt_rate_avg", "preempt_rate_max",
        "waiting_avg", "waiting_max",
        "running_avg", "running_max",
        "gpu_avg", "gpu_max",
        "parse_nan_rows"
    ] if c in phs.columns]
    phs_out = phs[phase_cols].copy()

    # 输出 CSV（写作用）
    cycle_csv = os.path.join(out_dir, "table_cycle.csv")
    phase_csv = os.path.join(out_dir, "table_phase.csv")
    cyc_out.to_csv(cycle_csv, index=False)
    phs_out.to_csv(phase_csv, index=False)

    # 输出 Markdown（可直接贴到报告/论文草稿）
    to_markdown_table(cyc_out, os.path.join(out_dir, "table_cycle.md"))
    to_markdown_table(phs_out, os.path.join(out_dir, "table_phase.md"))

    # ========== 3) 额外：生成一个“high vs low 的聚合摘要”（可选但常用） ==========
    # 便于在文字里引用：high/low 的均值、中位数、max 等
    if "phase" in phs_out.columns:
        num_cols = [c for c in phs_out.columns if c not in ("cond_dir", "phase", "mode")]
        phs_num = phs_out.copy()
        for c in num_cols:
            phs_num[c] = pd.to_numeric(phs_num[c], errors="coerce")

        summary = phs_num.groupby("phase").agg({
            "preempt_delta": ["mean", "median", "max"],
            "preempt_rate_max": ["mean", "median", "max"],
            "waiting_max": ["mean", "median", "max"],
            "waiting_avg": ["mean", "median", "max"],
        })
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        summary.to_csv(os.path.join(out_dir, "table_phase_aggregate.csv"), index=False)

    print("[OK] Artifacts written to:", out_dir)
    print("  - Figures:")
    print("      fig_preempt_rate_vs_time.png")
    print("      fig_waiting_vs_time.png")
    print("      (optional) fig_gpu_cache_vs_time.png")
    print("  - Tables:")
    print("      table_cycle.csv / table_cycle.md")
    print("      table_phase.csv / table_phase.md")
    print("      (optional) table_phase_aggregate.csv")


if __name__ == "__main__":
    main()
