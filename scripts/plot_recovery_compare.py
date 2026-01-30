#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def agg(path: Path):
    df = pd.read_csv(path)
    agg_spec = {
        "ttft_p99_s": ("ttft_p99_s", "median"),
        "ttft_p999_s": ("ttft_p999_s", "median"),
        "ttft_max_s": ("ttft_max_s", "median"),
        "preempt_sum_delta": ("preempt_sum_delta", "sum"),
        "ok_rate": ("ok_200", lambda s: s.sum() / df.loc[s.index, "total_rows"].sum()),
        "ok_200": ("ok_200", "sum"),
        "wall_elapsed_s": ("wall_elapsed_s", "sum"),
    }
    if "T_s" in df.columns:
        agg_spec["T_s"] = ("T_s", "sum")

    g = df.groupby("lambda_rps").agg(**agg_spec)
    g["throughput_rps"] = g["ok_200"] / g["wall_elapsed_s"]
    if "T_s" in g.columns and "N" in g.columns:
        g["throughput_intended_rps"] = g["N"] / g["T_s"]
        g["completion_rate"] = g["ok_200"] / g["N"]
    return g.sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="baseline summary.csv path")
    ap.add_argument("--a1", required=True, help="A1 (chunked) summary.csv path")
    ap.add_argument("--a2", required=False, help="A2 (gate) summary.csv path")
    ap.add_argument(
        "--out",
        default="logs/recovery_ctrl/plots/compare_ttft.png",
        help="output png path",
    )
    ap.add_argument(
        "--out-bar",
        default="logs/recovery_ctrl/plots/compare_bars.png",
        help="output png path for low/high bar chart",
    )
    args = ap.parse_args()

    base = agg(Path(args.baseline))
    a1 = agg(Path(args.a1))
    a2 = agg(Path(args.a2)) if args.a2 else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Lazy import so pandas-only environments still work for summary.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    series = [("baseline", base), ("A1", a1)]
    if a2 is not None:
        series.append(("A2", a2))

    for label, df in series:
        ax1.plot(df.index, df["ttft_p99_s"], marker="o", label=label)
        ax2.plot(df.index, df["ttft_p999_s"], marker="o", label=label)
        ax3.plot(df.index, df["ttft_max_s"], marker="o", label=label)
        ax4.plot(df.index, df["preempt_sum_delta"], marker="o", label=label)
        if "throughput_intended_rps" in df.columns:
            ax5.plot(
                df.index,
                df["throughput_intended_rps"],
                marker="o",
                label=f"{label} intended",
            )
        if "throughput_rps" in df.columns:
            ax5.plot(
                df.index,
                df["throughput_rps"],
                marker="o",
                linestyle="--",
                label=f"{label} real",
            )

    ax1.set_title("TTFT p99 (s)")
    ax2.set_title("TTFT p99.9 (s)")
    ax3.set_title("TTFT max (s)")
    ax4.set_title("preempt_sum_delta")
    ax5.set_title("throughput (rps)")
    ax6.axis("off")

    xticks = set(base.index).union(set(a1.index))
    if a2 is not None:
        xticks = xticks.union(set(a2.index))
    xticks = sorted(xticks)
    for ax in (ax1, ax2, ax3, ax4, ax5):
        ax.grid(True, alpha=0.3)
        ax.set_xticks(xticks)

    ax1.set_ylabel("seconds")
    ax2.set_ylabel("seconds")
    ax3.set_ylabel("seconds")
    ax4.set_ylabel("count")
    ax5.set_ylabel("rps")

    fig.supxlabel("lambda_rps")

    handles, labels = ax1.get_legend_handles_labels()
    handles5, labels5 = ax5.get_legend_handles_labels()
    handles += handles5
    labels += labels5
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)

    # ---- Optional low/high bar chart (TTFT + TPOT if available) ----
    out_bar = Path(args.out_bar)
    out_bar.parent.mkdir(parents=True, exist_ok=True)

    # Try to load TPOT columns if present in raw CSVs
    def load_raw(path: Path):
        return pd.read_csv(path)

    base_raw = load_raw(Path(args.baseline))
    a1_raw = load_raw(Path(args.a1))
    a2_raw = load_raw(Path(args.a2)) if args.a2 else None

    # Identify low/high lambdas
    lambdas = sorted(set(base.index).intersection(set(a1.index)))
    if a2 is not None:
        lambdas = sorted(set(lambdas).intersection(set(a2.index)))
    if len(lambdas) >= 2:
        low_lam, high_lam = lambdas[0], lambdas[-1]

        # Build bar data
        def pick(df, lam):
            sub = df[df["lambda_rps"] == lam]
            return {
                "ttft_p99_s": sub["ttft_p99_s"].median(),
                "ttft_p999_s": sub["ttft_p999_s"].median(),
                "tpot_p99_s": sub["tpot_p99_s"].median() if "tpot_p99_s" in sub else None,
                "tpot_p999_s": sub["tpot_p999_s"].median() if "tpot_p999_s" in sub else None,
            }

        base_low = pick(base_raw, low_lam)
        base_high = pick(base_raw, high_lam)
        a1_low = pick(a1_raw, low_lam)
        a1_high = pick(a1_raw, high_lam)
        a2_low = pick(a2_raw, low_lam) if a2_raw is not None else None
        a2_high = pick(a2_raw, high_lam) if a2_raw is not None else None

        # Prepare plots
        tpot_available = (
            "tpot_p99_s" in base_raw.columns
            and "tpot_p99_s" in a1_raw.columns
            and "tpot_p999_s" in base_raw.columns
            and "tpot_p999_s" in a1_raw.columns
            and (a2_raw is None or ("tpot_p99_s" in a2_raw.columns and "tpot_p999_s" in a2_raw.columns))
        )

        if tpot_available:
            fig2, axes2 = plt.subplots(2, 2, figsize=(9, 4.8), sharex=False)
            ax11, ax12, ax21, ax22 = axes2.flatten()
        else:
            fig2, axes2 = plt.subplots(1, 2, figsize=(9, 3.6), sharex=False)
            ax11, ax12 = axes2.flatten()
            ax21 = ax22 = None

        def bar_pair(ax, title, key):
            vals = [base_low.get(key), a1_low.get(key)]
            labels = [f"low {low_lam} base", f"low {low_lam} A1"]
            colors = ["#888", "#3a7"]
            if a2_low is not None:
                vals.append(a2_low.get(key))
                labels.append(f"low {low_lam} A2")
                colors.append("#f5a623")

            vals += [base_high.get(key), a1_high.get(key)]
            labels += [f"high {high_lam} base", f"high {high_lam} A1"]
            colors += ["#888", "#3a7"]
            if a2_high is not None:
                vals.append(a2_high.get(key))
                labels.append(f"high {high_lam} A2")
                colors.append("#f5a623")

            if any(v is None for v in vals):
                ax.set_visible(False)
                return
            ax.bar(labels, vals, color=colors)
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=15)
            ax.tick_params(axis="x", labelbottom=True)

        bar_pair(ax11, "TTFT p99 (s)", "ttft_p99_s")
        bar_pair(ax12, "TTFT p99.9 (s)", "ttft_p999_s")
        if tpot_available:
            bar_pair(ax21, "TPOT p99 (s)", "tpot_p99_s")
            bar_pair(ax22, "TPOT p99.9 (s)", "tpot_p999_s")

        # If some subplots are hidden, tighten layout to avoid large blanks.
        fig2.tight_layout()
        fig2.savefig(out_bar, dpi=150)

    # Print a small text summary
    print("== Baseline (median by lambda) ==")
    print(base.to_string())
    print("\n== A1 (median by lambda) ==")
    print(a1.to_string())
    if a2 is not None:
        print("\n== A2 (median by lambda) ==")
        print(a2.to_string())

    def print_delta(label, df):
        common = base.index.intersection(df.index)
        print(f"\n== Delta ({label} - baseline) ==")
        for lam in common:
            row = df.loc[lam] - base.loc[lam]
            print(
                f"lambda {lam:.2f}: p99 {row.ttft_p99_s:+.3f}s, "
                f"p999 {row.ttft_p999_s:+.3f}s, max {row.ttft_max_s:+.3f}s, "
                f"preempt_sum {row.preempt_sum_delta:+.0f}"
            )

    print_delta("A1", a1)
    if a2 is not None:
        print_delta("A2", a2)

    print(f"\n[OK] saved plot to {out_path}")
    if len(lambdas) >= 2:
        print(f"[OK] saved bar plot to {out_bar}")


if __name__ == "__main__":
    main()
