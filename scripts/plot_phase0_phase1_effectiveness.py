#!/usr/bin/env python3
"""Generate comparison plots for Phase0 vs Phase1 effectiveness."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def latest_file(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def f(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def read_summary(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def agg_summary(rows: List[Dict[str, str]]) -> Dict[float, Dict[str, float]]:
    g: Dict[float, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        g[f(r["lambda_rps"])].append(r)
    out: Dict[float, Dict[str, float]] = {}
    for lam, grp in sorted(g.items()):
        n = len(grp)
        out[lam] = {
            "ttft_p90_mean": sum(f(x["ttft_p90_s"]) for x in grp) / max(1, n),
            "ttft_p99_mean": sum(f(x["ttft_p99_s"]) for x in grp) / max(1, n),
            "preempt_sum_total": sum(f(x["preempt_sum_delta"]) for x in grp),
            "ok_rate_mean": sum(
                f(x.get("ok_200", "0")) / max(1.0, f(x.get("total_rows", "1"), 1.0))
                for x in grp
            ) / max(1, n),
        }
    return out


def read_events(path: Optional[str]) -> Tuple[List[int], Dict[str, int], Dict[str, int]]:
    # returns: swap_in blocks list, per-req swapin total, per-req swapout total
    k_done: List[int] = []
    in_total: Dict[str, int] = defaultdict(int)
    out_total: Dict[str, int] = defaultdict(int)
    if not path or not os.path.isfile(path):
        return k_done, in_total, out_total
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            req_id = obj.get("req_id")
            if req_id is None:
                req_id = f"seq:{obj.get('seq_id')}"
            req_id = str(req_id)
            event = obj.get("event", "")
            detail = obj.get("detail", {}) or {}
            blocks = int(detail.get("blocks_count", 0) or 0)
            if event == "SWAP_IN":
                if blocks > 0:
                    k_done.append(blocks)
                in_total[req_id] += blocks
            elif event == "SWAP_OUT":
                out_total[req_id] += blocks
    return k_done, in_total, out_total


def read_recovery_ts(path: Optional[str]) -> Dict[str, List[float]]:
    ts = {"t_rel_s": [], "preempt_cum": [], "swapin_cum": [], "swapout_cum": [], "stall_ms": []}
    if not path or not os.path.isfile(path):
        return ts
    pre = swi = swo = 0.0
    with open(path, "r", encoding="utf-8") as fp:
        rd = csv.DictReader(fp)
        for r in rd:
            pre += f(r.get("on_preempt_count_delta", "0"))
            swi += f(r.get("swapin_blocks", "0"))
            swo += f(r.get("swapout_blocks", "0"))
            ts["t_rel_s"].append(f(r.get("t_rel_s", "0")))
            ts["preempt_cum"].append(pre)
            ts["swapin_cum"].append(swi)
            ts["swapout_cum"].append(swo)
            ts["stall_ms"].append(f(r.get("restore_progress_stall_ms", "0")))
    return ts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-summary", default=None)
    ap.add_argument("--phase1-summary", default=None)
    ap.add_argument("--phase1-recovery-ts", default=None)
    ap.add_argument("--phase1-events", default=None)
    ap.add_argument("--out-dir", default="logs/RecoveryGen/Phase1/plots")
    args = ap.parse_args()

    p0_summary = args.phase0_summary or latest_file(
        "logs/RecoveryGen/Phase0/client_logs/*_pmode=*/summary.csv"
    )
    p1_summary = args.phase1_summary or latest_file(
        "logs/RecoveryGen/Phase1/client_logs/*_pmode=*/summary.csv"
    )
    p1_ts = args.phase1_recovery_ts or latest_file(
        "logs/RecoveryGen/Phase1/server_logs/*/recovery/recovery_ts.csv"
    )
    p1_events = args.phase1_events or latest_file(
        "logs/RecoveryGen/Phase1/server_logs/*/recovery/recovery_events.jsonl"
    )

    if not p0_summary or not os.path.isfile(p0_summary):
        raise SystemExit("Phase0 summary.csv not found.")
    if not p1_summary or not os.path.isfile(p1_summary):
        raise SystemExit("Phase1 summary.csv not found.")

    p0 = agg_summary(read_summary(p0_summary))
    p1 = agg_summary(read_summary(p1_summary))
    k_done, in_total, out_total = read_events(p1_events)
    tss = read_recovery_ts(p1_ts)

    # Lazy import matplotlib for environments that only want csv processing.
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)

    # Plot 1: summary compare
    lambdas = sorted(set(p0.keys()) | set(p1.keys()))
    x = list(range(len(lambdas)))
    labels = [f"{l:.2f}" for l in lambdas]

    def sget(d: Dict[float, Dict[str, float]], key: str) -> List[float]:
        return [d.get(l, {}).get(key, float("nan")) for l in lambdas]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(x, sget(p0, "ttft_p90_mean"), marker="o", label="Phase0")
    ax1.plot(x, sget(p1, "ttft_p90_mean"), marker="o", label="Phase1")
    ax1.set_title("TTFT p90 mean")
    ax1.set_xticks(x, labels)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(x, sget(p0, "ttft_p99_mean"), marker="o", label="Phase0")
    ax2.plot(x, sget(p1, "ttft_p99_mean"), marker="o", label="Phase1")
    ax2.set_title("TTFT p99 mean")
    ax2.set_xticks(x, labels)
    ax2.grid(alpha=0.3)

    ax3.plot(x, sget(p0, "preempt_sum_total"), marker="o", label="Phase0")
    ax3.plot(x, sget(p1, "preempt_sum_total"), marker="o", label="Phase1")
    ax3.set_title("preempt_sum_total")
    ax3.set_xticks(x, labels)
    ax3.grid(alpha=0.3)

    ax4.plot(x, sget(p0, "ok_rate_mean"), marker="o", label="Phase0")
    ax4.plot(x, sget(p1, "ok_rate_mean"), marker="o", label="Phase1")
    ax4.set_title("ok_rate_mean")
    ax4.set_xticks(x, labels)
    ax4.grid(alpha=0.3)

    fig.supxlabel("lambda_rps")
    fig.tight_layout()
    out1 = os.path.join(args.out_dir, "phase0_phase1_summary_compare.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)

    # Plot 2: micro swap-in distribution
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    ax21, ax22 = axes2

    if k_done:
        ax21.hist(k_done, bins=min(20, max(5, len(set(k_done)))), color="#3b82f6", alpha=0.8)
        ax21.axvline(sum(k_done) / len(k_done), color="red", linestyle="--", label="mean")
        ax21.set_title("Phase1 SWAP_IN blocks_count distribution")
        ax21.set_xlabel("k_done (blocks per SWAP_IN)")
        ax21.set_ylabel("count")
        ax21.legend()
        ax21.grid(alpha=0.3)
    else:
        ax21.text(0.5, 0.5, "No SWAP_IN events", ha="center", va="center")
        ax21.set_axis_off()

    reqs = sorted(set(in_total.keys()) | set(out_total.keys()))
    if reqs:
        out_vals = [out_total.get(r, 0) for r in reqs]
        in_vals = [in_total.get(r, 0) for r in reqs]
        ax22.scatter(out_vals, in_vals, alpha=0.8)
        mx = max(out_vals + in_vals + [1])
        ax22.plot([0, mx], [0, mx], linestyle="--", color="gray", label="y=x")
        ax22.set_title("Per-request swapin vs swapout blocks")
        ax22.set_xlabel("swapout total blocks")
        ax22.set_ylabel("swapin total blocks")
        ax22.grid(alpha=0.3)
        ax22.legend()
    else:
        ax22.text(0.5, 0.5, "No per-request swap stats", ha="center", va="center")
        ax22.set_axis_off()

    fig2.tight_layout()
    out2 = os.path.join(args.out_dir, "phase1_micro_swap_distribution.png")
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)

    # Plot 3: recovery timeline
    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax31, ax32 = axes3
    t = tss["t_rel_s"]
    if t:
        ax31.plot(t, tss["preempt_cum"], label="preempt_cum", linewidth=1.2)
        ax31.plot(t, tss["swapin_cum"], label="swapin_cum_blocks", linewidth=1.2)
        ax31.plot(t, tss["swapout_cum"], label="swapout_cum_blocks", linewidth=1.2)
        ax31.set_ylabel("cumulative")
        ax31.set_title("Phase1 recovery timeline (cumulative)")
        ax31.grid(alpha=0.3)
        ax31.legend()

        ax32.plot(t, tss["stall_ms"], label="restore_progress_stall_ms", linewidth=1.0)
        ax32.set_xlabel("t_rel_s")
        ax32.set_ylabel("ms")
        ax32.set_title("Phase1 restore stall proxy")
        ax32.grid(alpha=0.3)
        ax32.legend()
    else:
        ax31.text(0.5, 0.5, "No recovery_ts.csv data", ha="center", va="center")
        ax31.set_axis_off()
        ax32.set_axis_off()

    fig3.tight_layout()
    out3 = os.path.join(args.out_dir, "phase1_recovery_timeline.png")
    fig3.savefig(out3, dpi=150)
    plt.close(fig3)

    print(f"[OK] phase0 summary: {p0_summary}")
    print(f"[OK] phase1 summary: {p1_summary}")
    print(f"[OK] phase1 recovery_ts: {p1_ts}")
    print(f"[OK] phase1 events: {p1_events}")
    print(f"[OK] wrote: {out1}")
    print(f"[OK] wrote: {out2}")
    print(f"[OK] wrote: {out3}")


if __name__ == "__main__":
    main()
