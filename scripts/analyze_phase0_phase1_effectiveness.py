#!/usr/bin/env python3
"""Compare Phase0/Phase1 outputs and validate Phase1 effectiveness gates.

Usage:
  python scripts/analyze_phase0_phase1_effectiveness.py

Optional explicit paths:
  python scripts/analyze_phase0_phase1_effectiveness.py \
    --phase0-summary <.../Phase0/.../summary.csv> \
    --phase1-summary <.../Phase1/.../summary.csv> \
    --phase1-recovery-ts <.../recovery/recovery_ts.csv> \
    --phase1-events <.../recovery/recovery_events.jsonl>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def latest_file(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def to_float(v: str, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    s = sorted(values)
    k = (len(s) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def read_summary(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary_stats(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    # keyed by lambda_rps string
    out: Dict[str, Dict[str, float]] = {}
    by_lam: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_lam.setdefault(r["lambda_rps"], []).append(r)
    for lam, grp in by_lam.items():
        ok_rates = [
            to_float(x.get("ok_200", "0")) / max(1.0, to_float(x.get("total_rows", "1")))
            for x in grp
        ]
        out[lam] = {
            "n_segments": float(len(grp)),
            "ttft_p90_mean": statistics.fmean(to_float(x["ttft_p90_s"]) for x in grp),
            "ttft_p99_mean": statistics.fmean(to_float(x["ttft_p99_s"]) for x in grp),
            "preempt_sum_mean": statistics.fmean(
                to_float(x["preempt_sum_delta"]) for x in grp
            ),
            "preempt_sum_total": sum(to_float(x["preempt_sum_delta"]) for x in grp),
            "ok_rate_mean": statistics.fmean(ok_rates),
        }
    return out


@dataclass
class ReqSwapStats:
    preempt_cnt: int = 0
    swap_out_blocks_total: int = 0
    swap_in_blocks_total: int = 0
    swap_out_events: List[Tuple[int, int]] = field(default_factory=list)  # (ts_ns, blocks)
    swap_in_events: List[Tuple[int, int]] = field(default_factory=list)  # (ts_ns, blocks)
    commit_events: List[Tuple[int, int, int]] = field(default_factory=list)  # (ts_ns, s, e)


def parse_commit_range(detail: dict) -> Optional[Tuple[int, int]]:
    # tolerant parsing: different versions may use different keys
    cand_pairs = [
        ("start_block", "end_block"),
        ("start", "end"),
        ("from", "to"),
        ("left", "right"),
    ]
    for a, b in cand_pairs:
        if a in detail and b in detail:
            try:
                s, e = int(detail[a]), int(detail[b])
                return (s, e)
            except Exception:
                pass
    if "restore_frontier" in detail:
        try:
            e = int(detail["restore_frontier"])
            return (0, e)
        except Exception:
            pass
    return None


def read_events(path: str) -> Tuple[Dict[str, ReqSwapStats], Dict[str, int]]:
    by_req: Dict[str, ReqSwapStats] = {}
    event_cnt: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                event_cnt["<bad_json>"] = event_cnt.get("<bad_json>", 0) + 1
                continue
            event = str(obj.get("event", ""))
            event_cnt[event] = event_cnt.get(event, 0) + 1
            req_id = obj.get("req_id")
            if req_id is None:
                seq_id = obj.get("seq_id")
                req_id = f"seq:{seq_id}"
            req_id = str(req_id)
            ts_ns = int(obj.get("ts_ns", 0))
            detail = obj.get("detail", {}) or {}
            st = by_req.setdefault(req_id, ReqSwapStats())

            if event == "PREEMPT_TRIGGERED":
                st.preempt_cnt += 1
            elif event == "SWAP_OUT":
                blocks = int(detail.get("blocks_count", 0) or 0)
                st.swap_out_blocks_total += blocks
                st.swap_out_events.append((ts_ns, blocks))
            elif event == "SWAP_IN":
                blocks = int(detail.get("blocks_count", 0) or 0)
                st.swap_in_blocks_total += blocks
                st.swap_in_events.append((ts_ns, blocks))
            elif "COMMIT" in event:
                rng = parse_commit_range(detail)
                if rng is not None:
                    st.commit_events.append((ts_ns, rng[0], rng[1]))
    return by_req, event_cnt


def read_recovery_ts(path: str) -> Dict[str, float]:
    rows = 0
    pre = 0
    swi = 0
    swo = 0
    stall_vals: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows += 1
            pre += int(to_float(r.get("on_preempt_count_delta", "0"), 0.0))
            swi += int(to_float(r.get("swapin_blocks", "0"), 0.0))
            swo += int(to_float(r.get("swapout_blocks", "0"), 0.0))
            stall_vals.append(to_float(r.get("restore_progress_stall_ms", "0"), 0.0))
    return {
        "rows": float(rows),
        "preempt_total": float(pre),
        "swapin_total": float(swi),
        "swapout_total": float(swo),
        "stall_max_ms": max(stall_vals) if stall_vals else float("nan"),
        "stall_p99_ms": percentile(stall_vals, 99) if stall_vals else float("nan"),
    }


def detect_budget(path: str, default_budget: int) -> int:
    m = re.search(r"rbudget=(\d+)", path)
    if m:
        return int(m.group(1))
    return default_budget


def gate_eval(
    by_req: Dict[str, ReqSwapStats],
    ts_stats: Optional[Dict[str, float]],
    budget: int,
    stall_gap_thr_ms: float,
    amplification_thr: float,
) -> Dict[str, dict]:
    reqs = [r for r in by_req.values() if r.swap_out_blocks_total > 0]

    # Gate 1: micro-task effective
    k_done = []
    swapout_sizes = []
    multi_step_acc = 0
    for st in reqs:
        swapout_sizes.append(st.swap_out_blocks_total)
        swapin_events = sorted(st.swap_in_events, key=lambda x: x[0])
        k_done.extend([b for _, b in swapin_events if b > 0])
        if len(swapin_events) >= 2 and st.swap_in_blocks_total >= 0.9 * st.swap_out_blocks_total:
            multi_step_acc += 1
    med_k = statistics.median(k_done) if k_done else float("nan")
    p90_k = percentile([float(x) for x in k_done], 90) if k_done else float("nan")
    med_out = statistics.median(swapout_sizes) if swapout_sizes else float("nan")
    ratio = (med_k / med_out) if (k_done and swapout_sizes and med_out > 0) else float("nan")
    gate1_pass = bool(
        reqs
        and k_done
        and med_out > 0
        and ratio < 0.5
        and p90_k <= max(32.0, float(2 * budget))
        and multi_step_acc >= 1
    )

    # Gate 2: continuation correctness (no long recovery stall / unfinished restore)
    longest_gap_ms = 0.0
    unfinished = 0
    for st in reqs:
        evs = sorted(st.swap_in_events, key=lambda x: x[0])
        if st.swap_in_blocks_total < 0.9 * st.swap_out_blocks_total:
            unfinished += 1
        for i in range(1, len(evs)):
            gap_ms = (evs[i][0] - evs[i - 1][0]) / 1e6
            if gap_ms > longest_gap_ms:
                longest_gap_ms = gap_ms
    ts_stall_max = ts_stats["stall_max_ms"] if ts_stats else float("nan")
    stall_ok = math.isnan(ts_stall_max) or ts_stall_max <= stall_gap_thr_ms
    gate2_pass = bool(reqs and unfinished == 0 and longest_gap_ms <= stall_gap_thr_ms and stall_ok)

    # Gate 3: repeated work controlled
    total_swi = sum(st.swap_in_blocks_total for st in reqs)
    total_swo = sum(st.swap_out_blocks_total for st in reqs)
    amplification = (float(total_swi) / float(total_swo)) if total_swo > 0 else float("nan")
    commit_present = any(st.commit_events for st in reqs)
    commit_violation = 0
    if commit_present:
        for st in reqs:
            if not st.commit_events:
                continue
            events = sorted(st.commit_events, key=lambda x: x[0])
            prev_s, prev_e = -1, -1
            for _, s, e in events:
                if s > e or s < prev_s or e < prev_e:
                    commit_violation += 1
                    break
                prev_s, prev_e = s, e
    commit_ok = None if not commit_present else (commit_violation == 0)
    gate3_pass = bool(
        reqs and total_swo > 0 and amplification <= amplification_thr and (commit_ok is None or commit_ok)
    )

    return {
        "gate1_micro_task_effective": {
            "pass": gate1_pass,
            "reqs_with_swapout": len(reqs),
            "median_k_done": med_k,
            "p90_k_done": p90_k,
            "median_swapout_blocks": med_out,
            "k_done_vs_swapout_ratio": ratio,
            "multi_step_accumulated_reqs": multi_step_acc,
        },
        "gate2_continuation_correctness": {
            "pass": gate2_pass,
            "unfinished_reqs": unfinished,
            "longest_gap_ms_between_swapins": longest_gap_ms,
            "restore_progress_stall_max_ms": ts_stall_max,
            "threshold_ms": stall_gap_thr_ms,
        },
        "gate3_redundant_work_controlled": {
            "pass": gate3_pass,
            "total_swapin_blocks": total_swi,
            "total_swapout_blocks": total_swo,
            "swapin_amplification_ratio": amplification,
            "amplification_threshold": amplification_thr,
            "commit_events_present": commit_present,
            "commit_monotonic": commit_ok,
            "commit_violation_count": commit_violation,
        },
    }


def print_summary_compare(p0_stats: Dict[str, Dict[str, float]], p1_stats: Dict[str, Dict[str, float]]) -> None:
    lams = sorted(set(p0_stats.keys()) | set(p1_stats.keys()), key=lambda x: float(x))
    print("=== Phase0 vs Phase1 Summary Compare ===")
    for lam in lams:
        s0 = p0_stats.get(lam)
        s1 = p1_stats.get(lam)
        if s0 is None or s1 is None:
            print(f"lambda={lam}: missing in one phase (phase0={s0 is not None}, phase1={s1 is not None})")
            continue
        print(
            f"lambda={lam} | "
            f"ttft_p90 {s0['ttft_p90_mean']:.3f}->{s1['ttft_p90_mean']:.3f} | "
            f"ttft_p99 {s0['ttft_p99_mean']:.3f}->{s1['ttft_p99_mean']:.3f} | "
            f"preempt_sum_total {s0['preempt_sum_total']:.1f}->{s1['preempt_sum_total']:.1f} | "
            f"ok_rate {s0['ok_rate_mean']:.4f}->{s1['ok_rate_mean']:.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0-summary", default=None)
    ap.add_argument("--phase1-summary", default=None)
    ap.add_argument("--phase1-recovery-ts", default=None)
    ap.add_argument("--phase1-events", default=None)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--stall-gap-threshold-ms", type=float, default=5000.0)
    ap.add_argument("--amplification-threshold", type=float, default=1.2)
    ap.add_argument("--json-out", default=None, help="optional output json path")
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

    phase0_rows = read_summary(p0_summary)
    phase1_rows = read_summary(p1_summary)
    p0_stats = summary_stats(phase0_rows)
    p1_stats = summary_stats(phase1_rows)
    print(f"Phase0 summary: {p0_summary}")
    print(f"Phase1 summary: {p1_summary}")
    print_summary_compare(p0_stats, p1_stats)

    ts_stats = None
    if p1_ts and os.path.isfile(p1_ts):
        ts_stats = read_recovery_ts(p1_ts)
        print(f"\nPhase1 recovery_ts: {p1_ts}")
        print(
            "recovery_ts totals: "
            f"rows={int(ts_stats['rows'])}, "
            f"preempt_total={int(ts_stats['preempt_total'])}, "
            f"swapin_total={int(ts_stats['swapin_total'])}, "
            f"swapout_total={int(ts_stats['swapout_total'])}, "
            f"stall_max_ms={ts_stats['stall_max_ms']:.3f}"
        )
    else:
        print("\nPhase1 recovery_ts: <missing>")

    event_cnt: Dict[str, int] = {}
    by_req: Dict[str, ReqSwapStats] = {}
    if p1_events and os.path.isfile(p1_events):
        by_req, event_cnt = read_events(p1_events)
        print(f"Phase1 events: {p1_events}")
        print("event counts:", ", ".join(f"{k}={v}" for k, v in sorted(event_cnt.items())))
    else:
        print("Phase1 events: <missing>")

    budget = detect_budget(p1_summary, args.budget)
    gates = gate_eval(
        by_req,
        ts_stats,
        budget=budget,
        stall_gap_thr_ms=args.stall_gap_threshold_ms,
        amplification_thr=args.amplification_threshold,
    )

    print("\n=== Phase1 Effectiveness Gates ===")
    g1 = gates["gate1_micro_task_effective"]
    print(
        f"[{'PASS' if g1['pass'] else 'FAIL'}] micro-task 生效 | "
        f"median_k_done={g1['median_k_done']}, p90_k_done={g1['p90_k_done']}, "
        f"median_swapout={g1['median_swapout_blocks']}, ratio={g1['k_done_vs_swapout_ratio']}, "
        f"multi_step_reqs={g1['multi_step_accumulated_reqs']}"
    )
    g2 = gates["gate2_continuation_correctness"]
    print(
        f"[{'PASS' if g2['pass'] else 'FAIL'}] 续作正确 | "
        f"unfinished_reqs={g2['unfinished_reqs']}, "
        f"longest_gap_ms={g2['longest_gap_ms_between_swapins']:.3f}, "
        f"stall_max_ms={g2['restore_progress_stall_max_ms']:.3f}, "
        f"threshold={g2['threshold_ms']:.1f}"
    )
    g3 = gates["gate3_redundant_work_controlled"]
    commit_status = (
        "UNKNOWN(no commit events)"
        if g3["commit_monotonic"] is None
        else ("PASS" if g3["commit_monotonic"] else "FAIL")
    )
    print(
        f"[{'PASS' if g3['pass'] else 'FAIL'}] 重复工作受控 | "
        f"swapin={g3['total_swapin_blocks']}, swapout={g3['total_swapout_blocks']}, "
        f"amp_ratio={g3['swapin_amplification_ratio']:.3f}, commit_monotonic={commit_status}"
    )

    all_pass = bool(g1["pass"] and g2["pass"] and g3["pass"])
    print(f"\nPhase1 gate result: {'PASS' if all_pass else 'FAIL'}")

    if args.json_out:
        payload = {
            "phase0_summary": p0_summary,
            "phase1_summary": p1_summary,
            "phase1_recovery_ts": p1_ts,
            "phase1_events": p1_events,
            "phase0_stats_by_lambda": p0_stats,
            "phase1_stats_by_lambda": p1_stats,
            "phase1_event_counts": event_cnt,
            "phase1_recovery_ts_stats": ts_stats,
            "gates": gates,
            "all_pass": all_pass,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"json report: {args.json_out}")


if __name__ == "__main__":
    main()
