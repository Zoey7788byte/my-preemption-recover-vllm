#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd


def parse_cond_dir(s: str):
    # Example: .../rc_cycle01_low_lam0p60_modepoisson[_sliceXX]
    m = re.search(r"rc_cycle(\d+)_([a-z]+)_lam([0-9p]+)_mode([a-z]+)", s)
    if not m:
        return None
    cycle = int(m.group(1))
    phase = m.group(2)
    lam_str = m.group(3).replace("p", ".")
    mode = m.group(4)
    try:
        lam = float(lam_str)
    except ValueError:
        lam = None
    return cycle, phase, lam, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input summary.csv")
    ap.add_argument(
        "--out",
        default=None,
        help="output aggregated csv (default: <input>_agg.csv)",
    )
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out) if args.out else inp.with_name(inp.stem + "_agg.csv")

    df = pd.read_csv(inp)

    parsed = df["cond_dir"].apply(parse_cond_dir)
    df["cycle"] = parsed.apply(lambda x: x[0] if x else None)
    df["phase"] = parsed.apply(lambda x: x[1] if x else None)
    df["lam_from_path"] = parsed.apply(lambda x: x[2] if x else None)
    df["mode"] = parsed.apply(lambda x: x[3] if x else None)

    # Prefer lambda_rps column; fallback to parsed value if missing.
    if "lambda_rps" not in df.columns or df["lambda_rps"].isna().all():
        df["lambda_rps"] = df["lam_from_path"]

    group_cols = ["cycle", "phase", "lambda_rps", "mode"]
    if df[group_cols].isna().any(axis=None):
        # If parsing failed, fall back to cond_dir grouping.
        group_cols = ["cond_dir"]

    agg_spec = {
        "N": ("N", "sum"),
        "ok_200": ("ok_200", "sum"),
        "total_rows": ("total_rows", "sum"),
        "wall_elapsed_s": ("wall_elapsed_s", "sum"),
        "ttft_p50_s": ("ttft_p50_s", "median"),
        "ttft_p90_s": ("ttft_p90_s", "median"),
        "ttft_p99_s": ("ttft_p99_s", "median"),
        "ttft_p999_s": ("ttft_p999_s", "median"),
        "ttft_max_s": ("ttft_max_s", "median"),
        "preempt_sum_delta": ("preempt_sum_delta", "sum"),
    }
    if "T_s" in df.columns:
        agg_spec["T_s"] = ("T_s", "sum")

    agg = df.groupby(group_cols).agg(**agg_spec)

    agg["ok_rate"] = agg["ok_200"] / agg["total_rows"]
    agg["completion_rate"] = agg["ok_200"] / agg["N"]
    agg["throughput_rps"] = agg["ok_200"] / agg["wall_elapsed_s"]
    if "T_s" in agg.columns:
        # Intended throughput is defined by planned injections (N/T_s), not completions.
        agg["throughput_intended_rps"] = agg["N"] / agg["T_s"]
    agg = agg.reset_index()

    agg.to_csv(out, index=False)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
