#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, csv, json, re
import numpy as np

TS_PAT = re.compile(r"^@@TS\s+([0-9.]+)\s*$")
PRE_PAT = re.compile(r'^vllm:num_preemptions_total(?:\{.*\})?\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*$')

def read_preempt_delta(metrics_raw: str) -> float:
    vals = []
    if not os.path.exists(metrics_raw):
        return 0.0
    cur_val = None
    cur_ts = None
    with open(metrics_raw, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = TS_PAT.match(line)
            if m:
                if cur_ts is not None and cur_val is not None:
                    vals.append(cur_val)
                cur_ts = float(m.group(1))
                cur_val = None
                continue
            mm = PRE_PAT.match(line.strip())
            if mm:
                try:
                    cur_val = float(mm.group(1))
                except Exception:
                    pass
    if cur_ts is not None and cur_val is not None:
        vals.append(cur_val)
    if len(vals) < 2:
        return 0.0
    return float(vals[-1] - vals[0])

def read_client_ttft(client_csv: str):
    ok_200 = 0
    total_rows = 0
    tt = []
    with open(client_csv, "r", encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for row in r:
            total_rows += 1
            if row.get("http_status", "") == "200":
                ok_200 += 1
            try:
                tt.append(float(row["ttft_s"]))
            except Exception:
                pass
    arr = np.array(tt, dtype=float) if tt else np.array([], dtype=float)

    def q(p: float) -> float:
        return float(np.quantile(arr, p)) if len(arr) else float("nan")

    ttft_p50 = q(0.50)
    ttft_p90 = q(0.90)
    ttft_p99 = q(0.99)
    ttft_p999 = q(0.999)
    ttft_max = float(arr.max()) if len(arr) else float("nan")
    return ok_200, total_rows, ttft_p50, ttft_p90, ttft_p99, ttft_p999, ttft_max

def main():
    if len(sys.argv) != 7:
        print("Usage: summarize_one_run_sweep_trace.py COND_DIR LAMBDA_RPS T_S WALL_ELAPSED_S META_JSON_PATH OUT_CSV_APPEND",
              file=sys.stderr)
        sys.exit(2)

    cond_dir = sys.argv[1]
    lam = float(sys.argv[2])
    T_s = float(sys.argv[3])
    wall = float(sys.argv[4])
    meta_json_path = sys.argv[5]
    out_csv = sys.argv[6]

    client_csv = os.path.join(cond_dir, "client_results.csv")
    metrics_raw = os.path.join(cond_dir, "metrics_raw.txt")

    if not os.path.exists(meta_json_path):
        raise FileNotFoundError(meta_json_path)
    meta = json.load(open(meta_json_path, "r"))
    N = int(meta.get("N", 0))

    ok_200, total_rows, p50, p90, p99, p999, mx = read_client_ttft(client_csv)
    pre_delta = read_preempt_delta(metrics_raw)

    row = [
        cond_dir,
        f"{lam:.6f}",
        f"{T_s:.3f}",
        str(N),
        str(ok_200),
        str(total_rows),
        f"{wall:.6f}",
        f"{p50:.6f}",
        f"{p90:.6f}",
        f"{p99:.6f}",
        f"{p999:.6f}",
        f"{mx:.6f}",
        f"{pre_delta:.6f}",
    ]

    # append row
    write_header = (not os.path.exists(out_csv)) or (os.path.getsize(out_csv) == 0)
    with open(out_csv, "a", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        if write_header:
            w.writerow([
                "cond_dir","lambda_rps","T_s","N","ok_200","total_rows","wall_elapsed_s",
                "ttft_p50_s","ttft_p90_s","ttft_p99_s","ttft_p999_s","ttft_max_s",
                "preempt_sum_delta"
            ])
        w.writerow(row)

    print("[OK] appended:", ",".join(row))

if __name__ == "__main__":
    main()
