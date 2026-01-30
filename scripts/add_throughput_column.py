#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def add_throughput(path: Path, out: Path | None):
    df = pd.read_csv(path)
    if "ok_200" not in df.columns or "wall_elapsed_s" not in df.columns:
        raise ValueError(f"{path} missing ok_200 or wall_elapsed_s")
    df["throughput_rps"] = df["ok_200"] / df["wall_elapsed_s"]
    if "N" in df.columns:
        df["completion_rate"] = df["ok_200"] / df["N"]
    if "T_s" in df.columns and "N" in df.columns:
        # Intended throughput uses planned injections (N/T_s), not completions.
        df["throughput_intended_rps"] = df["N"] / df["T_s"]
    if out is None:
        df.to_csv(path, index=False)
        print(f"[OK] updated {path}")
    else:
        df.to_csv(out, index=False)
        print(f"[OK] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="summary.csv paths")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="optional output dir to write *_with_throughput.csv",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for inp in args.inputs:
        p = Path(inp)
        if out_dir:
            out = out_dir / f"{p.stem}_with_throughput{p.suffix}"
        else:
            out = None
        add_throughput(p, out)


if __name__ == "__main__":
    main()
