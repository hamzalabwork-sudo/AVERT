"""Per-claim computational overhead of AVERT over the deterministic decision layer (SGCC fleet, dev seed).

    python scripts/overhead.py --seed 310

Times only the decision/verification layer (the detector is shared by both defences).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avert.config import load_config  # noqa: E402
from avert.sgcc import COMM, VAL_END, W, SGCCCase, load_sgcc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=310)
    ap.add_argument("--data", default="data/sgcc_raw/data.csv")
    args = ap.parse_args()
    cfg = load_config("configs/default.yaml")
    case = SGCCCase(load_sgcc(args.data), cfg, args.seed)
    # precompute inputs once so only the defender is timed
    case.Xr = case.k0_report.copy()
    inputs = []
    for t in range(COMM + W, case.D):
        x = case.k0_report[:, t]
        case.Xr[:, t] = x
        case.peer = __import__("avert.sgcc", fromlist=["peer_stats"]).peer_stats(case.Xr, t)
        s = case.score(t, np.arange(case.A))
        K = case.K
        inputs.append((K - np.clip(x, 0, K), case.rate, K - np.clip(case.meas[:, t], 0, K), s,
                       np.zeros(case.A, bool), K - case.base[:, t]))
    t_det0 = time.perf_counter()
    for t in range(VAL_END, VAL_END + 50):
        case.score(t, np.arange(case.A))
    det_per_day = (time.perf_counter() - t_det0) / 50
    for name, params in (("DTA", None), ("AVERT", dict(cfg["avert"]))):
        D = case.make_defender(params)
        t0 = time.perf_counter()
        for inp in inputs:
            D.step(*inp)
        dt = time.perf_counter() - t0
        n = len(inputs) * case.A
        print(f"{name:6s}: {1e6 * dt / n:6.2f} us per claim ({len(inputs)} days x {case.A} customers)")
    print(f"detector: {1e6 * det_per_day / case.A:6.2f} us per claim")


if __name__ == "__main__":
    main()
