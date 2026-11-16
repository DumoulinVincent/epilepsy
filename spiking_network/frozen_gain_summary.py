"""Summarise frozen-HSP trials and sweeps (frozen_gain.py).

    python frozen_gain_summary.py results/frozen_gain/pilot

Per trial: low-state rate before any escape, time to first high-rate bin (right-censored at T), whether a high-rate
state, once entered, returned below 5 Hz for 10 s (self-termination), and the gain drift read back at the end
(checks the freeze). Per sweep: mean rate in the last half of each hold, ascending vs descending.
"""
import json
import sys
from pathlib import Path

import numpy as np

HIGH_HZ, LOW_HZ, RETURN_S = 10.0, 5.0, 10


def trial_stats(rE, kick):
    r = rE[kick:]
    hi = np.flatnonzero(r > HIGH_HZ)
    first = int(hi[0]) if len(hi) else None
    pre = r[: first if first is not None else len(r)]
    returned = None
    if first is not None:
        after = r[first:] < LOW_HZ
        run = np.convolve(after.astype(int), np.ones(RETURN_S, int), "valid")
        returned = bool((run == RETURN_S).any())
    return dict(first_escape_s=first, low_rate=float(pre[5:].mean()) if len(pre) > 5 else float("nan"),
                frac_high=float((r > HIGH_HZ).mean()), returned=returned, last20_rate=float(r[-20:].mean()))


def main(d):
    d = Path(d)
    rows = []
    for f in sorted(d.glob("trial_*.npz")):
        z = np.load(f); m = json.loads(str(z["meta"]))
        rows.append(dict(m, **trial_stats(z["rate_E"], m["kick_bins"])))
    print(f"{'kA':>5} {'init':>4} {'seed':>4} {'low rate':>8} {'1st esc':>8} {'frac hi':>7} {'returned':>8} "
          f"{'last20':>7} {'drift kA':>9}")
    for r in sorted(rows, key=lambda r: (r["kA"], r["init"], r["noise_seed"])):
        print(f"{r['kA']:5.2f} {r['init']:>4} {r['noise_seed']:4d} {r['low_rate']:8.2f} "
              f"{str(r['first_escape_s']):>8} {r['frac_high']:7.2f} {str(r['returned']):>8} {r['last20_rate']:7.1f} "
              f"{r['kA_drift']:9.1e}")
    for f in sorted(d.glob("sweep_*.npz")):
        z = np.load(f); m = json.loads(str(z["meta"]))
        rE, h, ks = z["rate_E"], int(m["hold_s"]), m["kAs"]
        n = len(ks) // 2
        print(f"\n{f.name}: kA  up-rate  down-rate  (mean of last half of each {h} s hold)")
        seg = lambda i: rE[i * h + h // 2:(i + 1) * h].mean()  # noqa: E731
        for i in range(n):
            j = len(ks) - 1 - i
            print(f"  {ks[i]:.2f}  {seg(i):8.2f}  {seg(j):8.2f}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results/frozen_gain/pilot")
