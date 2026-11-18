"""Input-rate sweep (reproduce_all.sh stage sweep): power spectrum of the slow homeostatic dynamics against the
external input rate, one network, all homeostatic time constants 50 s, 5000 s per rate.

    python input_sweep_figure.py [results/input_sweep]     # -> <dir>/input_sweep.png

For each rate: Welch power spectrum of the E-mean gain k_AMPA (sampled every 10 s, first 500 s discarded); a
cycle shows as a low-frequency peak.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(d):
    d = Path(d)
    rows = []
    for p in sorted((d / "raw").glob("*.npz")):
        z = np.load(p); m = json.loads(str(z["meta"]))
        t, k = z["t_k"], z["k_AMPA"]
        k = k[t >= 500]
        f, P = welch(k - k.mean(), fs=0.1, nperseg=min(256, len(k)))
        rows.append((m["freq_hz"], f, P))
    if not rows:
        print("no runs in", d / "raw"); return
    rows.sort(key=lambda r: r[0])
    fin = np.array([r[0] for r in rows]); f = rows[0][1]
    P = np.array([np.interp(f, r[1], r[2]) for r in rows]).T
    fig, ax = plt.subplots(figsize=(6.5, 4))
    im = ax.pcolormesh(fin, f * 1000, np.log10(P + 1e-12), shading="nearest", cmap="magma",
                       vmin=np.log10(P.max()) - 6, vmax=np.log10(P.max()))
    ax.set_ylim(0, 20); ax.set_xlabel("input rate $f_{in}$ (Hz)"); ax.set_ylabel("frequency (mHz)")
    ax.set_title("Power of the mean excitatory gain (log$_{10}$)", loc="left")
    fig.colorbar(im, ax=ax, pad=0.02)
    fig.savefig(d / "input_sweep.png", dpi=170, bbox_inches="tight")
    print("wrote", d / "input_sweep.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "results" / "input_sweep")
