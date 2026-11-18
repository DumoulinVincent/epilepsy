"""Figure of the reduced event model (event_model.py).

    python event_model_figures.py     # -> results/event_model/event_model.{png,pdf}

(a-c) k_E after deafferentation (seizures ended after 10 s) for three settings of the homeostatic time constants
      (insertion = k_E rising, internalization = k_E falling) and of the rate sensor.
(d)   variability of the gaps between seizures (SD/mean: 1 = random timing) against the sensor time constant.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import event_model as em  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "event_model"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE = "#e34948", "#2a78d6"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.5, 9.2))
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.4)
    cases = [(5000, 5000, 1000, "(a) sensor 1000 s: one seizure per cycle"),
             (5000, 5000, 2500, "(b) sensor 2500 s: clusters ended by a close pair"),
             (10000, 5000, 2000, "(c) slower insertion (10000 s), sensor 2000 s: longer clusters")]
    for i, (tu, td, ts, title) in enumerate(cases):
        k, on, _ = em.run(1_000_000, tu, td, ts, 10, seed=1)
        t = np.arange(len(k)) * 10
        a = fig.add_subplot(gs[i, :3])
        a.plot(t / 3600, k, color=BLUE, lw=1)
        a.plot(on / 3600, np.interp(on, t, k), "v", color=RED, ms=5, mec="k", mew=0.3)
        a.axhspan(0.945, 0.99, color=RED, alpha=0.06, lw=0)
        a.set_ylim(0.72, 0.98); a.set_xlim(0, 200); a.set_ylabel("$k_E$")
        a.set_title(f"{title}   (insertion {tu} s, internalization {td} s)", loc="left", fontsize=9.5)
    a.set_xlabel("time after deafferentation (h, model)")
    d = fig.add_subplot(gs[:2, 3])
    sens = np.array([100, 250, 500, 1000, 1500, 2000, 2500, 3000, 5000, 10000, 20000])
    for tu, td, col in ((5000, 5000, BLUE), (10000, 5000, "#7b4fbf")):
        cv = []
        for ts in sens:
            v = []
            for seed in range(5):
                on = em.run(1_000_000, tu, td, ts, 10, seed=seed)[1]
                isi = np.diff(on[on > 20000]); v.append(isi.std() / isi.mean())
            cv.append(np.mean(v))
        d.plot(sens, cv, "o-", color=col, ms=4, label=f"insertion {tu} s, internalization {td} s")
    d.axhline(1, color="k", ls="--", lw=0.8); d.text(110, 1.04, "random timing", fontsize=7.5)
    d.set_xscale("log"); d.set_ylim(0, 1.5)
    d.set_xlabel("rate-sensor time constant (s)"); d.set_ylabel("gap variability (SD / mean)")
    d.set_title("(d) Regular vs clustered", loc="left")
    d.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.55))
    fig.savefig(OUT / "event_model.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "event_model.pdf", bbox_inches="tight")
    print("wrote", OUT / "event_model.png")


if __name__ == "__main__":
    main()
