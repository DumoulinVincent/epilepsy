"""Validation checks and figures for the LFP proxy (lfp.py). Writes results/lfp/.

    python lfp_checks.py                    # checks + full runs + figures
    python lfp_checks.py --plots-only       # figures from the saved .npz only

Checks (checks_report.txt, run at the first --freqs entry):
  0. the probe equals the mean of the per-neuron currents recorded directly
  1. monitor array shape and MB for the full run
  2. spike trains identical with vs without the probe
  4. wall-time overhead of the summed variables
  aliasing: 1 ms (= integration step) vs 2 ms recording, PSD below 50 Hz
  (check 3, total vs recurrent not being the same array, is moot now that
  the proxy is recurrent-only -- there is a single proxy array)

Figures (psd_report.txt):
  timeseries_seed{S}_{F}Hz.png -- LFP = |I_exc|+|I_inh| (pA) and its two
      components, E rate, I rate, E/I rate ratio, k_AMPA, k_GABA, with the
      PSD windows shaded
  psd_windows_seed{S}.png -- PSD + aperiodic fit/residual for each window:
      interictal and peri-onset (bursting run), steady (non-bursting run)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from brian2 import StateMonitor, ms, second, seed as b2seed  # noqa: E402

import lfp  # noqa: E402
from progress import brian_progress_reporter  # noqa: E402

RNG_SEED = 12345
SHORT_S = 20.0
BURST_HZ = 8.0                # 4 x target rate, same threshold as HSP_Phase
POST_BURST_MARGIN_S = 600.0   # post-ictal rate depression lasts ~500 s at 36 Hz
WINDOW_S = 60.0
BIN_S = 1.0
N_E, N_I = 100, 25
PEAK_BAND = (12.0, 40.0)
FS = 1.0 / float(lfp.REC_DT / second)


# ---------------------------------------------------------------- checks
def short_run(seed, baseline, freq, with_probe, extra_fine=False):
    """20 s from the baseline after a compile warm-up; returns
    (wall_s, spike_i, spike_t, m_lfp, m_fine)."""
    net, m_lfp, _ = lfp.build_restored(seed, baseline, freq, with_probe=with_probe)
    m_fine = None
    if extra_fine:
        m_fine = StateMonitor(net["LFP"], ["I_exc", "I_inh"], record=0,
                              dt=lfp.DT, name="M_LFP_fine")
        net.add(m_fine)
    ns = lfp.model_namespace()
    net.store("pre")
    net.run(10 * ms, namespace=ns)          # compile everything once
    best = np.inf
    for _ in range(1 if extra_fine else 3):
        net.restore("pre")
        b2seed(RNG_SEED)
        t0 = float(net.t / second)
        tic = time.perf_counter()
        net.run(SHORT_S * second, namespace=ns)
        best = min(best, time.perf_counter() - tic)
    sp = net["sp_exc"]
    st = np.asarray(sp.t / second)
    keep = st >= t0
    return best, np.asarray(sp.i)[keep], st[keep], m_lfp, m_fine


def probe_vs_per_neuron(seed, baseline, freq, dur_s=2.0):
    """Record the probe AND the two recurrent currents of every E cell, both
    every dt, for dur_s; compare the probe to the per-neuron mean. Returns
    {probe var: (max rel err at lag 0, at lag 1 step)}."""
    net, _, _ = lfp.build_restored(seed, baseline, freq, with_probe=False)
    m_lfp, _ = lfp.add_lfp_probe(net, rec_dt=lfp.DT)
    m_pn = StateMonitor(net["P_E"], ["I_AMPA_rec", "I_GABA_rec"],
                        record=True, dt=lfp.DT, name="M_per_neuron")
    net.add(m_pn)
    b2seed(RNG_SEED)
    net.run(dur_s * second, namespace=lfp.model_namespace())
    out = {}
    for probe, per_neuron in (("I_exc", "I_AMPA_rec"), ("I_inh", "I_GABA_rec")):
        a = np.asarray(getattr(m_lfp, probe)[0])
        b = np.asarray(getattr(m_pn, per_neuron)).mean(axis=0)
        scale = np.abs(b).max()
        out[probe] = (np.abs(a - b).max() / scale, np.abs(a[1:] - b[:-1]).max() / scale)
    return out


def run_checks(seed, baseline, freq, say):
    say(f"seed {seed}, {freq:g} Hz input, untreated; dt={lfp.DT}, REC_DT={lfp.REC_DT}")

    say("\n[0] probe vs mean of the per-neuron currents (2 s, both every dt), "
        "max error relative to the signal's peak:")
    for name, (e0, e1) in probe_vs_per_neuron(seed, baseline, freq).items():
        say(f"    {name}: lag 0 {e0:.1e}, lag 1 step {e1:.1e}")
    say("    (the smaller one is the match; a 1-step lag means the summed "
        "variable is updated after the monitor samples)")

    wall_off, i_off, t_off, _, _ = short_run(seed, baseline, freq, False)
    wall_on, i_on, t_on, _, _ = short_run(seed, baseline, freq, True)
    same = (len(i_off) == len(i_on) and np.array_equal(i_off, i_on)
            and np.array_equal(t_off, t_on))
    say("\n[2] spike trains with vs without probe (20 s, same RNG seed): "
        f"{'IDENTICAL' if same else 'DIFFERENT'} ({len(i_off)} vs {len(i_on)} spikes)")
    say(f"\n[4] wall time for {SHORT_S:g} s, best of 3: without {wall_off:.2f} s, "
        f"with {wall_on:.2f} s -> overhead {100 * (wall_on / wall_off - 1):+.1f}%")

    _, _, _, m_c, m_f = short_run(seed, baseline, freq, True, extra_fine=True)
    _, *coarse = lfp.lfp_from_monitor(m_c)
    _, *fine = lfp.lfp_from_monitor(m_f)
    say("\n[alias] 1 ms (every integration step) vs 2 ms, relative PSD diff 0-50 Hz:")
    worst = 0.0
    for name, xc, xf in zip(["lfp", "lfp_exc", "lfp_inh"], coarse, fine):
        fc, Pc = lfp.psd_of(xc, FS)
        ff, Pf = lfp.psd_of(xf, 1.0 / float(lfp.DT / second))
        m = (fc > 0) & (fc < 50.0)
        rel = np.abs(Pc[m] - np.interp(fc[m], ff, Pf)) / np.interp(fc[m], ff, Pf)
        worst = max(worst, rel.max())
        say(f"    {name:14s} max {100 * rel.max():6.2f}% at {fc[m][np.argmax(rel)]:5.2f} Hz, "
            f"median {100 * np.median(rel):5.2f}%")
    say("    -> " + ("exceeds a few percent: aliasing is NOT negligible at 2 ms"
                    if worst > 0.05 else "within a few percent: 2 ms is safe below 50 Hz"))


# ------------------------------------------------------------ full runs
def npz_path(out, seed, freq):
    return out / f"seed{seed}_untreated_{freq:g}Hz.npz"


def full_run(seed, baseline, freq, duration, out, say):
    say(f"\nfull run {freq:g} Hz, {duration:g} s ...")
    tic = time.perf_counter()
    (t, _, exc, inh), (tw, kA, kG), (_, tE), (_, tI) = lfp.run_lfp(
        seed, baseline, freq, duration_s=duration, rng_seed=RNG_SEED,
        report=brian_progress_reporter(f"lfp seed{seed} {freq:g}Hz"))
    say(f"  ({time.perf_counter() - tic:.0f} s wall)")
    t_start, t_end = np.floor(t[0]), t[-1]
    rate_t, rate_E = lfp.binned_rate(tE, N_E, t_start, t_end, BIN_S)
    _, rate_I = lfp.binned_rate(tI, N_I, t_start, t_end, BIN_S)
    nbytes = sum(a.nbytes for a in (exc, inh))
    say(f"[1] M_LFP: 2 traces, each shape (1, {len(t)}) -> {nbytes / 1e6:.1f} MB; "
        f"weight monitor 2 x 100 x {len(tw)} = {2 * 100 * len(tw) * 8 / 1e6:.1f} MB "
        f"(per-neuron alternative: {100 * 2 * len(t) * 8 / 1e9:.2f} GB)")
    # lfp is the exact sum of the two components; not stored twice
    np.savez_compressed(npz_path(out, seed, freq), t=t, lfp_exc=exc, lfp_inh=inh,
                        t_w=tw, k_ampa=kA, k_gaba=kG,
                        rate_t=rate_t, rate_E=rate_E, rate_I=rate_I)


def load_run(path):
    d = dict(np.load(path))
    d["lfp"] = d["lfp_exc"] + d["lfp_inh"]
    with np.errstate(divide="ignore", invalid="ignore"):
        d["ei_ratio"] = np.where(d["rate_I"] > 0, d["rate_E"] / d["rate_I"], np.nan)
    return d


# -------------------------------------------------------------- windows
def burst_onsets(run):
    b = (run["rate_E"] > BURST_HZ).astype(int)
    return run["rate_t"][np.flatnonzero(np.diff(b) == 1) + 1] - BIN_S / 2


def quiet_window(run):
    """Start of a WINDOW_S window in the middle of the longest stretch that
    is burst-free and >= POST_BURST_MARGIN_S after any burst and after the
    run start (the input switch from the 40 Hz baseline is a transient).
    Interictal in a bursting run, steady state in a non-bursting one."""
    rate = run["rate_E"]
    margin = int(round(POST_BURST_MARGIN_S / BIN_S))
    ok = np.ones(len(rate), bool)
    ok[:margin] = False
    for k in np.flatnonzero(rate > BURST_HZ):
        ok[k: k + margin + 1] = False
    best_len, best_start, run_len = 0, None, 0
    for k, good in enumerate(ok):
        run_len = run_len + 1 if good else 0
        if run_len > best_len:
            best_len, best_start = run_len, k - run_len + 1
    if best_len * BIN_S < WINDOW_S:
        return None
    start = run["rate_t"][best_start] - BIN_S / 2
    return start + best_len * BIN_S / 2 - WINDOW_S / 2


def onset_window(run):
    """WINDOW_S centered on the first burst onset that fits inside the run."""
    t0, t1 = run["t"][0], run["t"][-1]
    for onset in burst_onsets(run):
        if onset - WINDOW_S / 2 >= t0 and onset + WINDOW_S / 2 <= t1:
            return onset - WINDOW_S / 2
    return None


def window_psd(run, w0, key="lfp"):
    t = run["t"]
    m = (t >= w0) & (t < w0 + WINDOW_S)
    f, P = lfp.psd_of(run[key][m], FS)
    resid, slope, icpt = lfp.aperiodic_fit(f, P)
    pk_f, pk_h = lfp.band_peak(f, resid, PEAK_BAND)
    in_fit = np.zeros(f.shape, bool)
    for lo, hi in lfp.FIT_BANDS:
        in_fit |= (f >= lo) & (f <= hi)
    band = (f >= PEAK_BAND[0]) & (f < PEAK_BAND[1])
    rm = (run["rate_t"] >= w0) & (run["rate_t"] < w0 + WINDOW_S)
    rE, rI = run["rate_E"][rm].mean(), run["rate_I"][rm].mean()
    return dict(f=f, P=P, resid=resid, slope=slope, icpt=icpt, pk_f=pk_f, pk_h=pk_h,
                band_mean=np.nanmean(resid[band]), noise_sd=np.nanstd(resid[in_fit]),
                # ratio of the window-mean rates: a mean of per-bin ratios is
                # dominated by the quiet bins and misreports a burst window
                ei=rE / rI if rI > 0 else np.nan, rE=rE, rI=rI,
                exc_share=run["lfp_exc"][m].mean() / run["lfp"][m].mean())


# ---------------------------------------------------------------- plots
def bin_mean(t, x, t0, t1):
    edges = np.arange(t0, t1 + 1e-9, BIN_S)
    n, _ = np.histogram(t, bins=edges)
    s, _ = np.histogram(t, bins=edges, weights=x)
    with np.errstate(invalid="ignore"):
        return edges[:-1] + BIN_S / 2, s / n


def plot_timeseries(run, seed, freq, windows, path):
    t0, t1 = np.floor(run["t"][0]), run["t"][-1]
    pA = 1e12
    tb, tot = bin_mean(run["t"], run["lfp"], t0, t1)
    comps = [(name, bin_mean(run["t"], run[f"lfp_{name}"], t0, t1)[1], c)
             for name, c in (("exc", "tab:red"), ("inh", "tab:blue"))]

    fig, ax = plt.subplots(7, 1, figsize=(11, 17), sharex=True)
    ax[0].plot(tb, tot * pA, color="k", lw=0.8)
    ax[0].set_ylabel("LFP proxy\n|I_exc|+|I_inh| (pA)")
    for name, y, c in comps:
        ax[1].plot(tb, y * pA, color=c, lw=0.8, label=f"|I_{name}|")
    ax[1].set_yscale("log")
    ax[1].set_ylim(bottom=1e-3)   # |I_inh| dips toward 0 while I cells are silent
    ax[1].set_ylabel("components (pA)")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[2].plot(run["rate_t"], run["rate_E"], color="tab:red", lw=0.8)
    ax[2].set_ylabel("E rate (Hz)")
    ax[3].plot(run["rate_t"], run["rate_I"], color="tab:blue", lw=0.8)
    ax[3].set_ylabel("I rate (Hz)")
    for a in (ax[2], ax[3]):
        a.set_yscale("symlog", linthresh=1)
        a.set_ylim(bottom=0)
    ax[4].plot(run["rate_t"], run["ei_ratio"], color="tab:purple", lw=0.8)
    ax[4].axhline(1, color="0.5", ls=":", lw=1)
    ax[4].set_yscale("log")
    ax[4].set_ylabel("E rate / I rate\n(gaps: I silent)")
    ax[5].plot(run["t_w"], run["k_ampa"], color="tab:orange", lw=1)
    ax[5].set_ylabel("mean k_AMPA")
    ax[6].plot(run["t_w"], run["k_gaba"], color="tab:green", lw=1)
    ax[6].set_ylabel("mean k_GABA")
    ax[6].set_xlabel("time (s)")
    for label, w0, color in windows:
        for a in ax:
            a.axvspan(w0, w0 + WINDOW_S, color=color, alpha=0.25,
                      label=label if a is ax[0] else None)
    if windows:
        ax[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"LFP proxy and E/I balance -- seed {seed}, {freq:g} Hz, untreated "
                 f"(LFP and rates in 1 s bins)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_psds(entries, seed, path, key="lfp"):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for label, color, p in entries:
        sel = p["f"] > 0
        f = p["f"][sel]
        a1.loglog(f, p["P"][sel], color=color, lw=1, label=f"{label} (slope {p['slope']:.2f})")
        a1.loglog(f, 10 ** (p["icpt"] + p["slope"] * np.log10(f)), color=color, lw=2.5,
                  alpha=0.35)
        a2.semilogx(f, p["resid"][sel], color=color, lw=1, label=label)
    for a in (a1, a2):
        for lo, hi in lfp.FIT_BANDS:
            a.axvspan(lo, hi, color="tab:blue", alpha=0.08)
        a.axvspan(*PEAK_BAND, color="tab:orange", alpha=0.10)
        a.set_xlim(0.5, FS / 2)
    a2.axhline(0, color="0.4", ls=":", lw=1)
    a1.set_ylabel("PSD of LFP proxy |I_exc|+|I_inh| (A$^2$/Hz)")
    a1.set_title(f"60 s windows, seed {seed} -- thick faint lines: aperiodic fits\n"
                 f"(blue shade = fit bands, orange = {PEAK_BAND[0]:g}-{PEAK_BAND[1]:g} Hz; "
                 f"peri-onset above ~100 Hz is aliased at 2 ms sampling)")
    a1.legend(fontsize=8)
    a2.set_ylabel("residual above aperiodic fit (log10)")
    a2.set_xlabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--freqs", default="36,40",
                    help="input rates; the first one is also used for the checks")
    ap.add_argument("--duration", type=float, default=3000.0)
    ap.add_argument("--plots-only", action="store_true",
                    help="skip simulation; build figures from the saved .npz files")
    args = ap.parse_args()
    freqs = [float(x) for x in args.freqs.split(",")]

    baseline = THIS_DIR / "results" / "baselines" / f"seed{args.seed}.bri"
    out = THIS_DIR / "results" / "lfp"
    out.mkdir(parents=True, exist_ok=True)

    if not args.plots_only:
        lines = []

        def say_check(s=""):
            print(s, flush=True)
            lines.append(s)

        run_checks(args.seed, baseline, freqs[0], say_check)
        for freq in freqs:
            full_run(args.seed, baseline, freq, args.duration, out, say_check)
        (out / "checks_report.txt").write_text("\n".join(lines) + "\n")

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    colors = {"interictal": "tab:green", "peri-onset": "tab:red", "steady": "tab:blue"}
    entries = []
    for freq in freqs:
        run = load_run(npz_path(out, args.seed, freq))
        bursting = len(burst_onsets(run)) > 0
        windows = []
        w = quiet_window(run)
        if w is not None:
            windows.append(("interictal" if bursting else "steady", w))
        w = onset_window(run) if bursting else None
        if w is not None:
            windows.append(("peri-onset", w))
        say(f"{freq:g} Hz: {len(burst_onsets(run))} burst onset(s)")
        shaded = []
        for kind, w0 in windows:
            label = f"{kind} {freq:g} Hz"
            shaded.append((label, w0, colors[kind]))
            p = window_psd(run, w0)
            entries.append((label, colors[kind], p))
            say(f"  {label} {w0:.0f}-{w0 + WINDOW_S:.0f} s: E {p['rE']:.2f} Hz, "
                f"I {p['rI']:.2f} Hz, E/I {p['ei']:.2f}; |I_exc| is "
                f"{100 * p['exc_share']:.0f}% of the LFP")
            say(f"    slope {p['slope']:+.2f}, {PEAK_BAND[0]:g}-{PEAK_BAND[1]:g} Hz "
                f"residual mean {p['band_mean']:+.3f} (max {p['pk_h']:+.3f} at "
                f"{p['pk_f']:.1f} Hz, fit-band SD {p['noise_sd']:.3f})")
        plot_timeseries(run, args.seed, freq, shaded,
                        out / f"timeseries_seed{args.seed}_{freq:g}Hz.png")
    plot_psds(entries, args.seed, out / f"psd_windows_seed{args.seed}.png")
    say("(peri-onset spans the transition, so it is non-stationary by construction; "
        "its PSD is dominated by the burst's onset transient)")
    (out / "psd_report.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
