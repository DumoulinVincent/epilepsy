"""Validation of the image-recovery workflow before any long run (brief §15).

    python test_image_task.py            # analysis-only tests (no network)
    python test_image_task.py --brian    # + the stimulus lookup in a tiny Brian2
                                         #   model (no network, a few seconds)

Every check prints PASS / FAIL; the script exits non-zero if any fails.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import worker  # noqa: E402,F401
import conditions as C  # noqa: E402
import image_stimulus as ist  # noqa: E402
import image_analysis as ia  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILED.append(name)


def test_stimulus():
    g = dict(C.IMAGE_GRID)
    masks, _ = ist.make_library(g)
    check("random library: every mask has 10 white pixels", bool(np.all(masks.sum(1) == 10)))
    check("random library: every cell white in exactly n_library_blocks masks",
          bool(np.all(masks.sum(0) == g["n_library_blocks"])))
    lines = ist.line_masks()
    check("line library: 20 masks, 10 white each, every cell white twice",
          lines.shape == (20, 100) and bool(np.all(lines.sum(1) == 10)) and bool(np.all(lines.sum(0) == 2)))
    from seeded_network import e_cell_xy
    x, y = ist.cell_xy()
    xs, ys = e_cell_xy()
    check("grid indexing i = 10 y + x matches seeded_network", bool(np.all(x == xs) and np.all(y == ys)
                                                                    and np.all(10 * y + x == np.arange(100))))
    check("row 0 of the line library is the top row y = 0", bool(np.all(np.flatnonzero(lines[0]) == np.arange(10))))
    for f in (36.0, 40.0):
        tab = ist.rate_table(masks, f, g["contrast_hz"], 10)
        check(f"rates at f = {f:g}: non-negative, summed rate 100 f in every row, blank row = f",
              bool(np.all(tab >= 0) and np.allclose(tab.sum(1), 100 * f) and np.allclose(tab[0], f)))
        check(f"white / black rates at f = {f:g}: f + 0.9 a, f - 0.1 a",
              bool(np.isclose(tab[1][masks[0] == 1][0], f + 0.9 * 4) and np.isclose(tab[1][masks[0] == 0][0], f - 0.1 * 4)))
    try:
        ist.rate_table(masks, 0.3, 4.0, 10)
        check("negative rates are refused (not clipped)", False)
    except ValueError:
        check("negative rates are refused (not clipped)", True)
    sch = ist.make_schedule(len(masks), 3000.0, g)
    sb = g["sched_ms"]
    dur = sch["offset"] - sch["onset"]
    check("every image lasts image_ms", bool(np.all(dur == g["image_ms"] // sb)))
    check("blanks within [200, 600] ms", bool(np.all((sch["blank"] >= 20) & (sch["blank"] <= 60))))
    check("first onset is after a blank, trials do not overlap",
          bool(sch["onset"][0] == sch["blank"][0] and np.all(sch["onset"][1:] == sch["offset"][:-1] + sch["blank"][1:])))
    tab = ist.rate_table(masks, 36.0, 4.0, 10)
    per_bin = tab[sch["img_series"]].sum(1)
    check("expected summed drive = 100 f in every schedule bin (images and blanks)",
          bool(np.allclose(per_bin, 3600.0)))
    ok = all(np.all(sch["img_series"][a:b] == m + 1) for a, b, m in zip(sch["onset"], sch["offset"], sch["mask_id"]))
    check("img_series shows mask m + 1 exactly during [onset, offset), 0 elsewhere",
          ok and int((sch["img_series"] > 0).sum()) == int(dur.sum()))
    counts = np.bincount(sch["mask_id"], minlength=len(masks))
    check("presentation order: every mask shown, counts differ by at most 1",
          bool(counts.min() >= 1 and counts.max() - counts.min() <= 1))


def test_binning():
    t0, dt = 15000.0, 0.001
    m = np.arange(0, 20000)
    t = np.round(t0 + m * dt, 6)
    k = ist.tick_bins(t, t0, dt, 10)
    check("tick binning: every tick in its half-open 10 ms bin, boundaries included",
          bool(np.all(k == m // 10)))
    naive = np.floor((t - t0) / 0.01).astype(int)
    check("(the floating-point floor it replaces does misbin boundary ticks)", bool(np.any(naive != m // 10)))
    check("pre-restore spike (t < t0) gets a negative bin and is excluded",
          bool(ist.tick_bins([t0 - 0.001], t0, dt, 10)[0] < 0))


def test_scores():
    rng = np.random.default_rng(0)
    masks = ist.random_masks(1, 10, 3)
    Y = masks[rng.integers(0, 10, 200)].astype(float)
    q = 0.1
    s = ia.trial_scores(Y.copy(), Y, 0.5, q)
    check("perfect reconstruction: BA = 1, AUROC = 1, S_B = 1",
          bool(np.allclose(s["BA"], 1) and np.allclose(s["AUROC"], 1) and np.allclose(s["SB"], 1)))
    s = ia.trial_scores(np.zeros_like(Y), Y, 0.5, q)
    check("all-black prediction: BA = 0.5", bool(np.allclose(s["BA"], 0.5)))
    s = ia.trial_scores(np.ones_like(Y), Y, 0.5, q)
    check("all-white prediction: BA = 0.5", bool(np.allclose(s["BA"], 0.5)))
    s = ia.trial_scores(np.full_like(Y, 0.37), Y, 0.5, q)
    check("tied probabilities: AUROC = 0.5", bool(np.allclose(s["AUROC"], 0.5)))
    s = ia.trial_scores(np.full_like(Y, q), Y, 0.5, q)
    check("prior-only probabilities: S_B = 0", bool(np.allclose(s["SB"], 0)))


def _synthetic_run(path, cond, seed, gain, g, silent=False, run_s=1500.0):
    """A fake image_cycle output: an independent-Poisson relay, base 1 Hz,
    white cells + gain Hz during images; seizure-free."""
    masks, names = ist.make_library(g)
    sch = ist.make_schedule(len(masks), run_s, g)
    rng = np.random.default_rng(seed * 100 + int(gain * 10))
    lam = np.full((sch["n_bins"], 100), 1.0)
    for a, b, m in zip(sch["onset"], sch["offset"], sch["mask_id"]):
        lam[a:b] += gain * masks[m]
    E = np.zeros_like(lam, dtype=np.uint8) if silent else rng.poisson(lam * g["sched_ms"] / 1000).astype(np.uint8)
    meta = dict(seed=seed, condition=cond, freq_hz=36.0, contrast_hz=g["contrast_hz"], n_E=100,
                grid={k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()}, completed=True)
    np.savez_compressed(path, E_counts=E, I_counts=np.zeros(sch["n_bins"], np.uint16), masks=masks,
                        mask_names=np.array(names), onset=sch["onset"], offset=sch["offset"],
                        blank=sch["blank"], mask_id=sch["mask_id"], t_k=np.arange(0, run_s, 0.2),
                        k_AMPA=np.zeros(int(run_s / 0.2)), k_GABA=np.zeros(int(run_s / 0.2)),
                        seizures=np.zeros((0, 2)), meta=json.dumps(meta))


def test_decoding():
    g = dict(C.IMAGE_GRID, settle_s=100.0, run_s=1500.0, C_grid=(0.01, 1.0, 100.0))
    q = 0.1
    with tempfile.TemporaryDirectory() as d:
        raw = Path(d) / "raw"
        raw.mkdir()
        _synthetic_run(raw / "seed1_reference_40Hz.npz", "reference 40Hz", 1, 20.0, g)
        _synthetic_run(raw / "seed1_untreated.npz", "untreated", 1, 8.0, g)
        _synthetic_run(raw / "seed1_DBS_20pA.npz", "DBS 20pA", 1, 0.0, g, silent=True)
        run = ia.load_run(raw / "seed1_reference_40Hz.npz")
        tt = ia.trial_table(run, g)
        X = tt["feats"][250]
        S, thr, fold, st = ia.run_adapted(X, tt["Y"], tt, "local", g, q)
        sc = ia.trial_scores(S[tt["post"]], tt["Y"][tt["post"]], thr[tt["post"]][:, None], q)
        check("synthetic relay: local adapted decoder well above chance",
              sc["BA"].mean() > 0.7, f"BA {sc['BA'].mean():.3f}")
        Yp = ia.permuted_labels(tt, run, np.random.default_rng(1))
        S2, thr2, _, _ = ia.run_adapted(X, Yp, tt, "local", g, q)
        sc2 = ia.trial_scores(S2[tt["post"]], tt["Y"][tt["post"]], thr2[tt["post"]][:, None], q)
        check("permuted labels remove the advantage (BA near 0.5)",
              abs(sc2["BA"].mean() - 0.5) < 0.03, f"BA {sc2['BA'].mean():.3f}")
        S3, thr3, _, _ = ia.run_adapted(X, tt["Y"], tt, "population", g, q)
        sc3 = ia.trial_scores(S3[tt["post"]], tt["Y"][tt["post"]], thr3[tt["post"]][:, None], q)
        check("synthetic relay: population decoder above chance", sc3["BA"].mean() > 0.65,
              f"BA {sc3['BA'].mean():.3f}")
        # outer test trials cannot change training choices
        Xp = X.copy()
        post = np.flatnonzero(tt["post"])
        first_block = post[:len(post) // g["outer_folds"]]
        Xp[first_block] = np.random.default_rng(5).poisson(9.0, Xp[first_block].shape)
        _, _, _, st_a = ia.run_adapted(X, tt["Y"], tt, "local", g, q)
        _, _, _, st_b = ia.run_adapted(Xp, tt["Y"], tt, "local", g, q)
        check("fold 0: scaling/C/threshold unaffected by its own test trials",
              st_a[0] == st_b[0], f"{st_a[0]} vs {st_b[0]}")
        # silent response
        run_s = ia.load_run(raw / "seed1_DBS_20pA.npz")
        tts = ia.trial_table(run_s, g)
        Ss, thrs, _, sts = ia.run_adapted(tts["feats"][250], tts["Y"], tts, "local", g, q)
        scs = ia.trial_scores(Ss[tts["post"]], tts["Y"][tts["post"]], thrs[tts["post"]][:, None], q)
        # BA is exactly 0.5 (one probability per pixel -> every pixel the same
        # call). AUROC is only ~0.5: with no spikes each pixel's intercept is
        # its training white-frequency, and in a balanced library pixels
        # over-represented in training are under-represented in the held-out
        # block, so the ranking deviates slightly (the label-permutation null
        # has the same baseline).
        check("silent response: no information (BA = 0.5, AUROC within 0.03 of 0.5)",
              np.allclose(scs["BA"], 0.5) and abs(scs["AUROC"].mean() - 0.5) < 0.03,
              f"BA {scs['BA'].mean():.3f}, AUROC {scs['AUROC'].mean():.3f}")
        # constant labels: a pixel never white in training
        Yc = tt["Y"][post].copy()
        Yc[:, 7] = 0
        fit = ia.fit_logit(X[post], Yc, "population", 1.0)
        p7 = ia.predict_proba(fit, X[post])[:, 7]
        check("constant-label pixel: flagged prior-only, no crash, never predicted white",
              bool(fit["prior_only"][7]) and bool(np.all(p7 < 0.01)))
        # frozen decoder: trained on the reference, fixed, applied to others
        by_cond = {c: raw / f"seed1_{s}.npz" for c, s in      # analyse_frozen takes file paths
                   (("reference 40Hz", "reference_40Hz"), ("untreated", "untreated"), ("DBS 20pA", "DBS_20pA"))}
        frz = ia.analyse_frozen(by_cond, g, models=("local",))
        ba = {c: np.nanmean(frz[(c, 250, "local")]["BA"]) for c in by_cond}
        check("frozen decoder: reference > weaker relay > silent", ba["reference 40Hz"] > ba["untreated"] > 0.49
              and abs(ba["DBS 20pA"] - 0.5) < 0.02, ", ".join(f"{c} {v:.3f}" for c, v in ba.items()))
        frz2 = ia.analyse_frozen(by_cond, g, models=("local",))
        check("frozen decoder: identical results when re-evaluated",
              all(np.allclose(frz[k]["BA"], frz2[k]["BA"], equal_nan=True) for k in frz))
        # whole pipeline on the synthetic runs (incl. figures)
        ia.main(d, n_jobs=1, fast=True)
        ok = (Path(d) / "analysis" / "summary.csv").exists() and (Path(d) / "analysis" / "plots" / "reconstructions.png").exists()
        check("full analysis pipeline runs on synthetic runs (summary + figures)", ok)


def test_brian_lookup():
    """The image rates seen by the network at every 1 ms tick, after jumping
    the clock to t = 15000 s as a baseline restore does."""
    from brian2 import NeuronGroup, StateMonitor, Network, TimedArray, ms, second, Hz, defaultclock, prefs
    prefs.codegen.target = "cython"
    defaultclock.dt = 1 * ms
    g = dict(C.IMAGE_GRID)
    masks = ist.random_masks(1, 10, 9)
    tab = ist.rate_table(masks, 36.0, 4.0, 10)
    series = np.array([0] * 23 + [1] * 60 + [0] * 27 + [5] * 60 + [0] * 30, np.int32)
    img_ta = TimedArray(series.astype(float), dt=10 * ms)
    rate_ta = TimedArray(tab * Hz, dt=1 * ms)
    t_start = 15000 * second
    G = NeuronGroup(100, "r = rate_ta((img_ta(t - t_img0) + 0.5) * ms, i) : Hz",
                    namespace=dict(img_ta=img_ta, rate_ta=rate_ta, t_img0=t_start))
    mon = StateMonitor(G, "r", record=True)
    net = Network(G, mon)
    net.t_ = float(t_start / second)        # as after restoring a baseline stored at 15000 s
    net.run(len(series) * 10 * ms)
    r = np.asarray(mon.r / Hz).T                       # (ticks, 100)
    tick = np.round((np.asarray(mon.t / second) - 15000) * 1000).astype(int)
    want = tab[series[tick // 10]]
    check("Brian2 lookup: rate at every tick equals the scheduled row (incl. boundaries, t0 = 15000 s)",
          bool(np.allclose(r, want)), f"{int((~np.isclose(r, want)).any(1).sum())} mismatching ticks of {len(tick)}")


if __name__ == "__main__":
    test_stimulus()
    test_binning()
    test_scores()
    test_decoding()
    if "--brian" in sys.argv:
        test_brian_lookup()
    print(f"\n{len(FAILED)} failure(s)" + (": " + "; ".join(FAILED) if FAILED else ""))
    sys.exit(1 if FAILED else 0)
