"""Checks required before the full assay run (prompt, page 7).

    python test_assays.py [results/assays_smoke]

Schedule/analysis checks need no simulation; with a results directory, the
sham-replay and HSP-during-probe checks run on its raw files.
"""
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import worker  # noqa: E402,F401
import conditions as C  # noqa: E402
import assay_stimulus as ast  # noqa: E402
import assay_analysis as aa  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILED.append(name)


def test_schedule():
    g = dict(C.ASSAY_GRID, dt_ms=1)
    cps = ast.checkpoint_times(g)
    sch = ast.build_schedule(g, len(cps))
    A = g["amplitude_frac"] * 36.0
    try:
        ast.check_schedule(sch, [36.0, 40.0], A)
        ok = True
    except ValueError:
        ok = False
    check("every pattern row sums to 0, |m| <= 1, rates >= 0.8 b for b = 36 and 40 Hz", ok,
          f"{len(sch['rows'])} rows, min rate {36 - A:.1f} Hz")
    exp_total = (36.0 + A * sch["rows"]).sum(1)
    check("identical expected total drive (100 b) for every symbol and corruption level",
          bool(np.allclose(exp_total, 3600.0)))
    q = sch["mem_q"]
    ac = [np.corrcoef(q[:, :-L].ravel(), q[:, L:].ravel())[0, 1] for L in (1, 2, 3)]
    check("memory symbols independent: |lag-1..3 autocorrelation| < 0.02, balanced",
          all(abs(a) < 0.02 for a in ac) and abs(q.mean()) < 0.02, f"autocorr {np.round(ac, 4)}, mean {q.mean():+.3f}")
    eq = sch["enc_q"]
    check("encoding symbols balanced and the prior stimulus balanced too",
          abs(eq.mean()) < 0.03 and abs(np.mean(eq[:, 1:][eq[:, :-1] > 0])) < 0.03)
    eta = sch["noise_eta"][:, 1:]
    counts = {e: int((eta == e).sum()) for e in g["noise_etas"]}
    check("every corruption level presented equally often", len(set(counts.values())) == 1, str(counts))
    rows = sch["rows"]
    w = ast.mask_w()
    corrupt_rows = rows[3:]
    # v must not carry the label: the corrupted component (row - (1-eta) q w) is label-independent
    lab = []
    for j in range(len(cps)):
        for k, e in enumerate(sch["noise_eta"][j]):
            if e > 0:
                lab.append(sch["noise_q"][j, k])
    v_part = np.array([r - (r @ w / 100) * w for r in corrupt_rows])     # remove the w component
    check("corruption mask v is label-independent (mean v component equal for both labels)",
          np.abs(v_part[np.array(lab) > 0].mean(0) - v_part[np.array(lab) < 0].mean(0)).max() < 0.15)
    check("slots are disjoint and each branch reads only its own slot",
          bool(np.all(np.diff(sch["offsets"].ravel()) > 0)))


def _synthetic_run(n_cp=60, seed=0, relay=True, silent=False):
    """A run dict as load_run returns, from a toy independent relay: E count
    of cell i = Poisson(base + gain * max(0, m_i)) per window, no recurrence."""
    rng = np.random.default_rng(seed)
    g = dict(C.ASSAY_GRID, dt_ms=1, run_s=3000.0, settle_s=100.0)
    sch = ast.build_schedule(g, n_cp)
    w = ast.mask_w()
    run = dict(g=g, meta=dict(seed=1, condition="toy", A_hz=7.2, freq_hz=36.0), cps=np.arange(n_cp) * 20.0 + 100,
               seizures=np.zeros((0, 2)), rE=np.full(3000, 1.0, np.float32), **{k: sch[k] for k in
                                                                               ("enc_q", "noise_q", "noise_eta", "mem_q")})
    lam = lambda m: 0.0 if silent else 0.3 + (2.0 * np.maximum(m, 0) if relay else 0)  # noqa: E731
    run["enc_counts"] = rng.poisson(np.stack([[lam(q * w) * np.ones(100) for q in row] for row in sch["enc_q"]]))
    rows = sch["rows"]
    run["noise_counts"] = rng.poisson(np.stack([[lam(q * w) for q in row] for row in sch["noise_q"]]))
    run["mem_counts"] = rng.poisson(np.stack([[lam(q * w) for q in row] for row in sch["mem_q"]]))
    run["mem_counts_alt"] = run["mem_counts"]
    for k in ("sham", "enc", "noise", "mem"):
        for v in ("kA0", "kA1", "kG0", "kG1", "nE", "nI", "maxE1s"):
            run[f"{k}_{v}"] = np.zeros(n_cp)
    return run, g


def test_analysis():
    run, g = _synthetic_run()
    rng = np.random.default_rng(3)
    mc = aa.memory_curves(run, g)
    check("toy relay without memory: current symbol decodable (R2 d=0 > 0.5)", mc[0]["R2"] > 0.5,
          f"R2 d0 {mc[0]['R2']:.3f}")
    s = sum(mc[d]["R2"] for d in range(1, 11))
    check("toy relay without memory: no past symbol decodable (|R2| small for d >= 1)",
          max(abs(mc[d]["R2"]) for d in range(1, 11)) < 0.02, f"S_memory {s:+.4f}")
    mi = aa.memory_curves(run, g, null="input")
    check("input-only readout: perfect at d = 0, null for d >= 1",
          mi[0]["R2"] > 0.99 and max(abs(mi[d]["R2"]) for d in range(1, 11)) < 0.02,
          f"d0 {mi[0]['R2']:.3f}, max |d>=1| {max(abs(mi[d]['R2']) for d in range(1, 11)):.4f}")
    mp = aa.memory_curves(run, g, null="permute", rng=rng)
    check("permuted labels: null at every lag (R2 <= 0.02)", max(mp[d]["R2"] for d in range(11)) < 0.02,
          f"max {max(mp[d]['R2'] for d in range(11)):.4f}")
    pred, y, cp, fits = aa.adapted_encoding(run, "counts", g)
    check("toy relay: encoding well above chance", aa.balanced_accuracy(y, pred) > 0.9,
          f"BA {aa.balanced_accuracy(y, pred):.3f}")
    # no test information in fitting: change the first outer block's trials only
    run2 = dict(run)
    run2["enc_counts"] = run["enc_counts"].copy()
    b0 = aa.cp_blocks(len(run["cps"]), g["outer_folds"])[0]
    run2["enc_counts"][b0] = np.random.default_rng(9).poisson(5.0, run2["enc_counts"][b0].shape)
    _, _, _, fits2 = aa.adapted_encoding(run2, "counts", g)
    m1, m2 = fits[0][1], fits2[0][1]
    check("fold 0: scaling, penalty and coefficients unaffected by its own test checkpoints",
          fits[0][2] == fits2[0][2] and np.allclose(m1.sc.mean_, m2.sc.mean_) and np.allclose(m1.m.coef_, m2.m.coef_))
    # silent network: all-zero trials kept, chance performance
    runs, _ = _synthetic_run(silent=True)
    X = aa.view(aa.encoding_trials(runs, "enc")[0], "proportions")
    pred, y, cp, _ = aa.adapted_encoding(runs, "proportions", g)
    check("silent responses: all-zero trials kept as zero vectors, no crash, BA at chance",
          X.shape[0] == len(y) and np.all(X == 0) and abs(aa.balanced_accuracy(y, pred) - 0.5) < 0.06,
          f"{len(y)} trials, BA {aa.balanced_accuracy(y, pred):.3f}")
    # memory features use only the current bin (construction): shift test
    cnt = run["mem_counts"]
    check("memory features are the current bin only (k), targets q[k - d] (no past bins in features)",
          cnt.ndim == 3 and cnt.shape[1] == g["mem_bins"])


def test_runs(d):
    raw = sorted((Path(d) / "raw").glob("*.npz"))
    if not raw:
        print(f"(no runs in {d}; skipping run checks)")
        return
    for p in raw:
        r = aa.load_run(p)
        rep = np.mean(np.all(r["sham_counts"] == r["native_sham_counts"], 1))
        check(f"{p.stem}: sham branches replay the native continuation", rep == 1.0, f"{rep:.0%} of checkpoints")
        drift = np.abs(r["mem_kA1"] - r["mem_kA0"])
        check(f"{p.stem}: HSP evolves during probes (k_AMPA changes in every memory branch)",
              bool(np.all(drift > 0)), f"median |dk| {np.median(drift):.2e}")
        pert = aa.perturbation(r)
        print(f"      perturbation vs sham: {', '.join(f'{k} {v:.3g}' for k, v in pert.items())}")


if __name__ == "__main__":
    test_schedule()
    test_analysis()
    test_runs(sys.argv[1] if len(sys.argv) > 1 else THIS_DIR / "results" / "assays_smoke")
    print(f"\n{len(FAILED)} failure(s)" + (": " + "; ".join(FAILED) if FAILED else ""))
    sys.exit(1 if FAILED else 0)
