"""Analysis of the HSP-active functional assays (reads <out>/raw/*.npz).

    python assay_analysis.py results/assays

Per (seed, condition):
  burden    S = ictal time / eligible time (existing detector: E rate > 10 Hz in
            1 s bins, eligible = [settle_s, run_s)), episodes per hour, mean
            duration (an episode reaching run_s is censored; its time counts).
  phases    each checkpoint is ictal (its branch windows touch a native
            seizure +- pad, or a branch itself crossed the detector), recovery
            (< recovery_s after a native seizure offset) or interictal.
            Checkpoints are uniform in time, so each carries the same weight.
  perturbation  probe branch vs paired sham: k_AMPA / k_GABA at branch end,
            E rate, branch episodes; tolerance = 3 SD of the sham's own k drift.
Metrics (splits by whole checkpoints: outer_folds contiguous blocks, gap_cps
dropped at the borders; scaling and penalties fitted on training checkpoints
only, inner blocked CV):
  C_encode  held-out balanced accuracy of the binary +/-w pattern (logistic),
            feature views: counts / proportions N_i/sum N (all-zero trials ->
            zero vector, kept) / total count; adapted readout and fixed
            reference readout (trained on the seed's 40 Hz reference only);
            thinning to a common spike budget as a sensitivity analysis.
  C_noise   the clean-trained decoder on corrupted presentations: BA(eta) and
            its trapezoidal area over eta in [0, 1].
  S_memory  ridge readout of q[k-d] from the current bin's last-20-ms counts,
            d = 0..10 (50-500 ms); held-out predictive R^2 per lag (may be
            negative, kept); S_memory = sum over d = 1..10. Nulls: an
            input-only readout (the current pattern) and labels permuted
            within train and within test, through the same pipeline.
Across seeds: condition means, seed-cluster bootstrap intervals, Pareto sets
(per metric vs burden, and joint) with bootstrap membership frequencies.
Outputs in <out>/analysis/: summary.csv, lag_curves.csv, noise_curves.csv,
pareto.csv, trials_<run>.csv and figures A-F.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import worker  # noqa: E402,F401
import conditions as C  # noqa: E402

REFERENCE = "reference 40Hz"
VIEWS = ("counts", "proportions", "total")


# =================================================================== loading
def load_run(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    d["g"] = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d["meta"]["grid"].items()}
    return d


def burden(run):
    g = run["g"]
    rE = run["rE"]
    s0, s1 = int(g["settle_s"]), int(round(g["run_s"]))
    ictal = rE[s0:s1] > g["seizure_hz"]
    sz = [(a, b) for a, b in run["seizures"] if b > s0]
    durs = [b - max(a, s0) for a, b in sz]
    return dict(burden=float(ictal.mean()), episodes_per_hour=len(sz) / ((s1 - s0) / 3600.0),
                mean_episode_s=float(np.mean(durs)) if durs else 0.0,
                censored_episode=bool(sz and sz[-1][1] >= s1))


def phases(run):
    g = run["g"]
    import assay_stimulus as ast
    longest = max(ast.branch_durations(g).values())
    pad0, pad1 = g["seizure_pad_s"]
    out = []
    for j, t in enumerate(run["cps"]):
        ictal = any(t + longest > a - pad0 and t < b + pad1 for a, b in run["seizures"])
        ictal |= any(run[f"{k}_maxE1s"][j] > g["seizure_hz"] for k in ("sham", "enc", "noise", "mem"))
        prev = [b for a, b in run["seizures"] if b <= t]
        if ictal:
            out.append("ictal")
        elif prev and t - max(prev) < g["recovery_s"]:
            out.append("recovery")
        else:
            out.append("interictal")
    return np.array(out)


def perturbation(run):
    sham_drift = run["sham_kA1"] - run["sham_kA0"]
    tol = 3 * np.std(sham_drift) if len(sham_drift) > 1 else np.inf
    g = run["g"]
    import assay_stimulus as ast
    dur = ast.branch_durations(g)
    res = dict(sham_kA_drift_sd=float(np.std(sham_drift)), tolerance_kA=float(tol))
    for k in ("enc", "noise", "mem"):
        # same duration as the sham only for mem; compare end-of-branch gains to
        # the sham's drift rescaled to the branch duration
        exp = sham_drift * dur[k] / dur["sham"]
        dk = (run[f"{k}_kA1"] - run[f"{k}_kA0"]) - exp
        res[f"{k}_kA_excess_mean"] = float(np.mean(dk))
        res[f"{k}_frac_beyond_tol"] = float(np.mean(np.abs(dk) > tol))
        res[f"{k}_E_rate_vs_sham"] = float(np.mean(run[f"{k}_nE"] / dur[k]) / max(np.mean(run["sham_nE"] / dur["sham"]), 1e-9))
        res[f"{k}_episodes_not_in_sham"] = int(np.sum((run[f"{k}_maxE1s"] > g["seizure_hz"])
                                                      & (run["sham_maxE1s"] <= g["seizure_hz"])))
    res["sham_replay_identical"] = float(np.mean(np.all(run["sham_counts"] == run["native_sham_counts"], 1)))
    return res


# ================================================================== features
def view(counts, v):
    counts = counts.astype(float)
    tot = counts.sum(1, keepdims=True)
    if v == "counts":
        return counts
    if v == "total":
        return tot
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(tot > 0, counts / np.where(tot > 0, tot, 1), 0.0)
    return p


def thin(counts, budget, rng):
    """Binomially thin each trial's counts toward a common total; trials
    already below the budget are kept as they are (and flagged)."""
    tot = counts.sum(1)
    p = np.where(tot > budget, budget / np.maximum(tot, 1), 1.0)
    return rng.binomial(counts.astype(np.int64), p[:, None]), tot < budget


# ================================================================= decoders
def balanced_accuracy(y, yhat):
    cls = np.unique(y)
    return float(np.mean([np.mean(yhat[y == c] == c) for c in cls]))


def cp_blocks(n_cp, k):
    return [b for b in np.array_split(np.arange(n_cp), k) if len(b)]


def _train_cps(n_cp, block, gap):
    lo, hi = block[0], block[-1]
    return np.r_[np.arange(0, max(lo - gap, 0)), np.arange(min(hi + 1 + gap, n_cp), n_cp)]


class Logit:
    def __init__(self, C):
        self.C = C

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        self.const = None
        if len(np.unique(y)) < 2:              # prior-only fallback (flagged)
            self.const = y[0]
            return self
        self.m = LogisticRegression(C=self.C, max_iter=2000).fit(self.sc.transform(X), y)
        return self

    def predict(self, X):
        if self.const is not None:
            return np.full(len(X), self.const)
        return self.m.predict(self.sc.transform(X))


class RidgeR:
    """Ridge with intercept on CENTRED counts (training means only). No
    per-feature scaling: cells that almost never fire in the 20 ms window
    would get a near-zero SD and turn a rare test spike into a huge input."""
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        self.sc = StandardScaler(with_std=False).fit(X)
        self.m = Ridge(alpha=self.alpha).fit(self.sc.transform(X), y)
        return self

    def predict(self, X):
        return self.m.predict(self.sc.transform(X))


def r2_test(y, yhat):
    sst = float(((y - y.mean()) ** 2).sum())
    return np.nan if sst == 0 else 1.0 - float(((y - yhat) ** 2).sum()) / sst


def select(make, X, y, cp_of, cps, grid, inner, gap, score):
    """Inner blocked CV over the training checkpoints `cps` only."""
    best, best_s = grid[0], -np.inf
    blocks = cp_blocks(len(cps), inner)
    for p in grid:
        yt, yp = [], []
        for b in blocks:
            tr_c = cps[_train_cps(len(cps), b, gap)]
            te_c = cps[b]
            tr, te = np.isin(cp_of, tr_c), np.isin(cp_of, te_c)
            if tr.sum() < 5 or te.sum() == 0:
                continue
            m = make(p).fit(X[tr], y[tr])
            yt.append(y[te])
            yp.append(m.predict(X[te]))
        if not yt:
            continue
        s = score(np.concatenate(yt), np.concatenate(yp))
        if s > best_s:
            best, best_s = p, s
    return best


def fit_selected(make, X, y, cp_of, cps, grid, g, score):
    p = select(make, X, y, cp_of, cps, grid, g["inner_folds"], g["gap_cps"], score)
    tr = np.isin(cp_of, cps)
    return make(p).fit(X[tr], y[tr]), p


# ============================================================ per-run assays
def encoding_trials(run, kind="enc"):
    """(counts (n,100), y in {0,1}, checkpoint index, eta) without lead-ins."""
    cnt = run[f"{kind}_counts"][:, 1:]
    q = run[f"{kind}_q"][:, 1:]
    n_cp, n_tr = q.shape
    eta = run["noise_eta"][:, 1:] if kind == "noise" else np.zeros_like(q, float)
    return (cnt.reshape(-1, cnt.shape[-1]), (q.reshape(-1) > 0).astype(int),
            np.repeat(np.arange(n_cp), n_tr), eta.reshape(-1))


def adapted_encoding(run, v, g, counts_override=None):
    X0, y, cp_of, _ = encoding_trials(run, "enc")
    X = view(counts_override if counts_override is not None else X0, v)
    n_cp = len(run["cps"])
    pred = np.full(len(y), -1)
    fits = []
    for b in cp_blocks(n_cp, g["outer_folds"]):
        tr_c = _train_cps(n_cp, b, g["gap_cps"])
        m, p = fit_selected(Logit, X, y, cp_of, tr_c, g["C_grid"], g, balanced_accuracy)
        te = np.isin(cp_of, b)
        pred[te] = m.predict(X[te])
        fits.append((b, m, p))
    return pred, y, cp_of, fits


def noise_curve(run, fits_or_model, v, fixed=False):
    """BA(eta) of clean-trained decoders on corrupted held-out presentations."""
    X0, y, cp_of, eta = encoding_trials(run, "noise")
    X = view(X0, v)
    pred = np.full(len(y), -1)
    if fixed:
        pred = fits_or_model.predict(X)
    else:
        for b, m, _ in fits_or_model:
            te = np.isin(cp_of, b)
            pred[te] = m.predict(X[te])
    return y, pred, eta, cp_of


def memory_curves(run, g, key="mem_counts", null=None, rng=None, test_mask=None):
    """Held-out R^2 per lag. null: None | 'input' | 'permute'. test_mask
    (per checkpoint): score only those held-out checkpoints (training is
    unchanged)."""
    cnt = run[key].astype(float)
    q = run["mem_q"].astype(float)
    n_cp, nb, nE = cnt.shape
    L = g["mem_max_lag"]
    ks = np.arange(L, nb)
    w = np.where(np.arange(nE) % 10 < 5, 1.0, -1.0)
    out = {}
    for d in range(L + 1):
        X = (q[:, ks, None] * w[None, None, :]).reshape(-1, nE) if null == "input" else cnt[:, ks].reshape(-1, nE)
        y = q[:, ks - d].reshape(-1)
        cp_of = np.repeat(np.arange(n_cp), len(ks))
        yt, yp = [], []
        for b in cp_blocks(n_cp, g["outer_folds"]):
            tr_c = _train_cps(n_cp, b, g["gap_cps"])
            tr, te = np.isin(cp_of, tr_c), np.isin(cp_of, b)
            if test_mask is not None:
                te &= test_mask[cp_of]
            ytr, yte = y.copy(), y[te].copy()
            if null == "permute":
                ytr[tr] = rng.permutation(ytr[tr])
                yte = rng.permutation(yte)
            if not te.any():
                continue
            m, _ = fit_selected(RidgeR, X, ytr, cp_of, tr_c, g["ridge_grid"], g, r2_test)
            yt.append(yte)
            yp.append(m.predict(X[te]))
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        out[d] = dict(R2=r2_test(yt, yp), r2=float(np.corrcoef(yt, yp)[0, 1] ** 2) if yp.std() > 0 else 0.0)
    return out


def analyse_seed(paths, g_all):
    """All conditions of one network seed (the fixed reference readout needs
    the seed's reference)."""
    runs = {load_run(p)["meta"]["condition"]: load_run(p) for p in paths}
    any_run = next(iter(runs.values()))
    g = dict(any_run["g"], **{k: g_all[k] for k in ("outer_folds", "inner_folds", "gap_cps", "C_grid",
                                                     "ridge_grid", "thin_repeats") if k in g_all})
    seed = any_run["meta"]["seed"]
    rng = np.random.default_rng(1000 + seed)
    ref_models = {}
    if REFERENCE in runs:
        r = runs[REFERENCE]
        X0, y, cp_of, _ = encoding_trials(r, "enc")
        for v in VIEWS:
            ref_models[v], _ = fit_selected(Logit, view(X0, v), y, cp_of, np.arange(len(r["cps"])),
                                            g["C_grid"], g, balanced_accuracy)
    # common spike budget for thinning: median trial total of the lowest-rate condition
    totals = {c: encoding_trials(r, "enc")[0].sum(1) for c, r in runs.items()}
    budget = min(np.median(t) for t in totals.values())
    summaries, lags, noise_rows, trials = [], [], [], []
    for c, run in runs.items():
        ph = phases(run)
        bur = burden(run)
        pert = perturbation(run)
        base = dict(seed=seed, condition=c, A_hz=run["meta"]["A_hz"], b_hz=run["meta"]["freq_hz"], **bur)
        X0, y, cp_of, _ = encoding_trials(run, "enc")
        no_spike = float(np.mean(X0.sum(1) == 0))
        row = dict(base, no_spike_frac=no_spike, n_checkpoints=len(run["cps"]),
                   **{f"phase_{p}": float(np.mean(ph == p)) for p in ("ictal", "recovery", "interictal")},
                   **{f"pert_{k}": v for k, v in pert.items()})
        enc_fits = {}
        for v in VIEWS:
            pred, yy, cpo, fits = adapted_encoding(run, v, g)
            enc_fits[v] = fits
            ok = pred >= 0
            row[f"C_encode_adapted_{v}"] = balanced_accuracy(yy[ok], pred[ok])
            for p_ in ("ictal", "recovery", "interictal"):
                s = ok & (ph[cpo] == p_)
                row[f"C_encode_adapted_{v}_{p_}"] = balanced_accuracy(yy[s], pred[s]) if s.sum() and len(np.unique(yy[s])) > 1 else np.nan
            if v == "counts":
                for i in np.flatnonzero(ok):
                    trials.append(dict(seed=seed, condition=c, assay="encoding", regime="adapted", view=v,
                                       checkpoint=int(cpo[i]), t_cp=float(run["cps"][cpo[i]]), phase=ph[cpo[i]],
                                       target=int(yy[i]), prediction=int(pred[i]),
                                       kA_drift=float(run["enc_kA1"][cpo[i]] - run["enc_kA0"][cpo[i]])))
            if v in ref_models:
                if c == REFERENCE:
                    row[f"C_encode_fixed_{v}"] = row[f"C_encode_adapted_{v}"]   # held-out folds of its own data
                else:
                    pf = ref_models[v].predict(view(X0, v))
                    row[f"C_encode_fixed_{v}"] = balanced_accuracy(y, pf)
            # noise robustness (adapted: clean-trained per fold; fixed: reference-trained)
            for regime in ("adapted", "fixed"):
                if regime == "adapted" or c == REFERENCE:
                    src, fixed = enc_fits[v], False        # the reference's fixed readout = its own held-out folds
                elif v in ref_models:
                    src, fixed = ref_models[v], True
                else:
                    continue
                yn, pn, eta, _ = noise_curve(run, src, v, fixed=fixed)
                curve = []
                for e in sorted(np.unique(eta)):
                    s = (eta == e) & (pn >= 0)
                    ba = balanced_accuracy(yn[s], pn[s]) if s.sum() and len(np.unique(yn[s])) > 1 else np.nan
                    curve.append((e, ba))
                    noise_rows.append(dict(seed=seed, condition=c, regime=regime, view=v, eta=e, BA=ba))
                es, bs = np.array([x[0] for x in curve]), np.array([x[1] for x in curve])
                row[f"C_noise_{regime}_{v}"] = float(np.trapz(bs, es))
                row[f"BA_eta0_{regime}_{v}"] = float(bs[0])
        # thinning sensitivity (counts view, adapted)
        bas, below = [], []
        for _ in range(g["thin_repeats"]):
            Xt, low = thin(X0, budget, rng)
            pred, yy, _, _ = adapted_encoding(run, "counts", g, counts_override=Xt)
            ok = pred >= 0
            bas.append(balanced_accuracy(yy[ok], pred[ok]))
            below.append(low.mean())
        row.update(C_encode_thinned_counts=float(np.mean(bas)), thin_budget=float(budget),
                   thin_frac_below_budget=float(np.mean(below)))
        # memory
        for key, tag in (("mem_counts", ""), ("mem_counts_alt", "_10ms")):
            mc = memory_curves(run, g, key)
            row[f"S_memory{tag}"] = float(np.nansum([mc[d]["R2"] for d in range(1, g["mem_max_lag"] + 1)]))
            row[f"R2_d0{tag}"] = mc[0]["R2"]
            for d, v_ in mc.items():
                lags.append(dict(seed=seed, condition=c, window=f"{20 if not tag else 10}ms", kind="actual",
                                 lag=d, lag_ms=d * g["mem_bin_ms"], **v_))
        mi = memory_curves(run, g, "mem_counts", test_mask=(ph == "interictal"))
        row["S_memory_interictal"] = float(np.nansum([mi[d]["R2"] for d in range(1, g["mem_max_lag"] + 1)]))
        row["R2_d0_interictal"] = mi[0]["R2"]
        for null in ("input", "permute"):
            mc = memory_curves(run, g, "mem_counts", null=null, rng=rng)
            row[f"S_memory_null_{null}"] = float(np.nansum([mc[d]["R2"] for d in range(1, g["mem_max_lag"] + 1)]))
            for d, v_ in mc.items():
                lags.append(dict(seed=seed, condition=c, window="20ms", kind=f"null_{null}",
                                 lag=d, lag_ms=d * g["mem_bin_ms"], **v_))
        row["E_rate_native_interictal"] = float(np.mean(run["rE"][int(g["settle_s"]):][
            run["rE"][int(g["settle_s"]):] <= g["seizure_hz"]]))
        summaries.append(row)
    return summaries, lags, noise_rows, trials


# ==================================================================== Pareto
def pareto_set(points):
    """points: {name: (S, C)}; minimise S, maximise C."""
    names = list(points)
    keep = []
    for a in names:
        Sa, Ca = points[a]
        dominated = any((points[b][0] <= Sa and points[b][1] >= Ca and
                         (points[b][0] < Sa or points[b][1] > Ca)) for b in names if b != a)
        if not dominated and np.isfinite(Ca):
            keep.append(a)
    return keep


def pareto_joint(points):
    """points: {name: (S, C1, C2, C3)}: minimise S, maximise the rest."""
    names = list(points)
    keep = []
    for a in names:
        pa = np.array(points[a], float)
        dom = False
        for b in names:
            if b == a:
                continue
            pb = np.array(points[b], float)
            better_eq = pb[0] <= pa[0] and np.all(pb[1:] >= pa[1:])
            strict = pb[0] < pa[0] or np.any(pb[1:] > pa[1:])
            if better_eq and strict:
                dom = True
                break
        if not dom:
            keep.append(a)
    return keep


METRICS = ("C_encode_adapted_counts", "C_noise_adapted_counts", "S_memory")


def pareto_analysis(summ, n_boot, rng):
    conds = sorted({r["condition"] for r in summ})
    seeds = sorted({r["seed"] for r in summ})

    def means(sample):
        out = {}
        for c in conds:
            R = [r for s in sample for r in summ if r["seed"] == s and r["condition"] == c]
            if R:
                out[c] = {k: float(np.mean([r[k] for r in R])) for k in ("burden",) + METRICS}
        return out

    m = means(seeds)
    rows = []
    sets = {k: pareto_set({c: (v["burden"], v[k]) for c, v in m.items()}) for k in METRICS}
    joint = pareto_joint({c: (v["burden"],) + tuple(v[k] for k in METRICS) for c, v in m.items()})
    freq = {k: {c: 0 for c in conds} for k in METRICS + ("joint",)}
    for _ in range(n_boot):
        mb = means(list(rng.choice(seeds, len(seeds), replace=True)))
        for k in METRICS:
            for c in pareto_set({c: (v["burden"], v[k]) for c, v in mb.items()}):
                freq[k][c] += 1
        for c in pareto_joint({c: (v["burden"],) + tuple(v[k] for k in METRICS) for c, v in mb.items()}):
            freq["joint"][c] += 1
    for c in conds:
        rows.append(dict(condition=c, **{f"pareto_{k}": c in sets[k] for k in METRICS},
                         pareto_joint=c in joint, **{f"freq_{k}": freq[k][c] / n_boot for k in METRICS},
                         freq_joint=freq["joint"][c] / n_boot,
                         **{f"mean_{k}": m[c][k] for k in ("burden",) + METRICS}))
    return rows


def write_csv(rows, path):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(out_dir, n_jobs=None):
    from joblib import Parallel, delayed
    import os
    import assay_plots
    out_dir = Path(out_dir)
    raw = sorted((out_dir / "raw").glob("*.npz"))
    if not raw:
        raise SystemExit(f"no runs in {out_dir / 'raw'}")
    g_all = dict(C.ASSAY_GRID)
    by_seed = {}
    for p in raw:
        m = json.loads(str(np.load(p)["meta"]))
        by_seed.setdefault(m["seed"], []).append(p)
    n_jobs = n_jobs or max(1, min(len(by_seed), (os.cpu_count() or 4) - 2))
    print(f"analysing {len(raw)} run(s) over {len(by_seed)} seed(s) ...", flush=True)
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(analyse_seed)(v, g_all) for v in by_seed.values())
    summ = [r for x in res for r in x[0]]
    lags = [r for x in res for r in x[1]]
    noise = [r for x in res for r in x[2]]
    trials = [r for x in res for r in x[3]]
    ana = out_dir / "analysis"
    (ana / "plots").mkdir(parents=True, exist_ok=True)
    write_csv(summ, ana / "summary.csv")
    write_csv(lags, ana / "lag_curves.csv")
    write_csv(noise, ana / "noise_curves.csv")
    write_csv(trials, ana / "trials_encoding.csv")
    pareto = pareto_analysis(summ, g_all["n_boot"], np.random.default_rng(7))
    write_csv(pareto, ana / "pareto.csv")
    with open(ana / "effective_config.json", "w") as fh:
        json.dump(dict(analysis=g_all, run_meta=json.loads(str(np.load(raw[0])["meta"]))), fh, indent=1,
                  default=str)
    assay_plots.make_all(out_dir, summ, lags, noise, pareto, ana / "plots")
    for r in sorted(summ, key=lambda r: (r["seed"], r["condition"])):
        print(f"seed{r['seed']} {r['condition']:15s} burden {r['burden']:.4f} | C_encode {r['C_encode_adapted_counts']:.3f} "
              f"(fixed {r.get('C_encode_fixed_counts', np.nan):.3f}, prop {r['C_encode_adapted_proportions']:.3f}, "
              f"total {r['C_encode_adapted_total']:.3f}, thinned {r['C_encode_thinned_counts']:.3f}) | "
              f"C_noise {r['C_noise_adapted_counts']:.3f} | S_memory {r['S_memory']:+.3f} (R2 d0 {r['R2_d0']:.3f}; "
              f"nulls input {r['S_memory_null_input']:+.3f}, permuted {r['S_memory_null_permute']:+.3f}) | "
              f"interictal S_memory {r['S_memory_interictal']:+.3f} (d0 {r['R2_d0_interictal']:.3f}) | "
              f"probe-induced episodes enc/noise/mem {r['pert_enc_episodes_not_in_sham']}/"
              f"{r['pert_noise_episodes_not_in_sham']}/{r['pert_mem_episodes_not_in_sham']} of {r['n_checkpoints']} | "
              f"replay {r['pert_sham_replay_identical']:.0%}", flush=True)
    print(f"wrote {ana}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else THIS_DIR / "results" / "assays")
