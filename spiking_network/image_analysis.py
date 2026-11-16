"""Image-recovery analysis (reads <out>/raw/*.npz from image_cycle.py).

    python image_analysis.py results/image_random [--fast]

Trials: one per image presentation. Features for readout duration T: E spike
counts per cell in [onset + response_start, onset + response_start + T)
(schedule bins, so exact). Trials before settle_s are dropped; a trial is
"seizure-overlapping" if its response span (the longest T) comes within
seizure_pad_s of a seizure; it is "train-eligible" (interictal) if it is
more than train_gap_s from any seizure.

Decoders (one regularised logistic regression per pixel, intercept, features
standardised with the training subset only, unweighted -- the white prior
q = 0.1 is kept):
  local       -- pixel i from its own cell's count x_i            (primary)
  population  -- pixel i from all 100 counts                       (secondary)
  count       -- pixel i from the total count (diagnostic: the input
                 brightness is matched, so this should carry little)
Regularisation C: inner blocked CV on the training trials (mean held-out
log-likelihood); ONE shared decision threshold per fitted decoder, chosen on
the same training-only out-of-fold probabilities to maximise mean trial
balanced accuracy. Pixels whose training labels are constant fall back to
the training prior (flagged).

Readout regimes:
  adapted   -- within each run: outer_folds contiguous blocks of trials; each
               block is predicted by a decoder fitted on train-eligible trials
               outside it (gap_trials dropped at the borders). Every trial,
               seizure-overlapping ones included, is predicted out-of-sample.
  heldout   -- adapted, but folds over MASK IDs: test trials show masks never
               seen in fitting or tuning (generalisation to new images).
  frozen    -- per seed: fitted once on the first frozen_train_frac of the
               steady 40 Hz reference's train-eligible trials, everything
               (scaling, C, coefficients, threshold) frozen, then applied to
               the rest of the reference and to every other condition.
  null      -- adapted pipeline re-run with mask labels permuted among trials
               (within seizure-overlap strata; same permutation for all
               pixels); n_perm repeats.
Scores per trial: balanced accuracy BA = (white recall + black specificity)/2
(all-black = 0.5), recall, specificity, spatial AUROC (white vs black pixels
ranked by probability, ties 0.5), Brier skill S_B = 1 - MSE / (q(1-q)).
Nothing is divided by the firing rate.

Outputs in <out>/analysis/: <run>_trials.npz (per-trial scores, splits,
fitted settings), summary.csv, and plots/ (reconstructions, BA vs readout
duration, recall/specificity and BA vs E rate, cycle traces per run).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import worker  # noqa: E402,F401  (sys.path)
import conditions as cond_mod  # noqa: E402

MODELS = ("local", "population", "count")
REGIMES = ("adapted", "heldout", "frozen", "null")
REFERENCE = "reference 40Hz"


# ======================================================================= data
def load_run(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    d["path"] = str(path)
    return d


def trial_table(run, g):
    """Per-trial metadata and features for every readout duration."""
    sb = g["sched_ms"]
    on = run["onset"].astype(np.int64)
    start = g["response_start_ms"] // sb
    E = run["E_counts"]                     # uint8/uint16; never copied to float as a whole
    feats = {}
    for T in g["readout_ms"]:
        a, b = on + start, on + start + T // sb
        # windows are disjoint and ordered: reduceat over [a0, b0, a1, b1, ...]
        # sums [a_j, b_j) at even positions (integer accumulator, no overflow)
        idx = np.empty(2 * len(a), np.int64)
        idx[0::2], idx[1::2] = a, b
        sums = np.add.reduceat(E, np.minimum(idx, len(E) - 1), axis=0, dtype=np.int64)
        feats[T] = sums[0::2].astype(np.float64)
    onset_s = on * sb / 1000.0
    span_end_s = (on + start + max(g["readout_ms"]) // sb) * sb / 1000.0
    sz = np.asarray(run["seizures"]).reshape(-1, 2)
    pad0, pad1 = g["seizure_pad_s"]
    overlap = np.zeros(len(on), bool)
    near = np.zeros(len(on), bool)
    for s0, s1 in sz:
        overlap |= (span_end_s > s0 - pad0) & (onset_s < s1 + pad1)
        near |= (span_end_s > s0 - g["train_gap_s"]) & (onset_s < s1 + g["train_gap_s"])
    post = onset_s >= g["settle_s"]
    t_since = np.full(len(on), np.nan)
    t_to = np.full(len(on), np.nan)
    for j, t in enumerate(onset_s):
        prv = sz[sz[:, 1] <= t, 1] if len(sz) else np.array([])
        nxt = sz[sz[:, 0] > t, 0] if len(sz) else np.array([])
        t_since[j] = t - prv.max() if prv.size else np.nan
        t_to[j] = nxt.min() - t if nxt.size else np.nan
    Y = run["masks"][run["mask_id"]].astype(np.float64)
    return dict(onset_s=onset_s, mask_id=run["mask_id"].astype(int), Y=Y, feats=feats, post=post,
                seizure_overlap=overlap, train_ok=post & ~near & ~overlap,
                t_since=t_since, t_to=t_to)


# ==================================================================== decoders
def _sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _design(X, kind):
    """Per-pixel feature matrix: local -> (n, 100) column i for pixel i;
    count -> (n, 1) total count shared by all pixels; population -> (n, 100)."""
    if kind == "count":
        return X.sum(1, keepdims=True)
    return X


def fit_logit(X, Y, kind, C, max_iter=50):
    """Fit one L2-logistic regression per pixel (intercept unpenalised).
    Returns a dict with the training scaler, coefficients and flags."""
    F = _design(X, kind)
    mu = F.mean(0)
    sd = F.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (F - mu) / sd
    prior = Y.mean(0)
    const = (prior <= 0) | (prior >= 1)
    b0 = np.log(np.clip(prior, 1e-6, 1 - 1e-6) / np.clip(1 - prior, 1e-6, 1))
    P = Y.shape[1]
    if kind in ("local", "count"):
        Zp = Z if kind == "local" else np.repeat(Z, P, axis=1)
        w = np.zeros(P)
        b = b0.copy()
        for _ in range(max_iter):
            p = _sig(b + w * Zp)
            r = p - Y
            v = p * (1 - p)
            gb, gw = r.sum(0), (r * Zp).sum(0) + w / C
            hbb, hbw, hww = v.sum(0) + 1e-9, (v * Zp).sum(0), (v * Zp * Zp).sum(0) + 1.0 / C
            det = hbb * hww - hbw ** 2
            db = (hww * gb - hbw * gw) / det
            dw = (-hbw * gb + hbb * gw) / det
            db, dw = np.clip(db, -5, 5), np.clip(dw, -5, 5)
            db[const], dw[const] = 0.0, 0.0
            b -= db
            w -= dw
            if max(np.abs(db).max(), np.abs(dw).max()) < 1e-8:
                break
        b[const] = b0[const]
        W = w
    else:  # population
        free = ~const

        def f(theta):
            Wm = theta[:P * Z.shape[1]].reshape(Z.shape[1], P)
            bb = theta[P * Z.shape[1]:]
            eta = Z @ Wm + bb
            p = _sig(eta)
            loss = (np.logaddexp(0, eta) - Y * eta)[:, free].sum() + (Wm[:, free] ** 2).sum() / (2 * C)
            r = (p - Y) * free
            gW = Z.T @ r + Wm * free / C
            gb = r.sum(0)
            return loss, np.concatenate([gW.ravel(), gb])

        theta0 = np.concatenate([np.zeros(Z.shape[1] * P), b0])
        res = optimize.minimize(f, theta0, jac=True, method="L-BFGS-B",
                                options=dict(maxiter=500, gtol=1e-6))
        W = res.x[:P * Z.shape[1]].reshape(Z.shape[1], P)
        b = res.x[P * Z.shape[1]:]
        W[:, const] = 0.0
        b[const] = b0[const]
    return dict(kind=kind, mu=mu, sd=sd, W=W, b=b, C=float(C), prior_only=const)


def predict_proba(fit, X):
    Z = (_design(X, fit["kind"]) - fit["mu"]) / fit["sd"]
    if fit["kind"] == "local":
        eta = fit["b"] + fit["W"] * Z
    elif fit["kind"] == "count":
        eta = fit["b"] + fit["W"] * Z            # (n, 1) broadcast against (P,)
    else:
        eta = Z @ fit["W"] + fit["b"]
    return _sig(eta)


# ===================================================================== scores
def trial_scores(S, Y, thr, q):
    """Per-trial BA, recall, specificity, AUROC, Brier skill."""
    pred = S >= thr
    K = Y.sum(1)
    recall = (Y * pred).sum(1) / K
    spec = ((1 - Y) * (~pred)).sum(1) / (Y.shape[1] - K)
    ranks = stats.rankdata(S, axis=1)                       # average ranks: ties -> 0.5
    auc = ((ranks * Y).sum(1) - K * (K + 1) / 2) / (K * (Y.shape[1] - K))
    mse = ((S - Y) ** 2).mean(1)
    return dict(BA=(recall + spec) / 2, recall=recall, spec=spec, AUROC=auc,
                SB=1.0 - mse / (q * (1 - q)))


def _loglik(S, Y):
    S = np.clip(S, 1e-9, 1 - 1e-9)
    return float((Y * np.log(S) + (1 - Y) * np.log(1 - S)).mean())


THRESHOLDS = np.unique(np.r_[np.linspace(0.01, 0.99, 99), 0.1])


def choose_threshold(S, Y, q):
    """Shared threshold maximising mean trial BA (ties: closest to q)."""
    best, best_ba = q, -1.0
    for t in THRESHOLDS:
        ba = trial_scores(S, Y, t, q)["BA"].mean()
        if ba > best_ba + 1e-12 or (abs(ba - best_ba) <= 1e-12 and abs(t - q) < abs(best - q)):
            best, best_ba = float(t), ba
    return best


def _blocked(n, k):
    return [c for c in np.array_split(np.arange(n), k) if len(c)]


def fit_selected(X, Y, kind, g, q):
    """Inner blocked CV on THESE training trials only: C by held-out
    log-likelihood, then the threshold from the same out-of-fold
    probabilities; refit on all of them. Returns the frozen-able fit."""
    n = len(X)
    folds = _blocked(n, g["inner_folds"])
    gap = g["gap_trials"]
    best_C, best_ll, best_oof = None, -np.inf, None
    for C in g["C_grid"]:
        oof = np.empty_like(Y)
        for fd in folds:
            lo, hi = fd[0], fd[-1]
            tr = np.r_[np.arange(0, max(lo - gap, 0)), np.arange(min(hi + 1 + gap, n), n)]
            oof[fd] = predict_proba(fit_logit(X[tr], Y[tr], kind, C), X[fd])
        ll = _loglik(oof, Y)
        if ll > best_ll:
            best_C, best_ll, best_oof = C, ll, oof
    fit = fit_logit(X, Y, kind, best_C)
    fit["threshold"] = choose_threshold(best_oof, Y, q)
    fit["inner_loglik"] = best_ll
    return fit


# =================================================================== regimes
def run_adapted(X, Y, tt, kind, g, q, mask_folds=False):
    """Out-of-sample probabilities for every post-settle trial. Folds are
    contiguous time blocks (adapted) or groups of mask IDs (heldout)."""
    post = np.flatnonzero(tt["post"])
    S = np.full(Y.shape, np.nan)
    thr = np.full(len(Y), np.nan)
    fold_of = np.full(len(Y), -1)
    settings = []
    if mask_folds:
        ids = np.unique(tt["mask_id"][post])
        groups = np.array_split(ids, g["heldout_mask_folds"])
        tests = [post[np.isin(tt["mask_id"][post], grp)] for grp in groups]
    else:
        tests = [post[c] for c in _blocked(len(post), g["outer_folds"])]
    for k, test in enumerate(tests):
        if mask_folds:
            tr = post[tt["train_ok"][post] & ~np.isin(tt["mask_id"][post], tt["mask_id"][test])]
        else:
            lo, hi = test[0], test[-1]
            pos_lo, pos_hi = np.searchsorted(post, lo), np.searchsorted(post, hi)
            keep = np.r_[post[:max(pos_lo - g["gap_trials"], 0)], post[pos_hi + 1 + g["gap_trials"]:]]
            tr = keep[tt["train_ok"][keep]]
        if len(tr) < 20 * g["inner_folds"]:
            continue
        fit = fit_selected(X[tr], Y[tr], kind, g, q)
        S[test] = predict_proba(fit, X[test])
        thr[test] = fit["threshold"]
        fold_of[test] = k
        settings.append(dict(fold=k, n_train=len(tr), C=fit["C"], threshold=fit["threshold"],
                             n_prior_only=int(fit["prior_only"].sum())))
    return S, thr, fold_of, settings


def permuted_labels(tt, run, rng):
    """Mask labels permuted among post-settle trials within seizure-overlap
    strata (one permutation for all pixels)."""
    mid = tt["mask_id"].copy()
    for stratum in (True, False):
        idx = np.flatnonzero(tt["post"] & (tt["seizure_overlap"] == stratum))
        mid[idx] = mid[rng.permutation(idx)]
    return run["masks"][mid].astype(np.float64)


def analyse_run(path, g, models=MODELS, regimes=("adapted", "heldout", "null")):
    """Adapted / heldout / null for one run (frozen needs the seed's
    reference and runs per seed). Returns per-trial arrays + settings."""
    run = load_run(path)
    tt = trial_table(run, g)
    q = g["k_white"] / 100.0
    out = dict(meta=run["meta"], path=str(path), tt={k: v for k, v in tt.items() if k != "feats"},
               res={})
    rng = np.random.default_rng(777 + run["meta"]["seed"])
    for T in g["readout_ms"]:
        X = tt["feats"][T]
        for kind in models:
            for regime in regimes:
                if regime == "null":
                    for rep in range(g["n_perm"]):
                        Yp = permuted_labels(tt, run, rng)
                        S, thr, fold, st = run_adapted(X, Yp, tt, kind, g, q)
                        out["res"][(T, kind, f"null{rep}")] = _pack(S, thr, fold, st, tt["Y"], q)
                    continue
                S, thr, fold, st = run_adapted(X, tt["Y"], tt, kind, g, q,
                                               mask_folds=(regime == "heldout"))
                out["res"][(T, kind, regime)] = _pack(S, thr, fold, st, tt["Y"], q)
    out["example_counts"] = {T: tt["feats"][T] for T in (g["primary_readout_ms"],)}
    return out


def _pack(S, thr, fold, settings, Y, q):
    ok = ~np.isnan(thr)
    sc = {k: np.full(len(Y), np.nan) for k in ("BA", "recall", "spec", "AUROC", "SB")}
    if ok.any():
        s = trial_scores(S[ok], Y[ok], thr[ok][:, None], q)
        for k in sc:
            sc[k][ok] = s[k]
    return dict(S=S.astype(np.float32), thr=thr, fold=fold, settings=settings, **sc)


def analyse_frozen(paths_by_cond, g, models=MODELS):
    """Per seed: decoders fitted once on the reference's early train-eligible
    trials, frozen, applied to the rest of the reference and to every other
    condition. Returns {(cond, T, kind): packed scores}."""
    if REFERENCE not in paths_by_cond:
        return {}
    runs_by_cond = {c: load_run(p) for c, p in paths_by_cond.items()}
    q = g["k_white"] / 100.0
    tts = {c: trial_table(r, g) for c, r in runs_by_cond.items()}
    tr_ref = tts[REFERENCE]
    post = np.flatnonzero(tr_ref["post"])
    n_fit = int(len(post) * g["frozen_train_frac"])
    fit_idx = post[:n_fit][tr_ref["train_ok"][post[:n_fit]]]
    out = {}
    for T in g["readout_ms"]:
        for kind in models:
            fit = fit_selected(tr_ref["feats"][T][fit_idx], tr_ref["Y"][fit_idx], kind, g, q)
            frozen = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in fit.items()}
            for c, tt in tts.items():
                idx = np.flatnonzero(tt["post"])
                if c == REFERENCE:
                    idx = idx[idx > (post[n_fit - 1] + g["gap_trials"] if n_fit else -1)]
                S = np.full(tt["Y"].shape, np.nan)
                thr = np.full(len(tt["Y"]), np.nan)
                S[idx] = predict_proba(frozen, tt["feats"][T][idx])
                thr[idx] = frozen["threshold"]
                out[(c, T, kind)] = _pack(S, thr, np.zeros(len(thr), int),
                                          [dict(C=frozen["C"], threshold=frozen["threshold"],
                                                n_train=len(fit_idx))], tt["Y"], q)
            # the frozen parameters must not have changed during evaluation
            assert all(np.array_equal(frozen[k], fit[k]) for k in ("W", "b", "mu", "sd"))
    return out


# ================================================================== outputs
def summarise(per_run, frozen, g):
    rows = []
    T0 = g["primary_readout_ms"]
    for stem, o in per_run.items():
        m, tt = o["meta"], o["tt"]
        items = list(o["res"].items())
        items += [((T, kind, "frozen"), v) for (c, T, kind), v in frozen.get(m["seed"], {}).items()
                  if c == m["condition"]]
        for (T, kind, regime), r in items:
            for phase, sel in (("all", tt["post"]), ("interictal", tt["post"] & ~tt["seizure_overlap"]),
                               ("seizure_overlap", tt["post"] & tt["seizure_overlap"])):
                s = sel & ~np.isnan(r["BA"])
                if not s.any():
                    continue
                rows.append(dict(seed=m["seed"], condition=m["condition"], T_ms=T, model=kind,
                                 regime=regime, phase=phase, n_trials=int(s.sum()),
                                 **{k: float(np.mean(r[k][s])) for k in ("BA", "recall", "spec", "AUROC", "SB")}))
    return rows


def descriptives(per_run, g):
    T0 = g["primary_readout_ms"]
    out = {}
    for stem, o in per_run.items():
        tt, m = o["tt"], o["meta"]
        X = o["example_counts"][T0]
        Y = tt["Y"]
        s = tt["post"] & ~tt["seizure_overlap"]
        rate = X[s].sum(1) / (100 * T0 / 1000)
        wb = (X[s] * Y[s]).sum(1) / Y[s].sum(1) - (X[s] * (1 - Y[s])).sum(1) / (1 - Y[s]).sum(1)
        out[stem] = dict(seed=m["seed"], condition=m["condition"], E_rate=float(rate.mean()),
                         white_minus_black=float(wb.mean()), count_sd=float(X[s].std(0).mean()))
    return out


PROTOCOL_KEYS = ("sched_ms", "dt_ms", "image_ms", "response_start_ms", "readout_ms", "k_white",
                 "library", "contrast_hz", "n_library_blocks", "stim_seed", "order_seed")
TUPLE_KEYS = ("readout_ms", "blank_ms", "seizure_pad_s", "C_grid")


def protocol_from_runs(raw):
    """The settings saved in the runs (never the live configuration). All
    runs analysed together must share the stimulus protocol."""
    grids = [json.loads(str(np.load(p)["meta"]))["grid"] for p in raw]
    for k in PROTOCOL_KEYS:
        vals = {json.dumps(gr.get(k)) for gr in grids}
        if len(vals) > 1:
            raise SystemExit(f"runs differ in protocol setting {k}: {sorted(vals)}")
    g = dict(grids[0])
    for k in TUPLE_KEYS:
        g[k] = tuple(g[k])
    return g


def main(out_dir, n_jobs=None, fast=False):
    from joblib import Parallel, delayed
    import os
    import image_plots
    out_dir = Path(out_dir)
    raw = sorted((out_dir / "raw").glob("*.npz"))
    if not raw:
        raise SystemExit(f"no runs in {out_dir / 'raw'}")
    g = protocol_from_runs(raw)
    ana = out_dir / "analysis"
    (ana / "plots").mkdir(parents=True, exist_ok=True)
    models = ("local", "count") if fast else MODELS
    regimes = ("adapted",) if fast else ("adapted", "heldout", "null")
    n_jobs = n_jobs or max(1, min(len(raw), (os.cpu_count() or 4) - 2))
    print(f"analysing {len(raw)} run(s), {n_jobs} at a time; models {models}, regimes "
          f"{regimes} + frozen", flush=True)
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(analyse_run)(p, g, models, regimes) for p in raw)
    per_run = {p.stem: r for p, r in zip(raw, res)}
    seeds = sorted({r["meta"]["seed"] for r in res})
    by_seed = {s: {per_run[p.stem]["meta"]["condition"]: p for p in raw
                   if per_run[p.stem]["meta"]["seed"] == s} for s in seeds}
    frz = Parallel(n_jobs=min(n_jobs, len(seeds)), backend="loky")(
        delayed(analyse_frozen)(by_seed[s], g, models) for s in seeds)
    frozen = dict(zip(seeds, frz))
    for stem, o in per_run.items():
        # per-pixel probabilities only for the primary readout, adapted regime
        # (figures); every other combination keeps per-trial scores only
        np.savez_compressed(ana / f"{stem}_trials.npz", **{
            f"{T}_{kind}_{regime}_{k}": v for (T, kind, regime), r in o["res"].items()
            for k, v in r.items() if k != "settings"
            and (k != "S" or (T == g["primary_readout_ms"] and regime == "adapted"))},
            settings=json.dumps({f"{T}_{kind}_{regime}": r["settings"]
                                 for (T, kind, regime), r in o["res"].items()}),
            **{f"tt_{k}": v for k, v in o["tt"].items() if k != "Y"})
    rows = summarise(per_run, frozen, g)
    with open(ana / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    desc = descriptives(per_run, g)
    T0 = g["primary_readout_ms"]
    for r in sorted(rows, key=lambda r: (r["seed"], r["condition"])):
        if r["T_ms"] == T0 and r["phase"] == "interictal" and r["model"] == "local":
            print(f"seed{r['seed']} {r['condition']:16s} local {r['regime']:8s} BA {r['BA']:.3f} "
                  f"recall {r['recall']:.3f} spec {r['spec']:.3f} AUROC {r['AUROC']:.3f} SB {r['SB']:+.3f}",
                  flush=True)
    image_plots.make_all(per_run, frozen, rows, desc, g, ana / "plots")
    print(f"wrote {ana}/summary.csv, *_trials.npz and plots/", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else THIS_DIR / "results" / "image_random", fast="--fast" in sys.argv)
