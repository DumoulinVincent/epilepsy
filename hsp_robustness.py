"""
hsp_robustness.py
=================
Turns a single-seed sweep into a defensible result:

  1. seed_average_sweep / aggregate_conditions
        run each (condition, frequency) over many seeds -> mean +/- SEM,
        answering "is the drug separation signal or RNG noise?"

  2. rate_control
        break the firing-rate confound in multi-information: regress capacity
        on interictal rate (pooled), and test whether condition still explains
        the residual. If the drug effect survives rate control, the metric is
        carrying information beyond "how active is the network".

Depends only on hsp_metrics.run_sweep_with_metrics (your existing, validated
machinery) + numpy/matplotlib. The pure-data functions (aggregation,
residualisation, OLS) are unit-tested on synthetic sweep dicts and do not need
Brian2.
"""

import os
import numpy as np

# Metrics we carry through the whole pipeline. Anything in a sweep-result dict
# that is a 1-D array aligned to input_freqs_hz can be aggregated.
_METRIC_KEYS = ["multi_information", "lzc", "count_entropy_bits", "mean_rate_hz",
                "tse_complexity", "H_joint", "interictal_fraction",
                "ictal_fraction", "n_bursts", "burst_rate_hz", "lf_power",
                "mean_overshoot"]


# ----------------------------------------------------------------------------
# 1. Seed averaging
# ----------------------------------------------------------------------------
def seed_average_sweep(net, input_freqs_hz, condition="normal",
                       seeds=(101, 202, 303, 404, 505), checkpoint_dir=None,
                       **sweep_kw):
    """
    Run run_sweep_with_metrics once per seed and stack the results.

    net           : your built Brian2 network (with 'baseline' stored).
    seeds         : iterable of integer seeds -> independent noise realisations.
    checkpoint_dir: if given, each (condition, seed) result is saved there as
                    .npy so a long run can resume; existing files are loaded
                    instead of recomputed.
    sweep_kw      : forwarded to run_sweep_with_metrics (run_time_s, bin_s,
                    target_hz, factor, tse_samples, alpha, and drug kwargs such
                    as na_vthr_mV / benzo_kGABA).

    Returns
    -------
    dict with:
        freqs      : (n_freqs,)
        seeds      : list
        condition  : str
        raw        : {metric: (n_seeds, n_freqs)}
        mean, sem  : {metric: (n_freqs,)}
    """
    from hsp_metrics import run_sweep_with_metrics

    freqs = np.asarray(input_freqs_hz, dtype=float)
    seeds = list(seeds)
    raw = {k: np.full((len(seeds), len(freqs)), np.nan) for k in _METRIC_KEYS}

    for s_idx, s in enumerate(seeds):
        ckpt = None
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt = os.path.join(checkpoint_dir, f"{condition}_seed{s}.npy")
        if ckpt and os.path.exists(ckpt):
            res = np.load(ckpt, allow_pickle=True).item()
        else:
            res = run_sweep_with_metrics(net, freqs, condition=condition,
                                         seed_val=s, **sweep_kw)
            res.pop("traces", None)          # drop heavy per-freq traces
            if ckpt:
                np.save(ckpt, res, allow_pickle=True)
        for k in _METRIC_KEYS:
            if k in res:
                raw[k][s_idx, :] = res[k]

    n = max(len(seeds), 1)
    mean = {k: np.nanmean(v, axis=0) for k, v in raw.items()}
    sem = {k: np.nanstd(v, axis=0, ddof=1) / np.sqrt(n) if n > 1
           else np.full(len(freqs), np.nan) for k, v in raw.items()}
    return dict(freqs=freqs, seeds=seeds, condition=condition,
                raw=raw, mean=mean, sem=sem)


def aggregate_conditions(net, input_freqs_hz,
                         conditions=("normal", "nablocker", "benzo"),
                         seeds=(101, 202, 303, 404, 505), checkpoint_dir=None,
                         **sweep_kw):
    """Run seed_average_sweep for several conditions. Returns {condition: agg}."""
    return {c: seed_average_sweep(net, input_freqs_hz, condition=c, seeds=seeds,
                                  checkpoint_dir=checkpoint_dir, **sweep_kw)
            for c in conditions}


# ----------------------------------------------------------------------------
# 2. Long-form table (one row per condition x seed x frequency)
# ----------------------------------------------------------------------------
def to_long(agg_by_cond, metrics=("multi_information", "mean_rate_hz", "lzc",
                                  "ictal_fraction")):
    """
    Flatten {condition: agg} into aligned 1-D arrays for regression/plotting.
    Returns dict of arrays: condition (str), freq, seed, + one per metric.
    """
    cols = {"condition": [], "freq": [], "seed": []}
    for m in metrics:
        cols[m] = []
    for cond, agg in agg_by_cond.items():
        freqs, seeds = agg["freqs"], agg["seeds"]
        for si, s in enumerate(seeds):
            for fi, f in enumerate(freqs):
                cols["condition"].append(cond)
                cols["freq"].append(f)
                cols["seed"].append(s)
                for m in metrics:
                    cols[m].append(agg["raw"][m][si, fi])
    out = {k: (np.array(v) if k == "condition" else np.asarray(v, float))
           for k, v in cols.items()}
    return out


# ----------------------------------------------------------------------------
# 3. Rate-control: does capacity survive controlling for interictal rate?
# ----------------------------------------------------------------------------
def _ols(X, y):
    """Plain OLS. Returns beta, yhat, residuals, R^2 (ignores NaN rows)."""
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xo, yo = X[ok], y[ok]
    beta, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    ss_res = np.nansum((yo - Xo @ beta) ** 2)
    ss_tot = np.nansum((yo - yo.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, yhat, resid, r2


def rate_control(agg_by_cond, capacity="multi_information",
                 rate="mean_rate_hz", rate_degree=2):
    """
    Two complementary controls for the firing-rate confound:

    (A) Residualisation. Fit  capacity ~ poly(rate, rate_degree)  pooled across
        ALL points (every condition/seed/freq), then look at the residual
        (capacity beyond what rate alone predicts). If conditions still
        separate in the residual, capacity carries drug information beyond
        activity level.

    (B) Nested OLS / ANCOVA-style variance test. Compare
            model_rate  : capacity ~ poly(rate)
            model_full  : capacity ~ poly(rate) + condition dummies
        The increase in R^2 (and the partial F) quantifies how much condition
        explains *after* rate is accounted for.

    Returns a dict with the long table augmented by 'residual', the two model
    R^2 values, the partial-F statistic, and per-(condition,freq) residual
    mean/SEM ready to plot.

    CAVEAT: points are not independent (frequencies within a seed are a
    trajectory; seeds are independent). Treat the F/R^2 as descriptive effect
    sizes, not a clean significance test. The honest claim is "condition
    explains X% of capacity variance beyond rate".
    """
    long = to_long(agg_by_cond, metrics=(capacity, rate))
    y = long[capacity]
    r = long[rate]
    conds = long["condition"]
    uconds = sorted(set(conds.tolist()))

    # design: intercept + rate polynomial
    poly = np.vstack([r ** d for d in range(rate_degree + 1)]).T  # (n, deg+1)
    beta_r, yhat_r, resid_r, r2_rate = _ols(poly, y)

    # full model: + condition dummies (drop first as reference)
    dummies = np.vstack([(conds == c).astype(float) for c in uconds[1:]]).T \
        if len(uconds) > 1 else np.zeros((len(y), 0))
    Xfull = np.hstack([poly, dummies]) if dummies.size else poly
    beta_f, yhat_f, resid_f, r2_full = _ols(Xfull, y)

    # partial F for the condition block
    ok = np.isfinite(y) & np.isfinite(r)
    n = ok.sum()
    p_r, p_f = poly.shape[1], Xfull.shape[1]
    ss_r = np.nansum(resid_r[ok] ** 2)
    ss_f = np.nansum(resid_f[ok] ** 2)
    df1 = max(p_f - p_r, 1)
    df2 = max(n - p_f, 1)
    partial_F = ((ss_r - ss_f) / df1) / (ss_f / df2) if ss_f > 0 else np.nan

    # re-aggregate residual-from-rate by (condition, freq)
    resid = resid_r  # residual after removing rate trend
    res_by_cond = {}
    for cond, agg in agg_by_cond.items():
        freqs = agg["freqs"]
        m = np.full(len(freqs), np.nan)
        e = np.full(len(freqs), np.nan)
        for fi, f in enumerate(freqs):
            sel = (conds == cond) & (long["freq"] == f)
            vals = resid[sel]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                m[fi] = vals.mean()
                e[fi] = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else np.nan
        res_by_cond[cond] = dict(freqs=freqs, resid_mean=m, resid_sem=e)

    return dict(long=long, residual=resid, capacity=capacity, rate=rate,
                r2_rate=r2_rate, r2_full=r2_full, r2_gain=r2_full - r2_rate,
                partial_F=partial_F, df1=df1, df2=df2,
                rate_fit_beta=beta_r, residual_by_condition=res_by_cond,
                conditions=uconds)


def rate_matched_comparison(agg_by_cond, capacity="multi_information",
                            rate="mean_rate_hz", n_bins=5):
    """
    Non-parametric complement to rate_control: bin all points by interictal
    rate into n_bins quantile bins, and within each rate bin report the mean
    capacity per condition. Comparing conditions *within a rate bin* removes
    the rate confound by construction. Returns bin edges + per-bin/per-cond
    means (NaN where a condition has no points in a bin).
    """
    long = to_long(agg_by_cond, metrics=(capacity, rate))
    r, y, conds = long[rate], long[capacity], long["condition"]
    ok = np.isfinite(r) & np.isfinite(y)
    edges = np.quantile(r[ok], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    uconds = sorted(set(conds.tolist()))
    table = np.full((n_bins, len(uconds)), np.nan)
    counts = np.zeros((n_bins, len(uconds)), int)
    bin_idx = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        for ci, c in enumerate(uconds):
            sel = ok & (bin_idx == b) & (conds == c)
            if sel.any():
                table[b, ci] = np.nanmean(y[sel])
                counts[b, ci] = sel.sum()
    centres = 0.5 * (edges[:-1] + edges[1:])
    return dict(edges=edges, centres=centres, conditions=uconds,
                table=table, counts=counts, capacity=capacity, rate=rate)


# ----------------------------------------------------------------------------
# 4. Plotting
# ----------------------------------------------------------------------------
_COL = {"normal": "tab:blue", "nablocker": "tab:green", "benzo": "tab:red"}


def plot_seed_averaged(agg_by_cond, metric="multi_information", ax=None,
                       ylabel=None):
    """Mean +/- SEM vs input frequency, one line per condition."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for cond, agg in agg_by_cond.items():
        f = agg["freqs"]
        m, e = agg["mean"][metric], agg["sem"][metric]
        ax.errorbar(f, m, yerr=e, fmt="-o", ms=4, capsize=3,
                    color=_COL.get(cond), label=cond)
    ax.set_xlabel("Input frequency (Hz)")
    ax.set_ylabel(ylabel or metric)
    ax.grid(alpha=.3)
    ax.legend()
    return ax


def plot_rate_control(rc, agg_by_cond):
    """
    Three panels:
      (1) capacity vs interictal rate (all points) + pooled rate fit
      (2) residual capacity (rate removed) vs frequency, mean +/- SEM
      (3) text summary of R^2 gain and partial F
    """
    import matplotlib.pyplot as plt
    long = rc["long"]
    cap, rate = rc["capacity"], rc["rate"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5),
                           gridspec_kw={"width_ratios": [1, 1, 0.7]})

    # (1) scatter + fit
    for c in rc["conditions"]:
        sel = long["condition"] == c
        ax[0].scatter(long[rate][sel], long[cap][sel], s=18, alpha=.6,
                      color=_COL.get(c), label=c)
    rgrid = np.linspace(np.nanmin(long[rate]), np.nanmax(long[rate]), 100)
    deg = len(rc["rate_fit_beta"]) - 1
    poly = np.vstack([rgrid ** d for d in range(deg + 1)]).T
    ax[0].plot(rgrid, poly @ rc["rate_fit_beta"], "k--", lw=2,
               label="pooled rate fit")
    ax[0].set_xlabel("Interictal mean rate (Hz)")
    ax[0].set_ylabel(cap)
    ax[0].set_title("Capacity vs rate (confound)")
    ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=8)

    # (2) residual vs frequency
    for cond, d in rc["residual_by_condition"].items():
        ax[1].errorbar(d["freqs"], d["resid_mean"], yerr=d["resid_sem"],
                       fmt="-o", ms=4, capsize=3, color=_COL.get(cond),
                       label=cond)
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xlabel("Input frequency (Hz)")
    ax[1].set_ylabel(f"{cap} residual (rate removed)")
    ax[1].set_title("Capacity beyond rate")
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8)

    # (3) summary text
    ax[2].axis("off")
    txt = (f"Rate-only model R^2 = {rc['r2_rate']:.3f}\n"
           f"Rate + condition R^2 = {rc['r2_full']:.3f}\n"
           f"R^2 gain from condition = {rc['r2_gain']:.3f}\n\n"
           f"partial F({rc['df1']}, {rc['df2']}) = {rc['partial_F']:.2f}\n\n"
           "Interpretation:\n"
           "large gain / F  -> drug effect on capacity\n"
           "survives the rate confound.\n"
           "~0 gain -> capacity was mostly rate.\n\n"
           "(descriptive effect size; points are\n"
           "not fully independent -- see docstring)")
    ax[2].text(0, 1, txt, va="top", ha="left", fontsize=10, family="monospace")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 5. Seizure-edge detection and edge-aligned capacity
# ----------------------------------------------------------------------------
def find_seizure_edge(freqs, burden, frac=0.5):
    """
    Upper edge of the cyclic-seizure window: the input frequency on the
    HIGH side of the burden hump where burden falls to `frac` of its peak.
    Linearly interpolated. Returns (edge_hz, status):
        status 'ok'        : a clean descending crossing was found
        status 'no_seizure': burden never rises -> edge undefined (NaN)
        status 'open'      : burden never drops below threshold within the
                             swept range -> window not closed; edge = last freq
    """
    f = np.asarray(freqs, float)
    b = np.asarray(burden, float)
    if np.nanmax(b) < 1e-6:
        return np.nan, "no_seizure"
    pk = int(np.nanargmax(b))
    thr = frac * b[pk]
    desc = b[pk:]
    below = np.where(desc < thr)[0]
    if below.size == 0:
        return f[-1], "open"
    j = pk + int(below[0])               # first point below thr on descent
    f0, f1, b0, b1 = f[j - 1], f[j], b[j - 1], b[j]
    edge = f0 + (thr - b0) * (f1 - f0) / (b1 - b0) if b1 != b0 else f1
    return float(edge), "ok"
 
 
def edge_align(agg_by_cond, capacity="multi_information",
               burden="ictal_fraction", frac=0.5):
    """
    Re-express capacity vs DISTANCE FROM THE UPPER SEIZURE EDGE instead of
    absolute input frequency, so the conditions can be compared on equal
    footing. Then test whether they overlap once aligned:
 
        model_aligned : capacity ~ poly(rel_freq)
        model_full    : capacity ~ poly(rel_freq) + condition
    A small R^2 gain after alignment is the positive result: the drugs
    TRANSLATE the capacity curve (they move the edge) rather than changing its
    height. Returns per-condition aligned curves, the fitted edges, and the
    overlap statistics.
    """
    edges, aligned = {}, {}
    long_rel, long_cap, long_cond = [], [], []
    for cond, agg in agg_by_cond.items():
        f = agg["freqs"]
        edge, status = find_seizure_edge(f, agg["mean"][burden], frac=frac)
        edges[cond] = dict(edge=edge, status=status)
        rel = f - edge
        aligned[cond] = dict(rel_freq=rel, edge=edge, status=status,
                             cap_mean=agg["mean"][capacity],
                             cap_sem=agg["sem"][capacity])
        # pool per-seed points for the overlap regression
        raw = agg["raw"][capacity]
        for si in range(raw.shape[0]):
            long_rel.extend(rel.tolist())
            long_cap.extend(raw[si].tolist())
            long_cond.extend([cond] * len(rel))
 
    long_rel = np.asarray(long_rel); long_cap = np.asarray(long_cap)
    long_cond = np.asarray(long_cond)
    uconds = sorted(set(long_cond.tolist()))
 
    poly = np.vstack([long_rel ** d for d in range(3)]).T      # up to quadratic
    _, _, _, r2_aligned = _ols(poly, long_cap)
    dummies = np.vstack([(long_cond == c).astype(float) for c in uconds[1:]]).T \
        if len(uconds) > 1 else np.zeros((len(long_cap), 0))
    Xfull = np.hstack([poly, dummies]) if dummies.size else poly
    _, _, _, r2_full = _ols(Xfull, long_cap)
 
    return dict(edges=edges, aligned=aligned, capacity=capacity,
                r2_aligned=r2_aligned, r2_full=r2_full,
                r2_gain=r2_full - r2_aligned, conditions=uconds)
 
 
# ----------------------------------------------------------------------------
# 6. Nested model ladder: offset vs frequency-interaction
# ----------------------------------------------------------------------------
def capacity_model_ladder(agg_by_cond, capacity="multi_information",
                          rate="mean_rate_hz", freq_degree=2):
    """
    Fit a ladder of nested OLS models and report how much each added block
    raises R^2. The key contrast is M1 (condition as an OFFSET) vs M3
    (condition x frequency INTERACTION): your drug effect is a sideways shift,
    which only the interaction block can represent.
 
        M0: capacity ~ poly(rate)
        M1: + condition (offset)            <- what the old rate_control used
        M2: + poly(freq)
        M3: + condition x poly(freq)        <- can encode a frequency shift
 
    Returns R^2 of each model and the partial F for each added block.
    """
    long = to_long(agg_by_cond, metrics=(capacity, rate))
    # need frequency too
    fr = []
    for cond, agg in agg_by_cond.items():
        for _ in agg["seeds"]:
            fr.extend(agg["freqs"].tolist())
    freq = np.asarray(fr, float)
    y = long[capacity]; r = long[rate]; conds = long["condition"]
    uconds = sorted(set(conds.tolist()))
    n = np.isfinite(y).sum()
 
    ratep = np.vstack([r ** d for d in range(3)]).T
    freqp = np.vstack([freq ** d for d in range(1, freq_degree + 1)]).T
    dums = np.vstack([(conds == c).astype(float) for c in uconds[1:]]).T \
        if len(uconds) > 1 else np.zeros((len(y), 0))
    inter = np.hstack([dums * freqp[:, [k]] for k in range(freqp.shape[1])]) \
        if dums.size else np.zeros((len(y), 0))
 
    designs = {
        "M0_rate":            ratep,
        "M1_+offset":         np.hstack([ratep, dums]) if dums.size else ratep,
        "M2_+freq":           np.hstack([ratep, dums, freqp]) if dums.size else np.hstack([ratep, freqp]),
        "M3_+interaction":    np.hstack([ratep, dums, freqp, inter]) if inter.size else np.hstack([ratep, freqp]),
    }
    out = {}
    prev_ss = prev_p = None
    order = ["M0_rate", "M1_+offset", "M2_+freq", "M3_+interaction"]
    for name in order:
        X = designs[name]
        _, _, resid, r2 = _ols(X, y)
        ss = np.nansum(resid[np.isfinite(y)] ** 2)
        p = X.shape[1]
        F = np.nan
        if prev_ss is not None and p > prev_p and ss > 0:
            df1 = p - prev_p; df2 = max(int(n) - p, 1)
            F = ((prev_ss - ss) / df1) / (ss / df2)
        out[name] = dict(r2=r2, n_params=p, partial_F_vs_prev=F)
        prev_ss, prev_p = ss, p
    return out
 
 
# ----------------------------------------------------------------------------
# 7. Edge-aligned plot
# ----------------------------------------------------------------------------
def plot_edge_aligned(ea):
    """Capacity vs distance-from-seizure-edge, conditions overlaid."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for cond, d in ea["aligned"].items():
        lbl = f"{cond} (edge={d['edge']:.1f} Hz, {d['status']})"
        ax.errorbar(d["rel_freq"], d["cap_mean"], yerr=d["cap_sem"],
                    fmt="-o", ms=4, capsize=3, color=_COL.get(cond), label=lbl)
    ax.axvline(0, color="k", lw=.8, ls=":")
    ax.set_xlabel("Input frequency relative to upper seizure edge (Hz)")
    ax.set_ylabel(ea["capacity"])
    ax.set_title(f"Edge-aligned capacity  (R^2 gain from condition = "
                 f"{ea['r2_gain']:.3f})")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
