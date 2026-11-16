import numpy as np
from hsp_metrics import bin_population, population_information
from hsp_capacity import make_input_signal, capacity_report


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _trace(mon, var):
    """Return a 1-D time series from a StateMonitor variable (mean over units
    if it recorded multiple)."""
    a = np.asarray(getattr(mon, var))
    if a.ndim == 2:
        return a.mean(axis=0)
    return a.ravel()


def _apply_drugs(net, drug):
    """Apply a drug spec dict to the network. Supports MIXTURES (na + benzo).
        drug keys (all optional):
          na_blocker : bool   + na_vthr_mV, na_refrac_ms
          benzo      : bool   + benzo_kGABA
          alpha      : input-weight scale (default 1.0)
    """
    from brian2 import mV, ms, second, pA, Hz
    P_E = net['P_E']; PG_E = net['PG_E']
    PG_E.w[:] = drug.get('alpha', 1.0) * float(PG_E.w[0])
    if drug.get('na_blocker', False):
        P_E.V_thr_E = drug.get('na_vthr_mV', -48.0) * mV
        P_E.tau_rp_E_var = drug.get('na_refrac_ms', 3.0) * ms
    if drug.get('benzo', False):
        P_E.k_GABA_E_hsp = drug.get('benzo_kGABA', 1.0)
        P_E.tau_ins_GABA_E = 1e99 * second
        P_E.tau_int_GABA_E = 1e99 * second
    if drug.get('dbs', False):
        net['dbs'].rates   = drug.get('dbs_freq_hz', 120.0) * Hz
        net['DBS_E'].w_dbs = drug.get('dbs_amp_pA', 30.0) * pA


def _freeze_hsp(net):
    from brian2 import second
    P_E = net['P_E']
    for nm in ["tau_ins_AMPA_E", "tau_int_AMPA_E",
               "tau_ins_GABA_E", "tau_int_GABA_E"]:
        setattr(P_E, nm, 1e99 * second)


def _runkw(net):
    ns = None
    for modname in ("hsp_capacity", "hsp_metrics"):
        try:
            mod = __import__(modname)
            ns = getattr(mod, "_MODEL_NS", None)
            if ns is None and hasattr(mod, "_build_model_namespace"):
                ns = mod._build_model_namespace()
            if ns:
                break
        except Exception:
            pass
    ns = ns or globals().get("_MODEL_NS", None)
    return dict(namespace=ns) if ns else {}


# ----------------------------------------------------------------------------
# 1. Characterize the homeostatic cycle (phase reference + E/I + power)
# ----------------------------------------------------------------------------
def _analyze_cycle(t, kA, kG, rE, rI, re, target_hz=2.0, burst_factor=4.0):
    """Pure analysis of recorded cycle traces (no Brian2). Testable."""
    thr = burst_factor * target_hz
    ictal = rE > thr
    edges = np.diff(ictal.astype(int))
    onsets = t[1:][edges == 1]
    offsets = t[1:][edges == -1]
    is_bursting = onsets.size > 0
    inter = ~ictal
    if inter.any():
        k_lo = float(np.percentile(kA[inter], 5))
        k_hi = float(np.percentile(kA[inter], 95))
    else:
        k_lo = k_hi = float(np.median(kA))
    period = float(np.median(np.diff(onsets))) if onsets.size >= 2 else np.nan
    recovery_s = np.nan
    if offsets.size and onsets.size:
        base = float(np.median(rE[inter])) if inter.any() else target_hz
        off = offsets[0]
        after = t >= off
        if after.any():
            rr = rE[after]; tt = t[after] - off
            below = np.where(rr <= 1.5 * base)[0]
            recovery_s = float(tt[below[0]]) if below.size else np.nan
    return dict(
        t=t, k_AMPA=kA, k_GABA=kG, rE=rE, rI=rI, r_est=re,
        is_bursting=is_bursting, burst_onsets=onsets, burst_offsets=offsets,
        k_lo=k_lo, k_hi=k_hi, period_s=period, recovery_s=recovery_s,
        cycle_lf_power=float(np.var(re)),
        E_rate=float(np.median(rE[inter])) if inter.any() else float(np.median(rE)),
        I_rate=float(np.median(rI[inter])) if inter.any() else float(np.median(rI)))


def characterize_cycle(net, input_freq_hz, drug=None, warmup_s=2000.0,
                       record_s=4000.0, target_hz=2.0, burst_factor=4.0):
    """
    Run at input_freq under `drug`, then read the monitors to characterize the
    cycle. Returns the k_AMPA interictal range (the phase reference), burst
    timing, recovery time, E/I activity, and the limit-cycle power.
    """
    from brian2 import second, Hz
    drug = drug or {}
    net.restore('baseline', filename='baseline_2hz.bri')
    inp = net['inp']; inp.rates = input_freq_hz * Hz
    _apply_drugs(net, drug)
    rk = _runkw(net)

    net.run(warmup_s * second, **rk)
    t_mark = float(net.t / second)
    net.run(record_s * second, **rk)

    kA_mon = net['k_AMPA_E_hsp']; kG_mon = net['k_GABA_E_hsp']
    rE_mon = net['rate_pop_mon_E']; rI_mon = net['rate_pop_mon_I']
    re_mon = net['r_est']

    t = np.asarray(kA_mon.t / second)
    m = t >= t_mark
    t = t[m] - t_mark
    kA = _trace(kA_mon, 'k_AMPA_E_hsp')[m]
    kG = _trace(kG_mon, 'k_GABA_E_hsp')[m]
    rE = _trace(rE_mon, 'r_pop_E')[m]
    rI = _trace(rI_mon, 'r_pop_I')[m]
    re = np.asarray(re_mon.r_est[0] / Hz)[m]

    out = _analyze_cycle(t, kA, kG, rE, rI, re, target_hz, burst_factor)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    out.update(dt=dt, input_freq_hz=input_freq_hz, drug=drug)
    return out


# ----------------------------------------------------------------------------
# 2. Snapshot at a target k_AMPA (= a cycle phase)
# ----------------------------------------------------------------------------
def snapshot_at_kphase(net, input_freq_hz, k_target, drug=None,
                       warmup_s=2000.0, chunk_s=50.0, max_scan_s=8000.0,
                       target_hz=2.0, burst_factor=4.0, store_name="phase_pt",
                       to_file=None):
    """
    Run forward and store a snapshot the first time mean k_AMPA crosses
    `k_target` during an interictal ramp (network not bursting). For a
    non-bursting condition k_AMPA is ~constant, so it stores once warmed up.
    """
    from brian2 import second, Hz
    drug = drug or {}
    net.restore('baseline', filename='baseline_2hz.bri')
    inp = net['inp']; inp.rates = input_freq_hz * Hz
    _apply_drugs(net, drug)
    rk = _runkw(net)
    kA_mon = net['k_AMPA_E_hsp']; rE_mon = net['rate_pop_mon_E']
    kG_mon = net['k_GABA_E_hsp']; rI_mon = net['rate_pop_mon_I']

    net.run(warmup_s * second, **rk)
    scanned = 0.0
    prev_k = float(_trace(kA_mon, 'k_AMPA_E_hsp')[-1])
    while scanned < max_scan_s:
        net.run(chunk_s * second, **rk)
        scanned += chunk_s
        kA = float(_trace(kA_mon, 'k_AMPA_E_hsp')[-1])
        rE = float(_trace(rE_mon, 'r_pop_E')[-1])
        interictal = rE < burst_factor * target_hz
        rising = kA >= prev_k
        crossed = (prev_k < k_target <= kA) or abs(kA - k_target) < 1e-3
        if interictal and rising and crossed:
            if to_file:
                net.store(store_name, filename=to_file)
            else:
                net.store(store_name)
            return dict(found=True, store_name=store_name,
                        t=float(net.t / second), k_AMPA=kA,
                        k_GABA=float(_trace(kG_mon, 'k_GABA_E_hsp')[-1]),
                        E_rate=rE, I_rate=float(_trace(rI_mon, 'r_pop_I')[-1]))
        prev_k = kA
    return dict(found=False, store_name=None,
                status="k_target not crossed in an interictal ramp")


# ----------------------------------------------------------------------------
# 3. Measure a snapshot: resting (multi-info, E/I, power) + driven (IPC)
# ----------------------------------------------------------------------------
def measure_state(net, store_name, u, r0_hz, amp_hz=8.0, step_s=0.05,
                  rest_s=60.0, bin_s=0.1, max_delay=40, to_file=None,
                  burst_rate_hz=50.0):
    from brian2 import second, Hz

    def _restore_and_freeze():
        if to_file:
            net.restore(store_name, filename=to_file)
        else:
            net.restore(store_name)
        _freeze_hsp(net)          # HSP off; fast state = the snapshot

    inp = net['inp']; P_E = net['P_E']
    sp_E = net['sp_exc']; sp_I = net['sp_inh']
    rE_mon = net['rate_pop_mon_E']
    rk = _runkw(net)
    n_E = int(P_E.N); n_I = int(net['P_I'].N)

    # ===== (b) DRIVEN PROBE — straight off the snapshot ===================
    _restore_and_freeze()
    t0p = float(net.t / second)
    rates = np.clip(r0_hz + amp_hz * np.asarray(u, float), 0.0, None)
    for k in range(len(u)):
        inp.rates = rates[k] * Hz
        net.run(step_s * second, **rk)
    siE = np.asarray(sp_E.i); stE = np.asarray(sp_E.t / second)
    edges = t0p + np.arange(len(u) + 1) * step_s
    win = (stE >= edges[0]) & (stE < edges[-1])
    Xp = np.zeros((len(u), n_E))
    for nrn in range(n_E):
        ts = stE[win][siE[win] == nrn]
        if ts.size:
            Xp[:, nrn], _ = np.histogram(ts, bins=edges)
    step_rate = Xp.sum(axis=1) / (n_E * step_s)
    burst = bool(np.any(step_rate > burst_rate_hz))
    rep = capacity_report(Xp, u, max_delay=max_delay)

    # ===== (a) RESTING WINDOW — also straight off the snapshot ============
    _restore_and_freeze()
    inp.rates = r0_hz * Hz
    t0 = float(net.t / second)
    net.run(rest_s * second, **rk)
    t1 = float(net.t / second)
    siE = np.asarray(sp_E.i); stE = np.asarray(sp_E.t / second)
    siI = np.asarray(sp_I.i); stI = np.asarray(sp_I.t / second)
    wE = (stE >= t0) & (stE < t1); wI = (stI >= t0) & (stI < t1)
    X, _ = bin_population(siE[wE], stE[wE], n_E, t0, t1, bin_s=bin_s)
    info = population_information(X, tse_samples=10) if X.shape[0] > 3 else \
        dict(multi_information=np.nan)
    E_rate = float(wE.sum() / (n_E * rest_s))
    I_rate = float(wI.sum() / (n_I * rest_s))
    rest = _trace(rE_mon, 'r_pop_E')
    rest_t = np.asarray(rE_mon.t / second)
    rmask = rest_t >= t0
    rest_power = float(np.var(rest[rmask])) if rmask.any() else np.nan

    return dict(
        multi_information=float(info.get("multi_information", np.nan)),
        linear_mc=rep["linear_mc"], nonlinear=rep["nonlinear"],
        ipc_total=rep["ipc_total"], state_rank=rep["state_rank"],
        E_rate=E_rate, I_rate=I_rate,
        EI_ratio=float(E_rate / I_rate) if I_rate > 0 else np.nan,
        rest_power=rest_power, burst_during_probe=burst,
        forget_curve=rep["forget_curve"])

# ----------------------------------------------------------------------------
# 4. Phase-resolved capacity across the cycle
# ----------------------------------------------------------------------------
_PHASE_FRAC = {"post_ictal": 0.10, "mid": 0.50, "pre_ictal": 0.90}


def phase_resolved_capacity(net, input_freq_hz, drug=None,
                            phases=("post_ictal", "mid", "pre_ictal"),
                            n_steps=2000, amp_hz=8.0, step_s=0.05,
                            signal_seed=0, char_kw=None, measure_kw=None):
    """
    Characterize the cycle, then snapshot+measure at each phase. For a
    non-bursting condition the phases collapse to one 'steady' point.
    Returns dict(cycle, rows) where rows is a list of per-phase records with
    IPC, multi-information, E/I (activity + k's), power, and stability.
    """
    drug = drug or {}
    char_kw = char_kw or {}; measure_kw = measure_kw or {}
    cyc = characterize_cycle(net, input_freq_hz, drug=drug, **char_kw)
    u = make_input_signal(n_steps, seed=signal_seed)

    if not cyc["is_bursting"]:
        ph_list = [("steady", 0.5)]
    else:
        ph_list = [(p, _PHASE_FRAC[p]) for p in phases]

    rows = []
    for name, frac in ph_list:
        k_target = cyc["k_lo"] + frac * (cyc["k_hi"] - cyc["k_lo"])
        snap = snapshot_at_kphase(net, input_freq_hz, k_target, drug=drug,
                                  store_name=f"phase_{name}")
        if not snap["found"]:
            rows.append(dict(phase=name, found=False, status=snap.get("status")))
            continue
        rec = measure_state(net, snap["store_name"], u, r0_hz=input_freq_hz,
                            amp_hz=amp_hz, step_s=step_s, **measure_kw)
        rec.update(phase=name, found=True, k_AMPA=snap["k_AMPA"],
                   k_GABA=snap["k_GABA"], k_target=k_target)
        rows.append(rec)
    return dict(cycle=cyc, rows=rows, input_freq_hz=input_freq_hz, drug=drug)


def _ipc_probe(net, store_name, u, r0_hz, amp_hz=8.0, step_s=0.05,
               max_delay=40, to_file=None, burst_rate_hz=50.0, rng_seed=None):
    """Restore a snapshot, freeze HSP, run ONLY the driven probe -> IPC.
    Cheap; safe to call many times on the same snapshot. rng_seed makes the
    Poisson realisation independent per seed (seeded AFTER restore so it isn't
    overwritten by whatever RNG state restore carries)."""
    from brian2 import second, Hz, seed as b2seed
    if to_file:
        net.restore(store_name, filename=to_file)
    else:
        net.restore(store_name)
    _freeze_hsp(net)
    if rng_seed is not None:
        b2seed(int(rng_seed))
    inp = net['inp']; P_E = net['P_E']; sp_E = net['sp_exc']
    rk = _runkw(net); n_E = int(P_E.N)
    t0p = float(net.t / second)
    rates = np.clip(r0_hz + amp_hz * np.asarray(u, float), 0.0, None)
    for k in range(len(u)):
        inp.rates = rates[k] * Hz
        net.run(step_s * second, **rk)
    siE = np.asarray(sp_E.i); stE = np.asarray(sp_E.t / second)
    edges = t0p + np.arange(len(u) + 1) * step_s
    win = (stE >= edges[0]) & (stE < edges[-1])
    Xp = np.zeros((len(u), n_E))
    for nrn in range(n_E):
        ts = stE[win][siE[win] == nrn]
        if ts.size:
            Xp[:, nrn], _ = np.histogram(ts, bins=edges)
    step_rate = Xp.sum(axis=1) / (n_E * step_s)
    burst = bool(np.any(step_rate > burst_rate_hz))
    rep = capacity_report(Xp, u, max_delay=max_delay)
    E_rate_probe = float(Xp.sum() / (n_E * len(u) * step_s))   # Hz, during the probe
    return dict(ipc_total=rep["ipc_total"], linear_mc=rep["linear_mc"],
                nonlinear=rep["nonlinear"], state_rank=rep["state_rank"],
                participation_ratio=rep["participation_ratio"],
                E_rate_probe=E_rate_probe,
                burst_during_probe=burst)


def phase_capacity_seeded(net, input_freq_hz, seeds, drug=None,
                          phases=("post_ictal", "mid", "pre_ictal"),
                          n_steps=2000, amp_hz=8.0, step_s=0.05, max_delay=40,
                          probe_r0=None, char_kw=None):
    """Characterise the cycle + snapshot each phase ONCE, then probe every
    snapshot with every input seed (paired across phases). Non-bursting ->
    single 'steady' snapshot. probe_r0 overrides the probe operating point
    (default = input_freq_hz)."""
    drug = drug or {}; char_kw = char_kw or {}
    r0 = input_freq_hz if probe_r0 is None else probe_r0
    cyc = characterize_cycle(net, input_freq_hz, drug=drug, **char_kw)
    ph_list = ([("steady", 0.5)] if not cyc["is_bursting"]
               else [(p, _PHASE_FRAC[p]) for p in phases])
    by_phase = {}
    for name, frac in ph_list:
        k_target = cyc["k_lo"] + frac * (cyc["k_hi"] - cyc["k_lo"])
        snap = snapshot_at_kphase(net, input_freq_hz, k_target, drug=drug,
                                  store_name=f"phase_{name}")
        if not snap["found"]:
            by_phase[name] = dict(found=False, status=snap.get("status"))
            continue
        acc = {k: [] for k in ("ipc_total", "linear_mc", "nonlinear", "E_rate_probe",  
                               "state_rank","participation_ratio", "burst_during_probe")}
        for s in seeds:
            u = make_input_signal(n_steps, seed=int(s))
            r = _ipc_probe(net, snap["store_name"], u, r0_hz=r0, amp_hz=amp_hz,
                           step_s=step_s, max_delay=max_delay, rng_seed=int(s))
            for k in acc:
                acc[k].append(r[k])
        acc = {k: np.array(v) for k, v in acc.items()}
        acc.update(found=True, seeds=list(seeds),
                   k_AMPA=snap["k_AMPA"], k_GABA=snap["k_GABA"])
        by_phase[name] = acc
    return dict(cycle=cyc, by_phase=by_phase, input_freq_hz=input_freq_hz,
                drug=drug)


def paired_phase_stats(by_phase, metric="ipc_total", ci=0.95,
                       contrasts=(("mid", "post_ictal"),
                                  ("pre_ictal", "mid"),
                                  ("pre_ictal", "steady"))):
    """Per-phase mean/SD/SEM/t-CI, and paired contrasts (matched on seed index).
    Positive mean_diff = first phase > second (the direction you predict)."""
    from scipy import stats
    def summarize(x):
        x = np.asarray(x, float); n = len(x)
        m = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        tcrit = stats.t.ppf(0.5 + ci / 2, n - 1) if n > 1 else np.nan
        return dict(n=n, mean=m, sd=sd, sem=sem,
                    lo=m - tcrit * sem, hi=m + tcrit * sem)
    per_phase = {ph: summarize(v[metric]) for ph, v in by_phase.items()
                 if v.get("found")}
    out = {}
    for a, b in contrasts:
        if a not in by_phase or b not in by_phase: continue
        if not (by_phase[a].get("found") and by_phase[b].get("found")): continue
        xa = np.asarray(by_phase[a][metric], float)
        xb = np.asarray(by_phase[b][metric], float)
        d = xa - xb; s = summarize(d)
        try: p_t = float(stats.ttest_rel(xa, xb).pvalue)
        except Exception: p_t = np.nan
        try: p_w = float(stats.wilcoxon(xa, xb).pvalue) if len(d) >= 6 else np.nan
        except Exception: p_w = np.nan
        out[f"{a}-{b}"] = dict(mean_diff=s["mean"], sd=s["sd"], sem=s["sem"],
                               lo=s["lo"], hi=s["hi"], n=s["n"],
                               p_ttest=p_t, p_wilcoxon=p_w)
    return dict(per_phase=per_phase, contrasts=out)


def run_phase_replication(net, seeds, burst_freq_hz=36.0, steady_freq_hz=40.0,
                          drug=None, metric="ipc_total", **kw):
    """Bursting + steady conditions with the SAME seeds -> paired across all
    four states. Returns raw arrays + per-phase and contrast stats."""
    burst = phase_capacity_seeded(net, burst_freq_hz, seeds, drug=drug, **kw)
    steady = phase_capacity_seeded(net, steady_freq_hz, seeds, drug=drug, **kw)
    by_phase = {}
    by_phase.update({k: v for k, v in burst["by_phase"].items() if v.get("found")})
    by_phase.update({k: v for k, v in steady["by_phase"].items() if v.get("found")})
    return dict(burst=burst, steady=steady, by_phase=by_phase,
                stats=paired_phase_stats(by_phase, metric=metric))

# ----------------------------------------------------------------------------
# 5. Intervention comparison (incl. mixtures) + trade-off ranking
# ----------------------------------------------------------------------------
def intervention_comparison(net, interventions, input_freq_hz,
                            phases=("post_ictal", "mid", "pre_ictal"),
                            **prc_kw):
    """
    interventions : list of dicts, each
        {'name': 'na+benzo', 'drug': {'na_blocker':True, 'na_vthr_mV':-48,
                                       'benzo':True, 'benzo_kGABA':0.7}}
    Mixtures are just a drug dict with several keys set.

    Returns a per-intervention summary with the seizure/capacity trade-off
    quantities you optimize on:
        seizure side : is_bursting, period_s, recovery_s, cycle_lf_power
        capacity side: ipc at 'mid' (or 'steady'), and ipc retained post-ictal
        balance      : E_rate, I_rate, EI_ratio, multi_information
    plus a simple trade-off score (higher = better: more capacity, less seizure).
    """
    summary = []
    for iv in interventions:
        name = iv["name"]; drug = iv.get("drug", {})
        prc = phase_resolved_capacity(net, input_freq_hz, drug=drug,
                                      phases=phases, **prc_kw)
        cyc = prc["cycle"]
        rows = {r["phase"]: r for r in prc["rows"] if r.get("found")}
        ref = rows.get("mid") or rows.get("steady")
        post = rows.get("post_ictal")
        ipc_ref = ref["ipc_total"] if ref else np.nan
        ipc_post = post["ipc_total"] if post else ipc_ref
        retained = (ipc_post / ipc_ref) if (ref and ipc_ref > 0) else np.nan
        # trade-off score: reward capacity, penalize seizure power
        score = (ipc_ref if np.isfinite(ipc_ref) else 0.0) \
            / (1.0 + cyc["cycle_lf_power"])
        summary.append(dict(
            name=name, is_bursting=cyc["is_bursting"],
            period_s=cyc["period_s"], recovery_s=cyc["recovery_s"],
            cycle_lf_power=cyc["cycle_lf_power"],
            ipc_ref=ipc_ref, ipc_post_ictal=ipc_post, ipc_retained=retained,
            multi_information=(ref["multi_information"] if ref else np.nan),
            E_rate=(ref["E_rate"] if ref else cyc["E_rate"]),
            I_rate=(ref["I_rate"] if ref else cyc["I_rate"]),
            EI_ratio=(ref["EI_ratio"] if ref else np.nan),
            tradeoff_score=score, detail=prc))
    summary.sort(key=lambda d: (-d["tradeoff_score"]))
    return summary