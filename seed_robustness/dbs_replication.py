"""Do the DBS-vs-benzo dissociation and the high-dose DBS IPC decline
replicate across network seeds?

    python dbs_replication.py [results_dir]      -> results/dbs_replication.txt

Per network seed, IPC of a condition = mean over probe seeds of the 36 Hz
network's own IPC ('mid' if still bursting, own 'steady' if suppressed; see
plot_ipc_summary.condition_ipc_values). Only DBS doses that actually
suppressed bursting in THAT seed are compared to benzo -- a dose that still
seizes isn't a competing treatment.

  A. dissociation: benzo vs untreated-40Hz reference; suppressing DBS vs benzo
     and vs reference
  B. high-dose decline: Spearman rho(amplitude, IPC) over suppressing doses
  C. (post hoc) decline over DBS >= HIGH_DOSE_PA only: the curve turned out to
     be an inverted U, which B's whole-range rho averages away. The 60 pA
     cutoff was chosen after seeing the data -- weaker evidence than A/B.

Probes during which the network seized are excluded (plot_ipc_summary.probe_ipc).

Replication = the sign agreeing across seeds; the sign-test p is two-sided
binomial(k, n, 0.5). With 5 seeds the smallest attainable p is 0.0625.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import orderings  # noqa: E402
from plot_ipc_summary import condition_ipc_values  # noqa: E402

BENZO = "benzo (k=1.0)"
HIGH_DOSE_PA = 60.0


def per_seed_table(results):
    """{seed: {'ref': float, 'benzo': (ipc, bursting), 'dbs': {amp: (ipc, bursting)}}}"""
    table = {}
    for r in results:
        if not r.get("ok"):
            continue
        s = table.setdefault(r["seed"], {"ref": np.nan, "benzo": None, "dbs": {}})
        ref = r.get("by_phase_raw", {}).get("reference")
        if ref:
            s["ref"] = float(np.mean(ref["ipc_total"]))
        vals = condition_ipc_values(r)
        ipc = float(np.mean(vals)) if vals is not None and len(vals) else np.nan
        entry = (ipc, bool(r["cycle"]["is_bursting"]))
        if r["condition"] == BENZO:
            s["benzo"] = entry
        elif (r.get("drug") or {}).get("dbs"):
            s["dbs"][float(r["drug"]["dbs_amp_pA"])] = entry
    return table


def sign_test(k, n):
    return float(stats.binomtest(k, n, 0.5).pvalue) if n else np.nan


def analyse(results):
    table = per_seed_table(results)
    lines = []
    say = lines.append
    seeds = sorted(table)
    agree = {"benzo>=ref": [], "dbs<benzo": [], "dbs<ref": [], "decline": [],
             "high_decline": []}

    say(f"{len(seeds)} network seed(s); IPC = mean over probe seeds, 36 Hz network\n")
    for seed in seeds:
        s = table[seed]
        say(f"seed {seed}: reference (untreated 40 Hz) {s['ref']:.4f}")
        if s["benzo"]:
            b, b_burst = s["benzo"]
            say(f"  benzo {b:.4f}{'  (still bursting)' if b_burst else ''}  "
                f"benzo - ref {b - s['ref']:+.4f}")
            agree["benzo>=ref"].append(b >= s["ref"])
        else:
            b = np.nan
            say("  benzo: missing")
        amps = sorted(s["dbs"])
        row = "  DBS  " + "  ".join(
            f"{a:g}:{s['dbs'][a][0]:.3f}{'*' if s['dbs'][a][1] else ''}" for a in amps)
        say(row + "   (* = still bursting)")
        supp = [a for a in amps if not s["dbs"][a][1] and np.isfinite(s["dbs"][a][0])]
        if not supp:
            say("  no DBS dose suppressed bursting")
            continue
        ipc_supp = np.array([s["dbs"][a][0] for a in supp])
        say(f"  suppressing doses {supp[0]:g}-{supp[-1]:g} pA ({len(supp)}): "
            f"mean {ipc_supp.mean():.4f}")
        if np.isfinite(b):
            below = int(np.sum(ipc_supp < b))
            say(f"    below benzo: {below}/{len(supp)} doses, mean DBS - benzo "
                f"{ipc_supp.mean() - b:+.4f}")
            agree["dbs<benzo"].append(ipc_supp.mean() < b)
        below_ref = int(np.sum(ipc_supp < s["ref"]))
        say(f"    below reference: {below_ref}/{len(supp)} doses, mean DBS - ref "
            f"{ipc_supp.mean() - s['ref']:+.4f}")
        agree["dbs<ref"].append(ipc_supp.mean() < s["ref"])
        if len(supp) >= 3:
            rho, p = stats.spearmanr(supp, ipc_supp)
            say(f"    dose trend over suppressing doses: Spearman rho {rho:+.2f} "
                f"(p {p:.3f}, n {len(supp)}); IPC {supp[0]:g} pA -> {supp[-1]:g} pA: "
                f"{ipc_supp[0]:.4f} -> {ipc_supp[-1]:.4f}")
            agree["decline"].append(rho < 0)
        high = [a for a in supp if a >= HIGH_DOSE_PA]
        if len(high) >= 3:
            ipc_high = [s["dbs"][a][0] for a in high]
            rho, p = stats.spearmanr(high, ipc_high)
            say(f"    (post hoc) {high[0]:g}-{high[-1]:g} pA only: rho {rho:+.2f}; IPC "
                f"{ipc_high[0]:.4f} -> {ipc_high[-1]:.4f}")
            agree["high_decline"].append(rho < 0)
        say("")

    say("replication across seeds (sign agreement; two-sided sign test):")
    names = {"benzo>=ref": "A1 benzo IPC >= reference",
             "dbs<benzo": "A2 suppressing DBS IPC < benzo",
             "dbs<ref": "A3 suppressing DBS IPC < reference",
             "decline": "B  IPC declines with DBS amplitude (rho < 0)",
             "high_decline": f"C  (post hoc) declines over >= {HIGH_DOSE_PA:g} pA"}
    for key, label in names.items():
        v = agree[key]
        k, n = int(np.sum(v)), len(v)
        say(f"  {label:46s} {k}/{n} seeds  (p {sign_test(k, n):.3f})")
    return "\n".join(lines)


if __name__ == "__main__":
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else THIS_DIR / "results")
    text = analyse(orderings.load_results(results_dir / "raw"))
    print(text)
    (results_dir / "dbs_replication.txt").write_text(text + "\n")
