"""Pure-Python analysis of the per-(seed, condition) results: no brian2
import, safe to run standalone against saved JSON without re-simulating
anything.

Two checks, matching what "check the orderings hold across seeds" means for
this pipeline:

1. Phase ordering (per condition): HSP_Phase.paired_phase_stats' contrasts
   (e.g. pre_ictal > mid > post_ictal in IPC) -- does the sign of each
   contrast agree across all network seeds?
2. Intervention ordering (per seed): ranking the conditions by ipc_ref (the
   same quantity HSP_Phase.intervention_comparison sorts on) -- is that
   ranking the same network seed to network seed?
"""
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_results(raw_dir):
    results = []
    for p in sorted(Path(raw_dir).glob("*.json")):
        with open(p) as fh:
            results.append(json.load(fh))
    return results


def phase_ordering_report(results):
    by_cond = defaultdict(dict)
    for r in results:
        if not r.get("ok"):
            continue
        by_cond[r["condition"]][r["seed"]] = r

    report = {}
    for cond, by_seed in by_cond.items():
        cond_report = {}
        for metric_label, stats_key in (("ipc_total", "ipc_stats"),
                                        ("participation_ratio", "pr_stats")):
            contrast_names = set()
            for r in by_seed.values():
                contrast_names |= set(r[stats_key]["contrasts"].keys())
            for cname in sorted(contrast_names):
                per_seed = {}
                for seed, r in by_seed.items():
                    c = r[stats_key]["contrasts"].get(cname)
                    if c is None:
                        continue
                    sign = 1 if c["mean_diff"] > 0 else (-1 if c["mean_diff"] < 0 else 0)
                    per_seed[seed] = dict(sign=sign, mean_diff=c["mean_diff"],
                                          p_wilcoxon=c.get("p_wilcoxon"))
                signs = {v["sign"] for v in per_seed.values()}
                consistent = len(per_seed) > 0 and len(signs - {0}) <= 1
                cond_report[f"{metric_label}:{cname}"] = dict(
                    per_seed=per_seed, consistent=consistent, n_seeds=len(per_seed))
        report[cond] = cond_report
    return report


def intervention_ordering_report(results, metric="ipc_ref"):
    by_seed = defaultdict(dict)
    for r in results:
        if not r.get("ok"):
            continue
        by_seed[r["seed"]][r["condition"]] = r.get(metric)

    rankings = {}
    for seed, cond_vals in by_seed.items():
        vals = {c: v for c, v in cond_vals.items() if v is not None and v == v}  # drop NaN
        rankings[seed] = sorted(vals, key=lambda c: -vals[c])

    seeds = sorted(rankings)
    identical = len({tuple(rankings[s]) for s in seeds}) <= 1 if seeds else False

    pairwise_tau = {}
    if len(seeds) >= 2:
        from scipy.stats import kendalltau
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                a, b = seeds[i], seeds[j]
                common = [c for c in rankings[a] if c in rankings[b]]
                rank_b = {c: k for k, c in enumerate(rankings[b])}
                xa = list(range(len(common)))
                xb = [rank_b[c] for c in common]
                if len(common) > 1:
                    tau, p = kendalltau(xa, xb)
                else:
                    tau, p = float("nan"), float("nan")
                pairwise_tau[f"seed{a}_vs_seed{b}"] = dict(tau=tau, p=p, n_common=len(common))

    return dict(rankings=rankings, identical_order=identical,
               pairwise_kendall_tau=pairwise_tau)


def print_report(results):
    phase_report = phase_ordering_report(results)
    rank_report = intervention_ordering_report(results)

    print("=" * 78)
    print("PHASE ORDERING (within condition, across network seeds)")
    print("=" * 78)
    for cond, contrasts in phase_report.items():
        print(f"\n[{cond}]")
        for cname, info in contrasts.items():
            flag = "OK  " if info["consistent"] else "FAIL"
            seed_bits = ", ".join(
                f"seed{s}: {'+' if v['sign'] > 0 else ('-' if v['sign'] < 0 else '0')}"
                f"{v['mean_diff']:+.3f} (p={v['p_wilcoxon']:.3f})" if v['p_wilcoxon'] == v['p_wilcoxon']
                else f"seed{s}: {v['mean_diff']:+.3f}"
                for s, v in info["per_seed"].items()
            )
            print(f"  [{flag}] {cname:28s} n_seeds={info['n_seeds']}  {seed_bits}")

    print("\n" + "=" * 78)
    print("INTERVENTION ORDERING (ipc_ref ranking, across network seeds)")
    print("=" * 78)
    for seed, ranking in rank_report["rankings"].items():
        print(f"  seed{seed}: " + " > ".join(ranking))
    print(f"\n  identical ranking across all seeds: {rank_report['identical_order']}")
    for pair, info in rank_report["pairwise_kendall_tau"].items():
        print(f"  {pair}: kendall tau={info['tau']:.3f} (p={info['p']:.3f}, n={info['n_common']})")

    return dict(phase_ordering=phase_report, intervention_ordering=rank_report)


def save_summary_csv(results, path):
    rows = []
    for r in results:
        if not r.get("ok"):
            rows.append(dict(seed=r.get("seed"), condition=r.get("condition"),
                            error=r.get("error")))
            continue
        row = dict(seed=r["seed"], condition=r["condition"],
                  is_bursting=r["cycle"]["is_bursting"],
                  period_s=r["cycle"]["period_s"], recovery_s=r["cycle"]["recovery_s"],
                  cycle_lf_power=r["cycle"]["cycle_lf_power"],
                  E_rate=r["cycle"]["E_rate"], I_rate=r["cycle"]["I_rate"],
                  ipc_ref=r["ipc_ref"], ipc_post_ictal=r["ipc_post_ictal"],
                  ipc_retained=r["ipc_retained"], tradeoff_score=r["tradeoff_score"])
        for ph, d in r["by_phase_raw"].items():
            ipc = d.get("ipc_total") or []
            pr = d.get("participation_ratio") or []
            if ipc:
                row[f"ipc_{ph}_mean"] = sum(ipc) / len(ipc)
            if pr:
                row[f"pr_{ph}_mean"] = sum(pr) / len(pr)
        rows.append(row)

    fieldnames = sorted({k for row in rows for k in row})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
