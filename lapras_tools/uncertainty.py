"""Uncertainty on D and on the bill ranking.

STRING interactions are noisy: edges are missing or spurious and scores are estimates.
This perturbs the network - drop a fraction of edges at random and multiply surviving
weights by lognormal noise - recomputes D and the bill ranking each time, and reports:

    D median and 5-95% interval, P(D > 3.03)
    verdict: MODULAR / GLOBAL if >=90% of perturbations agree, else UNCERTAIN
    top-10 stability: mean overlap of the perturbed top-10 bill list with the original

    python3 -m lapras_tools.uncertainty               # all 31 diseases, one row each
    python3 -m lapras_tools.uncertainty breast_cancer # one disease, with per-protein rank stability

The perturbation (default: 10% of edges dropped, 10% weight noise) is a modelling choice,
not a fitted error model; it measures sensitivity to plausible data error, not a
confidence interval in the statistical sense.
"""
import argparse
import numpy as np
from .core import analyze_network, load_disease, disease_keys, LINE


def perturb(C, rng, drop=0.1, sigma=0.1):
    iu = np.triu_indices(C.shape[0], 1)
    vals = C[iu]
    keep = (vals > 0) & (rng.random(len(vals)) >= drop)
    new = np.where(keep, vals * rng.lognormal(0, sigma, len(vals)), 0.0)
    new = np.minimum(new, 1.0)
    P = np.zeros_like(C)
    P[iu] = new
    return P + P.T


def d_uncertainty(C, names, n=100, drop=0.1, sigma=0.1, seed=0, top=10):
    base = analyze_network(C, names)
    rng = np.random.default_rng(seed)
    base_top = set(base.ranked()[:top].tolist())
    Ds, overlaps = [], []
    rank_sum = np.zeros(len(names))
    for _ in range(n):
        a = analyze_network(perturb(base.C, rng, drop, sigma), names)
        Ds.append(a.D)
        overlaps.append(len(base_top & set(a.ranked()[:top].tolist())) / top)
        r = np.empty(len(names)); r[a.ranked()] = np.arange(len(names)); rank_sum += r
    Ds = np.array(Ds)
    p_global = float((Ds > LINE).mean())
    verdict = "GLOBAL" if p_global >= 0.9 else "MODULAR" if p_global <= 0.1 else "UNCERTAIN"
    fin = Ds[np.isfinite(Ds)]
    return dict(D=base.D, D_med=float(np.median(Ds)),
                lo=float(np.percentile(fin, 5)) if len(fin) else float("nan"),
                hi=float(np.percentile(fin, 95)) if len(fin) else float("nan"),
                p_global=p_global, verdict=verdict, top_overlap=float(np.mean(overlaps)),
                mean_rank=rank_sum / n, base=base)


SWEEP = [(0.05, 0.05), (0.10, 0.10), (0.20, 0.10), (0.10, 0.20), (0.0, 0.10), (0.10, 0.0)]


def sweep(n=60, score=0.9):
    """Is each verdict an artefact of the perturbation settings?  Recompute P(global) and the
    verdict under several (edge-drop, weight-noise) settings and two seeds each.
    Returns {key: (label, D, [(drop, sigma, seed, p_global, verdict)...])}."""
    out = {}
    for k in disease_keys():
        C, names, _, label = load_disease(k, score)
        runs = []
        for drop, sigma in SWEEP:
            for seed in (0, 1):
                u = d_uncertainty(C, names, n=n, drop=drop, sigma=sigma, seed=seed)
                runs.append((drop, sigma, seed, u["p_global"], u["verdict"]))
        out[k] = (label, u["D"], runs)
    return out


def print_sweep(res):
    print(f"{'Disease':34} {'D':>5}  P(global) per setting [drop/noise]: " +
          " ".join(f"{int(d*100)}/{int(s*100)}" for d, s in SWEEP) + "   verdicts agree?")
    stable = {"MODULAR": 0, "GLOBAL": 0}
    unstable = []
    for k, (label, D, runs) in sorted(res.items(), key=lambda kv: kv[1][1]):
        per = []
        for i in range(0, len(runs), 2):
            per.append(np.mean([runs[i][3], runs[i + 1][3]]))
        verdicts = {r[4] for r in runs}
        base_v = "GLOBAL" if D > LINE else "MODULAR"
        agree = verdicts == {base_v}
        if agree:
            stable[base_v] += 1
        else:
            unstable.append(label)
        print(f"{label[:34]:34} {D:5.2f}  " + " ".join(f"{p:5.2f}" for p in per) +
              f"   {'yes (' + base_v + ')' if agree else 'NO: ' + '/'.join(sorted(verdicts))}")
    print(f"\nverdict identical to the unperturbed one under EVERY setting and seed: "
          f"{stable['MODULAR']} modular, {stable['GLOBAL']} global; unstable: {len(unstable)} -> {', '.join(unstable)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="vary the perturbation settings and seeds; report which verdicts survive all")
    ap.add_argument("--n-sweep", type=int, default=60, help="perturbations per setting in --sweep")
    ap.add_argument("disease", nargs="?")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--score", type=float, default=0.9)
    a = ap.parse_args()
    if a.sweep:
        print_sweep(sweep(a.n_sweep, a.score))
        return
    keys = [a.disease] if a.disease else disease_keys()
    rows = []
    for k in keys:
        C, names, _, label = load_disease(k, a.score)
        u = d_uncertainty(C, names, a.n, a.drop, a.sigma)
        rows.append((u["D"], label, u))
    rows.sort(key=lambda r: r[0])
    print(f"{'Disease':40s} {'D':>5} {'median':>6} {'5-95%':>13} {'P(global)':>9}  {'Verdict':9} top10-stab")
    for D, label, u in rows:
        print(f"{label:40s} {D:5.2f} {u['D_med']:6.2f} [{u['lo']:5.2f},{u['hi']:5.2f}] {u['p_global']:9.2f}  "
              f"{u['verdict']:9} {u['top_overlap']:.2f}")
    if len(rows) > 1:
        v = [u["verdict"] for _, _, u in rows]
        print(f"\nrobust MODULAR: {v.count('MODULAR')}, robust GLOBAL: {v.count('GLOBAL')}, "
              f"UNCERTAIN (straddles {LINE}): {v.count('UNCERTAIN')}")
    else:
        u = rows[0][2]; base = u["base"]
        order = np.argsort(u["mean_rank"])[:15]
        print("\ntop proteins by mean perturbed rank (original rank in brackets):")
        orig = {base.names[i]: r for r, i in enumerate(base.ranked())}
        for i in order:
            print(f"  {base.names[i]:10s} mean rank {u['mean_rank'][i]+1:5.1f}  [{orig[base.names[i]]+1}]")


if __name__ == "__main__":
    main()
