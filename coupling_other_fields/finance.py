"""Tool 3 - Financial contagion table.

Bank interconnection network -> bill (systemic importance), D (modular vs global)
and a default-cascade check of whether the ranking means anything.

Input (CSV, header required):
    exposures.csv   lender,borrower,amount        (interbank loans)
    banks.csv       bank,assets,capital           (optional but recommended)

Coupling between banks i, j is the contagion strength: the larger of
L_ij/K_i and L_ji/K_j (loss from the other's default as a share of own capital,
capped at 1).  Without capital data it falls back to raw exposure.

    python3 -m coupling_other_fields.finance exposures.csv [banks.csv]
    python3 -m coupling_other_fields.finance --experiment          # synthetic validation

REAL bilateral interbank exposures are confidential (supervisory data); the
--experiment mode uses SYNTHETIC core-periphery networks, so it tests whether
the bill carries signal about cascades in that model class, not about the real
financial system.
"""
import argparse, csv
import numpy as np
from lapras_tools.core import analyze_network, spearman, LINE


# ---------------------------------------------------------------- model

def coupling_from_exposures(L, K=None):
    """L[i,j] = amount i lent to j.  K = capital vector."""
    if K is None:
        C = L + L.T
        return C / C.max() if C.max() > 0 else C
    A = L / np.maximum(K[:, None], 1e-12)        # loss to i if j defaults, share of i's capital
    return np.minimum(np.maximum(A, A.T), 1.0)


def cascade(L, K, first, lgd=1.0):
    """Default cascade started by the failure of bank `first`.
    Returns the boolean vector of defaulted banks."""
    n = len(K)
    dead = np.zeros(n, bool)
    dead[first] = True
    while True:
        loss = lgd * (L[:, dead].sum(1))
        new = (loss > K) & ~dead
        if not new.any():
            return dead
        dead |= new


def cascade_losses(L, K, lgd=1.0):
    """Per-bank: share of the system's capital destroyed (other banks failing) if it fails."""
    n = len(K)
    out = np.zeros(n)
    for i in range(n):
        d = cascade(L, K, i, lgd)
        d[i] = False
        out[i] = K[d].sum() / K.sum()
    return out


# ---------------------------------------------------------------- synthetic networks

def synthetic_banks(n=50, core_frac=0.2, p_core=0.8, p_periph=0.08, cap_ratio=0.06, seed=0):
    rng = np.random.default_rng(seed)
    assets = np.sort(rng.lognormal(0, 1.0, n))[::-1] * 100
    core = np.arange(n) < max(2, int(core_frac * n))
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = p_core if (core[i] and core[j]) else p_periph if not (core[i] or core[j]) else 0.35
            if rng.random() < p:
                L[i, j] = rng.lognormal(0, 0.7) * np.sqrt(assets[i] * assets[j]) * 0.05
    # cap interbank lending at 25% of each lender's assets
    row = L.sum(1)
    scale = np.minimum(1.0, 0.25 * assets / np.maximum(row, 1e-12))
    L = L * scale[:, None]
    K = cap_ratio * assets * rng.uniform(0.7, 1.3, n)
    return L, K, assets


def experiment(n_nets=60, n=50):
    print("Synthetic core-periphery interbank networks; cascade = defaults triggered by one bank failing.")
    print("Does the Lapras bill rank banks by cascade damage better than size / degree / strength?\n")
    rows, Ds, spread, mcs = [], [], [], []
    for s in range(n_nets):
        rng = np.random.default_rng(1000 + s)
        L, K, A = synthetic_banks(n=n, core_frac=rng.uniform(.1, .4), p_core=rng.uniform(.4, .95),
                                  p_periph=rng.uniform(.02, .2), cap_ratio=rng.uniform(.02, .10), seed=s)
        dmg = cascade_losses(L, K)
        if dmg.max() == 0:
            continue                         # nothing cascades; correlations undefined
        C = coupling_from_exposures(L, K)
        an = analyze_network(C, [f"B{i}" for i in range(n)])
        deg = ((L + L.T) > 0).sum(1)
        stre = C.sum(1)
        rows.append(dict(bill=spearman(an.bill, dmg), size=spearman(A, dmg), deg=spearman(deg, dmg),
                         strength=spearman(stre, dmg),
                         # top-5 overlap with the 5 most damaging banks
                         ov_bill=len(set(an.ranked()[:5]) & set(np.argsort(-dmg)[:5])),
                         ov_size=len(set(np.argsort(-A)[:5]) & set(np.argsort(-dmg)[:5])),
                         ov_str=len(set(np.argsort(-stre)[:5]) & set(np.argsort(-dmg)[:5]))))
        Ds.append(an.D); spread.append(float((dmg > 0.05).mean())); mcs.append(float(C[C > 0].mean()))
    print(f"{len(rows)} networks with non-trivial cascades (of {n_nets})")
    for k, lab in [("bill", "Lapras bill"), ("size", "assets"), ("deg", "degree"), ("strength", "coupling strength")]:
        v = np.array([r[k] for r in rows]); v = v[~np.isnan(v)]
        print(f"  Spearman vs cascade damage  {lab:18s} mean {v.mean():+.2f}  (n={len(v)})")
    for k, lab in [("size", "assets"), ("strength", "coupling strength"), ("deg", "degree")]:
        d = np.array([r["bill"] - r[k] for r in rows]); d = d[~np.isnan(d)]
        print(f"  paired difference (bill - {lab}): {d.mean():+.3f} +/- {d.std(ddof=1)/np.sqrt(len(d)):.3f} SE")
    for k, lab in [("ov_bill", "bill"), ("ov_size", "assets"), ("ov_str", "strength")]:
        print(f"  top-5 overlap with 5 most damaging banks  {lab:9s} {np.mean([r[k] for r in rows]):.2f} / 5")
    Ds, spread = np.array(Ds), np.array(spread)
    print(f"\nD vs contagion breadth (share of banks whose failure destroys >5% of system capital):")
    print(f"  Spearman(D, breadth) = {spearman(Ds, spread):+.2f}   D range {Ds.min():.2f}-{Ds.max():.2f}, "
          f"{(Ds > LINE).mean()*100:.0f}% of networks above the line")
    print(f"  baseline: Spearman(mean coupling, breadth) = {spearman(mcs, spread):+.2f}  "
          f"(a trivial predictor; D adds nothing if this is higher)")
    lo, hi = Ds <= np.median(Ds), Ds > np.median(Ds)
    print(f"  breadth when D <= median: {spread[lo].mean():.3f};  D > median: {spread[hi].mean():.3f}")


# ---------------------------------------------------------------- real-data path

def load_exposures(path, banks_path=None):
    names, rows = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((r["lender"], r["borrower"], float(r["amount"])))
            for b in (r["lender"], r["borrower"]):
                if b not in names:
                    names.append(b)
    idx = {b: i for i, b in enumerate(names)}
    L = np.zeros((len(names), len(names)))
    for a, b, x in rows:
        L[idx[a], idx[b]] += x
    K = A = None
    if banks_path:
        with open(banks_path) as f:
            d = {r["bank"]: r for r in csv.DictReader(f)}
        K = np.array([float(d[b]["capital"]) if b in d else np.nan for b in names])
        A = np.array([float(d[b]["assets"]) if b in d else np.nan for b in names])
        K = np.where(np.isnan(K), np.nanmedian(K), K)
    return names, L, K, A


def report(names, L, K=None):
    C = coupling_from_exposures(L, K)
    an = analyze_network(C, names)
    print(f"{len(names)} banks, D = {an.D:.2f} ({an.verdict}; "
          f"{'global: a failure can cascade across the system' if an.D > LINE else 'modular: failures stay inside modules'})")
    dmg = cascade_losses(L, K) if K is not None else None
    print(f"  {'Rank':>4} {'Bank':14} {'Bill':>7} {'Module':>6}" + ("  Cascade damage" if dmg is not None else ""))
    part = an.level_with_at_most(5)
    lab = an.label_vector(part)
    for r, i in enumerate(an.ranked()[:15]):
        extra = f"  {dmg[i]*100:6.1f}% of system capital" if dmg is not None else ""
        print(f"  {r+1:>4} {names[i]:14} {an.bill[i]:7.3f} {lab[i]:>6}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exposures", nargs="?")
    ap.add_argument("banks", nargs="?")
    ap.add_argument("--experiment", action="store_true")
    a = ap.parse_args()
    if a.experiment:
        experiment()
    elif a.exposures:
        names, L, K, _ = load_exposures(a.exposures, a.banks)
        report(names, L, K)
    else:
        ap.error("give exposures.csv or --experiment")


if __name__ == "__main__":
    main()
