"""Tool 5 - Ecosystem collapse predictor.

Food web -> bill (keystone ranking), D (modular: losses stay contained;
global: losses cascade), and a secondary-extinction check of whether the bill
finds the species whose loss does the most damage.

Input: CSV with columns  consumer,resource  (one row per feeding link).

    python3 -m coupling_other_fields.ecosystem web.csv
    python3 -m coupling_other_fields.ecosystem web.csv --evaluate   # bill vs other rankings on ONE real web
    python3 -m coupling_other_fields.ecosystem --experiment     # SYNTHETIC niche-model webs

Coupling: C_ij = how much the pair depends on each other, the larger of
(1/#prey of i if i eats j) and (1/#prey of j if j eats i) - a consumer with one
prey is fully coupled to it, a generalist weakly.

Extinction model (topological): a consumer goes extinct when it loses >= theta
of its prey (default 0.5; theta=1.0 is the classic 'loses all prey' rule);
extinctions propagate to fixpoint.

No real food web was reachable when this was built, so --experiment uses
Williams & Martinez (2000) niche-model webs.  Run it on real data
(Web of Life, Mangal) before believing any conclusion about real ecosystems.
"""
import argparse, csv
import numpy as np
from lapras_tools.core import analyze_network, spearman, LINE


def niche_web(n=60, conn=0.1, seed=0):
    """Returns diet[i] = set of resources of species i."""
    rng = np.random.default_rng(seed)
    niche = np.sort(rng.random(n))
    beta = 1.0 / (2 * conn) - 1
    r = niche * rng.beta(1, beta, n)
    c = np.array([rng.uniform(r[i] / 2, max(niche[i], r[i] / 2 + 1e-9)) for i in range(n)])
    diet = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and abs(niche[j] - c[i]) <= r[i] / 2:
                diet[i].add(j)
    # niche model leaves the lowest species basal by construction (r ~ 0); keep as is
    return diet


def coupling_from_diet(diet):
    n = len(diet)
    C = np.zeros((n, n))
    for i, d in enumerate(diet):
        for j in d:
            w = 1.0 / len(d)
            C[i, j] = C[j, i] = max(C[i, j], w)
    return C


def extinction_cascade(diet, removed, theta=0.5):
    """Set of species lost (including `removed`) after propagating secondary extinctions."""
    n = len(diet)
    dead = set(removed)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if i in dead or not diet[i]:
                continue                      # basal species have no prey to lose
            lost = len(diet[i] & dead)
            if lost / len(diet[i]) >= theta:
                dead.add(i); changed = True
    return dead


def secondary_damage(diet, theta=0.5):
    return np.array([len(extinction_cascade(diet, [i], theta)) - 1 for i in range(len(diet))])


def robustness(diet, order, theta=0.5):
    """Fraction of species that must be removed (in `order`) before >= 50% of species are lost."""
    n = len(diet)
    for k in range(1, n + 1):
        if len(extinction_cascade(diet, list(order[:k]), theta)) >= n / 2:
            return k / n
    return 1.0


def experiment(n_webs=40, n=60, theta=0.5):
    print(f"SYNTHETIC niche-model webs, n={n}, connectance 0.05-0.20, extinction threshold theta={theta}\n")
    sp = {"bill": [], "degree": [], "consumers": []}
    rob = {"bill": [], "degree": [], "consumers": [], "random": []}
    Ds, meandmg = [], []
    rng = np.random.default_rng(1)
    used = 0
    for s in range(n_webs):
        diet = niche_web(n, rng.uniform(.05, .2), seed=s)
        dmg = secondary_damage(diet, theta)
        if dmg.max() == 0:
            continue
        used += 1
        C = coupling_from_diet(diet)
        an = analyze_network(C, [f"s{i}" for i in range(n)])
        deg = (C > 0).sum(1)
        ncons = np.array([sum(1 for d in diet if i in d) for i in range(n)])
        Ds.append(an.D); meandmg.append(dmg.mean())
        for k, v in (("bill", an.bill), ("degree", deg), ("consumers", ncons)):
            sp[k].append(spearman(v, dmg))
        rob["bill"].append(robustness(diet, an.ranked(), theta))
        rob["degree"].append(robustness(diet, np.argsort(-deg), theta))
        rob["consumers"].append(robustness(diet, np.argsort(-ncons), theta))
        rob["random"].append(np.mean([robustness(diet, rng.permutation(n), theta) for _ in range(5)]))
    print(f"{used}/{n_webs} webs had any secondary extinction")
    print("Spearman of ranking vs single-species secondary extinctions (higher = finds the keystones):")
    for k, v in sp.items():
        v = np.array(v); v = v[~np.isnan(v)]
        print(f"  {k:10s} {v.mean():+.2f}")
    print("Robustness: fraction of species removed (targeted order) before half the web is gone (LOWER = ranking is a better attack list):")
    for k, v in rob.items():
        print(f"  {k:10s} {np.mean(v):.3f}")
    print("Paired difference in robustness (bill - other; negative = bill is the better attack list):")
    for k in ("consumers", "degree", "random"):
        d = np.array(rob["bill"]) - np.array(rob[k])
        print(f"  bill - {k:10s} {d.mean():+.3f} +/- {d.std(ddof=1)/np.sqrt(len(d)):.3f} SE")
    d = np.array(sp["bill"]) - np.array(sp["consumers"])
    print(f"Paired difference in Spearman (bill - consumers): {d.mean():+.3f} +/- {d.std(ddof=1)/np.sqrt(len(d)):.3f} SE")
    Ds = np.array(Ds)
    print(f"\nD range {np.nanmin(Ds[np.isfinite(Ds)]):.2f}-{np.nanmax(Ds[np.isfinite(Ds)]):.2f}, "
          f"{(Ds > LINE).mean()*100:.0f}% of webs above the line ({np.isinf(Ds).sum()} infinite)")
    md = np.array(meandmg)
    print(f"Does D separate contained from cascading webs?  mean secondary extinctions per species loss: "
          f"D<=line {md[Ds <= LINE].mean():.2f} (n={(Ds <= LINE).sum()})  vs  D>line {md[Ds > LINE].mean():.2f} "
          f"(n={(Ds > LINE).sum()});  Spearman(D, damage) = {spearman(Ds, md):+.2f}")


def load_web(path):
    names, rows = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((r["consumer"], r["resource"]))
            for x in rows[-1]:
                if x not in names:
                    names.append(x)
    idx = {n: i for i, n in enumerate(names)}
    diet = [set() for _ in names]
    for c, r in rows:
        if c != r:
            diet[idx[c]].add(idx[r])
    return names, diet


def evaluate_web(names, diet, theta=0.5, n_random=300, seed=0):
    """Is the bill a good keystone ranking on ONE real web?  Compared with other rankings and with
    a random-order null (single web, so uncertainty comes from the random orders, not from many webs).

    Nodes with no prey at all are treated as EXTERNAL inputs (detritus, nutrients): they never go
    extinct and are not candidates for removal.  Without this, a web with a single prey-less root
    (Little Rock Lake in KONECT has exactly one) makes every cascade 'remove the root, everything
    dies', which says nothing about keystones."""
    n = len(names)
    ext = {i for i in range(n) if not diet[i]}
    keep = [i for i in range(n) if i not in ext]
    pos = {i: k for k, i in enumerate(keep)}
    d2 = [set(pos[j] for j in diet[i] if j in pos) for i in keep]
    nm = [names[i] for i in keep]
    m = len(keep)
    C = coupling_from_diet(d2)
    an = analyze_network(C, nm)
    dmg = secondary_damage(d2, theta)
    ncons = np.array([sum(1 for dd in d2 if i in dd) for i in range(m)])
    deg = (C > 0).sum(1)
    print(f"{n} nodes -> {m} species after treating {len(ext)} prey-less node(s) as external inputs; "
          f"{sum(1 for dd in d2 if not dd)} of them are now basal; D = {an.D:.2f} ({an.verdict})")
    print(f"single-species removal: mean secondary extinctions {dmg.mean():.2f}, max {dmg.max()}, "
          f"{(dmg > 0).mean()*100:.0f}% of species cause any")
    rng = np.random.default_rng(seed)
    rand = np.array([robustness(d2, rng.permutation(m), theta) for _ in range(n_random)])
    print(f"robustness (fraction removed before half the web is gone; LOWER = better attack list):")
    print(f"  random orders: mean {rand.mean():.3f}, sd {rand.std():.3f}")
    for name, order in (("bill", an.ranked()), ("degree", np.argsort(-deg, kind="stable")),
                        ("n consumers", np.argsort(-ncons, kind="stable")),
                        ("single-removal damage", np.argsort(-dmg, kind="stable"))):
        r = robustness(d2, order, theta)
        print(f"  {name:22s} {r:.3f}   ({(rand < r).mean()*100:.0f}% of random orders do better; z = {(r - rand.mean()) / rand.std():+.1f})")
    print(f"Spearman with single-removal damage: bill {spearman(an.bill, dmg):+.2f}, degree {spearman(deg, dmg):+.2f}, "
          f"n consumers {spearman(ncons, dmg):+.2f}")


def report(names, diet, theta=0.5, top=15):
    C = coupling_from_diet(diet)
    an = analyze_network(C, names)
    dmg = secondary_damage(diet, theta)
    print(f"{len(names)} species, {int(sum(len(d) for d in diet))} links, D = {an.D:.2f} -> {an.verdict}: "
          + ("loss cascades through the web." if an.D > LINE else "losses stay inside modules."))
    lab = an.label_vector(an.level_with_at_most(6))
    print(f"  {'Rank':>4} {'Species':20} {'Bill':>7} {'Module':>6}  Secondary extinctions if lost (theta={theta})")
    for r, i in enumerate(an.ranked()[:top]):
        print(f"  {r+1:>4} {names[i]:20} {an.bill[i]:7.3f} {lab[i]:>6}  {dmg[i]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("web", nargs="?")
    ap.add_argument("--experiment", action="store_true")
    ap.add_argument("--evaluate", action="store_true",
                    help="with web.csv: test the bill as a keystone ranking on this one real web")
    ap.add_argument("--theta", type=float, default=0.5)
    a = ap.parse_args()
    if a.evaluate and a.web:
        evaluate_web(*load_web(a.web), theta=a.theta)
    elif a.experiment:
        experiment(theta=a.theta)
    elif a.web:
        names, diet = load_web(a.web)
        report(names, diet, a.theta)
    else:
        ap.error("give web.csv or --experiment")


if __name__ == "__main__":
    main()
