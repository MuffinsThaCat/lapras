"""Systematic test: are approved COMBINATION partners linked in the Lapras network?

Replaces the hand-picked list in validate_combos.py (8 pairs, chosen by me) with every fixed-dose
combination product in Drugs@FDA (openFDA API): products with exactly two active ingredients.
Ingredients are matched to the drug library (salt words stripped); a pair is evaluable when both
drugs have a target on the 599-protein union network and share no target.

    python3 -m lapras_tools.validate_fdc --fetch     # download + cache (~60 API calls)
    python3 -m lapras_tools.validate_fdc             # analysis

For each evaluable pair, and for random drug pairs drawn from the SAME pool of drugs (so drug
class and target-count effects cancel), it asks: is some target of drug A directly linked to some
target of drug B?  do they share a module (<=40 groups)?  Reported with a binomial test, plus a
permutation null that reshuffles partners among the same drugs.

Limits: fixed-dose products are formulated together for pharmacological AND commercial reasons
(many are two drugs of one class, e.g. antihypertensives), and oncology combinations are
co-administered rather than co-formulated, so they are mostly absent; this tests "partners that
were developed together sit near each other", not "good combinations".
"""
import argparse, json, os, re, urllib.parse, urllib.request
import numpy as np
from .druglib import default_library

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "fda_fdc_products.json")
API = "https://api.fda.gov/drug/drugsfda.json"


def fetch():
    products, seen = [], set()
    for sort in ("application_number:asc", "application_number:desc"):     # skip is capped at 25,000
        for skip in range(0, 25001, 1000):
            q = urllib.parse.urlencode({"limit": 1000, "skip": skip, "sort": sort})
            try:
                d = json.load(urllib.request.urlopen(f"{API}?{q}", timeout=120))
            except Exception as e:
                print(f"  stop {sort} at skip {skip}: {e}")
                break
            for r in d["results"]:
                for p in r.get("products", []):
                    ing = sorted({i["name"].upper() for i in p.get("active_ingredients", [])})
                    key = (r["application_number"], p.get("product_number"))
                    if len(ing) == 2 and key not in seen:
                        seen.add(key)
                        products.append(dict(app=r["application_number"], brand=p.get("brand_name"), ing=ing,
                                             status=p.get("marketing_status")))
            print(f"  {sort} skip {skip}: {len(products)} two-ingredient products", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(products, open(CACHE, "w"))
    print(f"saved {len(products)} -> {CACHE}")


def normalise(name, names):
    """Map an ingredient string to a library drug name by stripping salt words."""
    toks = [t for t in re.split(r"[\s,/]+", name.lower()) if t]
    while toks:
        cand = " ".join(toks)
        if cand in names:
            return cand
        toks = toks[:-1]          # drop the trailing word (salts like 'hydrochloride', strengths) and retry
    return None


def pairs_from_cache(lib):
    prods = json.load(open(CACHE))
    names = set(lib)
    out = {}
    for p in prods:
        if p["status"] not in ("Prescription", "Over-the-counter", None):
            continue
        a, b = (normalise(x, names) for x in p["ing"])
        if a and b and a != b:
            out.setdefault(frozenset((a, b)), p["brand"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--modules", type=int, default=40)
    a = ap.parse_args()
    if a.fetch:
        fetch()
        return
    if not os.path.exists(CACHE):
        raise SystemExit("no cache; run with --fetch first")
    from .interactions import Context
    from scipy.stats import binom
    lib = default_library()
    ctx = Context(None, modules=a.modules)
    pairs = pairs_from_cache(lib)
    print(f"{len(pairs)} distinct two-drug FDC pairs whose ingredients map to library drugs")

    def on_net(d):
        return [ctx.idx[t] for t in d.targets if t in ctx.idx]

    def linked_and_module(d1, d2):
        t1, t2 = on_net(d1), on_net(d2)
        if not t1 or not t2 or set(t1) & set(t2):
            return None
        link = any(ctx.C[i, j] > 0 for i in t1 for j in t2)
        mod = any(ctx.lab[i] == ctx.lab[j] for i in t1 for j in t2)
        return link, mod

    ev = []
    for pr in pairs:
        x, y = sorted(pr)
        r = linked_and_module(lib[x], lib[y])
        if r is not None:
            ev.append((x, y, r))
    drugs = sorted({d for x, y, _ in ev for d in (x, y)})
    print(f"evaluable (both drugs target the network, no shared target): {len(ev)} pairs over {len(drugs)} drugs")
    if len(ev) < 10:
        raise SystemExit("too few evaluable pairs")
    lk = np.array([r[0] for _, _, r in ev]); md = np.array([r[1] for _, _, r in ev])

    # null 1: random pairs from the same drug pool
    rng = np.random.default_rng(0)
    null_l, null_m = [], []
    while len(null_l) < 20000:
        i, j = rng.choice(len(drugs), 2, replace=False)
        r = linked_and_module(lib[drugs[i]], lib[drugs[j]])
        if r is not None:
            null_l.append(r[0]); null_m.append(r[1])
    pl, pm = float(np.mean(null_l)), float(np.mean(null_m))
    for name, obs, p0 in (("directly linked", lk, pl), ("same module", md, pm)):
        k, n = int(obs.sum()), len(obs)
        print(f"{name:16s}: FDC pairs {k}/{n} = {k/n*100:.0f}%   random pairs from the same drugs {p0*100:.0f}%   "
              f"binomial P(>= {k}) = {binom.sf(k - 1, n, p0):.2g}")

    # null 2: reshuffle partners (keeps each drug's frequency in the FDC list)
    xs = [x for x, _, _ in ev]; ys = [y for _, y, _ in ev]
    obs_l = lk.mean(); cnt = 0; trials = 3000
    for _ in range(trials):
        perm = rng.permutation(len(ys))
        s = []
        for i in range(len(xs)):
            r = linked_and_module(lib[xs[i]], lib[ys[perm[i]]])
            if r is not None:
                s.append(r[0])
        if s and np.mean(s) >= obs_l:
            cnt += 1
    print(f"partner-shuffle permutation: P(linked fraction >= {obs_l:.2f}) = {(cnt + 1) / (trials + 1):.3f}")

    # PSEUDO-REPLICATION CHECK.  One drug can sit in many FDC pairs (ethinyl estradiol alone is paired
    # with eight progestins), so the pairs above are not independent.  Resample sets of pairs in which
    # every drug appears at most once, and compare each set with an equally sized random-pair null.
    from collections import Counter
    use = Counter(d for x, y, _ in ev for d in (x, y))
    print(f"\ndrugs appearing in many evaluable pairs: " + ", ".join(f"{d} ({n})" for d, n in use.most_common(5)))
    ind_l, ind_m, ind_n = [], [], []
    for _ in range(2000):
        order = rng.permutation(len(ev)); used = set(); sel = []
        for k in order:
            x, y, r = ev[k]
            if x not in used and y not in used:
                used |= {x, y}; sel.append(r)
        ind_n.append(len(sel)); ind_l.append(np.mean([r[0] for r in sel])); ind_m.append(np.mean([r[1] for r in sel]))
    print(f"independent-pair resamples (each drug once; ~{np.mean(ind_n):.0f} pairs per set): "
          f"directly linked {np.mean(ind_l)*100:.0f}% (5-95%: {np.percentile(ind_l,5)*100:.0f}-{np.percentile(ind_l,95)*100:.0f}%) "
          f"vs {pl*100:.0f}% random; same module {np.mean(ind_m)*100:.0f}% "
          f"({np.percentile(ind_m,5)*100:.0f}-{np.percentile(ind_m,95)*100:.0f}%) vs {pm*100:.0f}%")
    k_eq = int(round(np.mean(ind_l) * np.mean(ind_n)))
    print(f"   -> binomial at the mean set size: P(>= {k_eq} linked of {int(np.mean(ind_n))} | {pl*100:.0f}%) = "
          f"{binom.sf(k_eq - 1, int(np.mean(ind_n)), pl):.2g}")
    # sex-hormone products (receptor-ligand pairs, mostly contraceptives) excluded outright
    horm = {"ESR1", "PGR", "AR", "ESR2", "NR3C1"}
    ev_nh = [(x, y, r) for x, y, r in ev if not ((set(lib[x].targets) | set(lib[y].targets)) & horm)]
    k2 = sum(r[0] for _, _, r in ev_nh); n2 = len(ev_nh)
    pool = sorted({d for x, y, _ in ev_nh for d in (x, y)})
    nl2 = []
    while len(nl2) < 20000:
        i, j = rng.choice(len(pool), 2, replace=False)
        r = linked_and_module(lib[pool[i]], lib[pool[j]])
        if r is not None:
            nl2.append(r[0])
    p2 = float(np.mean(nl2))
    print(f"excluding products with a hormone-receptor drug (ESR1/PGR/AR/ESR2/NR3C1): {k2}/{n2} linked = {k2/max(n2,1)*100:.0f}% "
          f"vs {p2*100:.0f}% random (same pool); binomial P(>= {k2}) = {binom.sf(k2 - 1, n2, p2):.2g}")
    ex2 = [f"{x}+{y}" for x, y, r in ev_nh if r[0]][:12]
    print("   linked non-hormonal pairs: " + "; ".join(ex2))

    # what dominates?  same-class partners
    same_t = sum(1 for x, y in (tuple(sorted(p)) for p in pairs) if set(lib[x].targets) & set(lib[y].targets))
    print(f"(excluded as sharing a target, i.e. same drug class: {same_t} of {len(pairs)} pairs)")
    ex =[f"{x}+{y}" for x, y, r in ev if r[0]][:12]
    print("examples of linked FDC pairs: " + "; ".join(ex))


if __name__ == "__main__":
    main()
