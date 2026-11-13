"""Combination-partner linkage on a DISEASE-AGNOSTIC network.

Every other combination test (validate_combos*, validate_fdc) measures linkage on the 599-protein
'union network', which is assembled from per-disease protein lists.  Drugs for one disease
therefore sit in one disease's neighbourhood by construction, and that circularity is only partly
removed by the indication-matched null (validate_combos_matched.py).  Here the network is STRING
itself: the human interactions (combined score >= threshold) among ALL target genes of the drugs
in the pair pool, fetched directly from the STRING API, with no disease lists involved.

    python3 -m lapras_tools.validate_combos_string [--score 900] [--fetch]

Reports, for approved combination pairs (label-mined + fixed-dose, same pair set as
validate_combos_matched): P(some target of drug A has a direct STRING link to some target of
drug B), against (a) every pool pair and (b) every pool pair sharing a specific phase-4
indication, exact (all pairs, no sampling), plus independent-pair resamples.
"""
import argparse, csv, io, os, urllib.parse, urllib.request
import numpy as np
from collections import Counter
from .druglib import default_library
from .pair_set import get_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
STRING_URL = "https://string-db.org/api/tsv/network"


def fetch_edges(genes, score, path):
    if os.path.exists(path):
        return path
    data = urllib.parse.urlencode({"identifiers": "\r".join(genes), "species": 9606,
                                   "required_score": score, "caller_identity": "lapras_tools"}).encode()
    txt = urllib.request.urlopen(urllib.request.Request(STRING_URL, data=data), timeout=300).read().decode()
    open(path, "w").write(txt)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=int, default=900, help="STRING combined score threshold x1000")
    ap.add_argument("--max-heading", type=int, default=60)
    ap.add_argument("--fetch", action="store_true", help="refetch even if cached")
    ap.add_argument("--sweep", action="store_true",
                    help="repeat over STRING score {400,700,900} x heading cutoff {20,60,200}; one line each")
    a = ap.parse_args()
    if a.sweep:
        import contextlib, io, re
        print(f"{'STRING':>7} {'max-head':>8} | {'approved':>9} {'all pairs':>9} {'matched':>8} | "
              f"{'P vs matched':>12} {'sharing: approved/matched':>26} {'P':>7} {'indep P':>8}")
        for sc in (400, 700, 900):
            for mh in (20, 60, 200):
                b = argparse.Namespace(score=sc, max_heading=mh, fetch=False, sweep=False)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    r = run(b)
                print(f"{sc/1000:7.2f} {mh:8d} | {r['k']/r['n']*100:8.1f}% {r['pa']*100:8.1f}% {r['pb']*100:7.1f}% | "
                      f"{r['p_matched']:12.2g} {r['ks']:>9d}/{r['ns']:<3d} {r['ks']/max(r['ns'],1)*100:5.1f}% vs {r['pb']*100:4.1f}% "
                      f"{r['p_share']:7.2g} {r['p_indep']:8.2g}")
        return
    run(a)


def run(a):
    from scipy.stats import binom
    lib = default_library()
    pairs = get_pairs(lib)
    pool = sorted({d for p in pairs for d in p})
    genes = sorted({t for d in pool for t in lib[d].targets if "_" not in t})
    path = os.path.join(HERE, "data", f"string_target_network_{a.score}.tsv")
    if a.fetch and os.path.exists(path):
        os.remove(path)
    fetch_edges(genes, a.score, path)
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    gi = {g: i for i, g in enumerate(genes)}
    A = np.zeros((len(genes), len(genes)), bool)
    for r in rows:
        u, v = r["preferredName_A"].upper(), r["preferredName_B"].upper()
        if u in gi and v in gi and u != v:
            A[gi[u], gi[v]] = A[gi[v], gi[u]] = True
    print(f"STRING (score >= {a.score/1000:.2f}): {len(genes)} target genes, {int(A.sum()//2)} interactions among them "
          f"({A.sum() / (len(genes) * (len(genes) - 1)) * 100:.1f}% density)")

    # drug x gene incidence (drugs with >= 1 gene in the query set)
    G = np.zeros((len(pool), len(genes)), bool)
    for i, d in enumerate(pool):
        for t in lib[d].targets:
            if t in gi:
                G[i, gi[t]] = True
    has = G.any(1)
    shared = (G.astype(int) @ G.T.astype(int)) > 0                    # any common target
    linked = (G.astype(int) @ A.astype(int) @ G.T.astype(int)) > 0    # some target of i linked to some target of j
    usable = has[:, None] & has[None, :] & ~shared
    np.fill_diagonal(usable, False)

    heads = {d: {h for h, p in lib[d].indications.items() if p >= 4} for d in pool}
    freq = Counter(h for d in pool for h in heads[d])
    spec = {h for h, c in freq.items() if c <= a.max_heading}
    share = np.array([[bool((heads[x] & spec) & (heads[y] & spec)) for y in pool] for x in pool])
    iu = np.triu_indices(len(pool), 1)
    ua = usable[iu]; ub = (usable & share)[iu]
    pa, pb = float(linked[iu][ua].mean()), float(linked[iu][ub].mean())
    idx = {d: i for i, d in enumerate(pool)}
    ev = [(x, y) for x, y in pairs if usable[idx[x], idx[y]]]
    k = sum(linked[idx[x], idx[y]] for x, y in ev); n = len(ev)
    print(f"{n} evaluable approved pairs over {len({d for p in ev for d in p})} drugs; "
          f"{len(pool)} drugs in the pool, {int(ua.sum())} usable pool pairs")
    print(f"directly linked: approved pairs {k}/{n} = {k/n*100:.1f}%;  all pool pairs {pa*100:.1f}%;  "
          f"indication-sharing pool pairs {pb*100:.1f}% ({int(ub.sum())} pairs)")
    print(f"  vs all pool pairs:          binomial P(>= {k}) = {binom.sf(k - 1, n, pa):.2g}")
    print(f"  vs indication-matched pairs: binomial P(>= {k}) = {binom.sf(k - 1, n, pb):.2g}")
    es = [(x, y) for x, y in ev if share[idx[x], idx[y]]]
    ks = sum(linked[idx[x], idx[y]] for x, y in es)
    print(f"  pairs sharing an indication: {ks}/{len(es)} = {ks/max(len(es),1)*100:.1f}% vs {pb*100:.1f}%; "
          f"P(>= {ks}) = {binom.sf(ks - 1, len(es), pb):.2g}")
    rng = np.random.default_rng(0)
    res = []
    for _ in range(2000):
        used, sel = set(), []
        for j in rng.permutation(len(es)):
            x, y = es[j]
            if x not in used and y not in used:
                used |= {x, y}; sel.append(linked[idx[x], idx[y]])
        res.append((np.mean(sel), len(sel)))
    ml, mn = np.mean([r for r, _ in res]), int(np.mean([m for _, m in res]))
    kk = int(round(ml * mn))
    p_indep = float(binom.sf(kk - 1, mn, pb))
    print(f"  independent resamples of indication-sharing pairs (~{mn} pairs, each drug once): {ml*100:.0f}% linked vs {pb*100:.1f}%; "
          f"P(>= {kk}) = {p_indep:.2g}")
    return dict(k=k, n=n, pa=pa, pb=pb, ks=int(ks), ns=len(es), p_matched=float(binom.sf(k - 1, n, pb)),
                p_share=float(binom.sf(ks - 1, len(es), pb)) if es else 1.0, p_indep=p_indep)


if __name__ == "__main__":
    main()
