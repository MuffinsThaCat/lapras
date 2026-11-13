"""Confound check for the combination-partner linkage result: same-disease drugs.

validate_fdc.py and validate_combos_large.py compare approved combination pairs with random pairs
from the same drug pool.  But combination partners are usually drugs for the SAME disease, and the
union network is built from per-disease protein sets, so drugs for one disease sit in one disease's
neighbourhood whether or not they are ever combined.  A fair null is random pairs of drugs that
SHARE AN INDICATION but are not known combination partners.

    python3 -m lapras_tools.validate_combos_matched

Uses the pair set from validate_combos_large.parse (+ FDC pairs) and ChEMBL phase-4 indication
headings from the library.  Headings held by more than --max-heading drugs (e.g. 'Neoplasms',
'Hypertension') are ignored when deciding 'shares an indication', because they hardly constrain
anything.
"""
import argparse
import numpy as np
from collections import Counter
from .druglib import default_library
from .pair_set import get_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-heading", type=int, default=60,
                    help="ignore indication headings shared by more than this many pool drugs")
    ap.add_argument("--modules", type=int, default=40)
    a = ap.parse_args()
    from scipy.stats import binom
    from .interactions import Context
    lib = default_library()
    pairs = {frozenset(p): None for p in get_pairs(lib)}
    ctx = Context(None, modules=a.modules)

    def tgt(d):
        return [ctx.idx[t] for t in lib[d].targets if t in ctx.idx]

    def best(d):                                   # highest-bill network target of a drug
        return max(tgt(d), key=lambda i: ctx.an.bill[i])

    def usable(x, y):
        tx, ty = tgt(x), tgt(y)
        return bool(tx and ty and not (set(tx) & set(ty)))

    ev = [tuple(sorted(p)) for p in pairs if usable(*sorted(p))]
    pool = sorted({d for p in ev for d in p})
    print(f"{len(ev)} evaluable approved combination pairs over {len(pool)} drugs")

    heads = {d: {h for h, p in lib[d].indications.items() if p >= 4} for d in pool}
    freq = Counter(h for d in pool for h in heads[d])
    specific = {h for h, c in freq.items() if c <= a.max_heading}
    sh = {d: heads[d] & specific for d in pool}

    def shares(x, y):
        return bool(sh[x] & sh[y])

    def linked(x, y):
        return ctx.C[best(x), best(y)] > 0

    comb_share = [p for p in ev if shares(*p)]
    comb_no = [p for p in ev if not shares(*p)]
    print(f"approved pairs that share a (specific) phase-4 indication: {len(comb_share)}/{len(ev)}")

    rng = np.random.default_rng(0)
    # Exact nulls over ALL pool pairs (no sampling):
    #   null A: every usable pool pair (what the earlier analyses sampled)
    #   null B: every usable pool pair that SHARES an indication
    best_i = {d: best(d) for d in pool}
    lk_m = np.array([[ctx.C[best_i[x], best_i[y]] > 0 for y in pool] for x in pool])
    us_m = np.array([[x != y and usable(x, y) for y in pool] for x in pool])
    sh_m = np.array([[x != y and shares(x, y) for y in pool] for x in pool])
    iu = np.triu_indices(len(pool), 1)
    ua, ub = us_m[iu], (us_m & sh_m)[iu]
    pa, pb = float(lk_m[iu][ua].mean()), float(lk_m[iu][ub].mean())
    print(f"\nall {int(ua.sum())} usable pool pairs: {pa*100:.1f}% directly linked;  the {int(ub.sum())} that SHARE an indication: "
          f"{pb*100:.1f}% (exact, not sampled)")
    print(f"=> same-disease alone raises the base rate {pb/pa:.1f}x")

    def report(name, subset, p0):
        k, n = sum(linked(*p) for p in subset), len(subset)
        if n:
            print(f"  {name:48s} {k:3d}/{n:3d} = {k/n*100:4.1f}% linked vs {p0*100:4.1f}% null; "
                  f"binomial P(>= {k}) = {binom.sf(k - 1, n, p0):.2g}")

    print("\nApproved combination pairs vs the matched null:")
    report("all evaluable pairs vs random pool pairs", ev, pa)
    report("all evaluable pairs vs indication-matched pairs", ev, pb)
    report("pairs that share an indication vs matched pairs", comb_share, pb)
    report("pairs that do NOT share an indication vs random", comb_no, pa)

    # independent-pair resamples (each drug once) against the matched null
    ind = []
    for _ in range(1000):
        order = rng.permutation(len(comb_share)); used = set(); sel = []
        for k in order:
            x, y = comb_share[k]
            if x not in used and y not in used:
                used |= {x, y}; sel.append(linked(x, y))
        ind.append((np.mean(sel) if sel else 0, len(sel)))
    ml, mn = np.mean([r for r, _ in ind]), int(np.mean([n for _, n in ind]))
    print(f"  independent resamples of indication-sharing pairs (~{mn} pairs, each drug once): {ml*100:.0f}% linked "
          f"vs {pb*100:.1f}% matched null; binomial P(>= {int(round(ml*mn))}) = {binom.sf(int(round(ml*mn)) - 1, mn, pb):.2g}")


if __name__ == "__main__":
    main()
