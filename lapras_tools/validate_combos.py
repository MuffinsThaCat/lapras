"""Do approved drug combinations hit DIFFERENT Lapras modules?

The Combination Designer's premise is "one drug per module": hit separate parts of the coupling
network.  Test: for approved targeted combinations in diseases we have networks for, are the two
drugs' targets in different modules more often than two random approved drugs of that disease?

Pairs below are approved (FDA) targeted-therapy combinations written from general knowledge of
the approvals; they are NOT exhaustive and not independently verified here - edit PAIRS to test
others.  Pairs where both drugs share a target, or a target is not on the disease network, are
reported as not evaluable.

    python3 -m lapras_tools.validate_combos [--modules 5]
"""
import argparse
import numpy as np
from .core import load_disease, analyze_network
from .druglib import default_library

PAIRS = [  # (disease key, drug A, drug B, note)
    ("melanoma", "dabrafenib", "trametinib", "BRAF+MEK, BRAF V600"),
    ("melanoma", "vemurafenib", "cobimetinib", "BRAF+MEK"),
    ("melanoma", "encorafenib", "binimetinib", "BRAF+MEK"),
    ("melanoma", "nivolumab", "ipilimumab", "PD-1 + CTLA-4"),
    ("breast_cancer", "alpelisib", "fulvestrant", "PI3K + ER (PIK3CA-mutant HR+)"),
    ("breast_cancer", "capivasertib", "fulvestrant", "AKT + ER"),
    ("breast_cancer", "palbociclib", "fulvestrant", "CDK4/6 + ER"),
    ("breast_cancer", "abemaciclib", "fulvestrant", "CDK4/6 + ER"),
    ("breast_cancer", "ribociclib", "letrozole", "CDK4/6 + aromatase"),
    ("breast_cancer", "tucatinib", "trastuzumab", "HER2 + HER2 (same target)"),
    ("colorectal_cancer", "encorafenib", "cetuximab", "BRAF + EGFR, BRAF V600E"),
    ("colorectal_cancer", "sotorasib", "panitumumab", "KRAS G12C + EGFR"),
    ("ovarian_cancer", "olaparib", "bevacizumab", "PARP + VEGF"),
    ("liver_cancer", "atezolizumab", "bevacizumab", "PD-L1 + VEGF"),
    ("lung_cancer", "dabrafenib", "trametinib", "BRAF+MEK, BRAF V600E NSCLC"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", type=int, default=5)
    a = ap.parse_args()
    lib = default_library()
    rng = np.random.default_rng(0)
    cache, ev_diff, ev_n, seen = {}, [], 0, set()
    print(f"{'Disease':20} {'Combination':30} {'targets':>22}  modules (<= {a.modules} groups)   verdict")
    for key, x, y, note in PAIRS:
        if x not in lib or y not in lib:
            print(f"{key:20} {x}+{y:20} not in library"); continue
        if key not in cache:
            C, names, _, _ = load_disease(key)
            an = analyze_network(C, names)
            lab = an.label_vector(an.level_with_at_most(a.modules))
            cache[key] = (names, {n: i for i, n in enumerate(names)}, lab, an)
        names, idx, lab, an = cache[key]
        tx = [t for t in lib[x].targets if t in idx]
        ty = [t for t in lib[y].targets if t in idx]
        if not tx or not ty:
            print(f"{key:20} {x+'+'+y:30} {'target(s) not on network':>22}"); continue
        if set(tx) & set(ty):
            print(f"{key:20} {x+'+'+y:30} {'/'.join(tx[:1])+'='+'/'.join(ty[:1]):>22}  same target: not evaluable"); continue
        # most informative target pair: the highest-bill target of each drug
        bx = max(tx, key=lambda t: an.bill[idx[t]]); by = max(ty, key=lambda t: an.bill[idx[t]])
        if (key, frozenset((bx, by))) in seen:        # same target pair already counted (e.g. 3 BRAF+MEK approvals)
            print(f"{key:20} {x+'+'+y:30} {bx+'/'+by:>22}  duplicate target pair, not counted again")
            continue
        seen.add((key, frozenset((bx, by))))
        same = lab[idx[bx]] == lab[idx[by]]
        ev_n += 1
        ev_diff.append(0 if same else 1)
        print(f"{key:20} {x+'+'+y:30} {bx+'/'+by:>22}  {lab[idx[bx]]} vs {lab[idx[by]]}   "
              f"{'SAME module' if same else 'different modules'}")
    if not ev_n:
        return
    print(f"\napproved combinations evaluable: {ev_n}; hit different modules: {sum(ev_diff)} ({sum(ev_diff)/ev_n*100:.0f}%)")

    # null: random pairs of approved drugs with distinct targets on the same disease networks
    keys = sorted({k for k, *_ in PAIRS if k in cache})
    fr = []
    for key in keys:
        names, idx, lab, an = cache[key]
        pool = [d for d in lib.values() if d.status == "approved" and any(t in idx for t in d.targets)]
        diffs = []
        for _ in range(4000):
            d1, d2 = rng.choice(len(pool), 2, replace=False)
            t1 = [t for t in pool[d1].targets if t in idx]; t2 = [t for t in pool[d2].targets if t in idx]
            if set(t1) & set(t2):
                continue
            b1 = max(t1, key=lambda t: an.bill[idx[t]]); b2 = max(t2, key=lambda t: an.bill[idx[t]])
            diffs.append(lab[idx[b1]] != lab[idx[b2]])
        fr.append(np.mean(diffs))
    print(f"random approved-drug pairs on the same networks hit different modules {np.mean(fr)*100:.0f}% of the time "
          f"(per-disease range {min(fr)*100:.0f}-{max(fr)*100:.0f}%)")
    from scipy.stats import binom
    p_same_null = 1 - np.mean(fr)
    k_same = ev_n - sum(ev_diff)
    print(f"binomial: P(>= {k_same} same-module of {ev_n} | null same-module rate {p_same_null:.2f}) = "
          f"{binom.sf(k_same - 1, ev_n, p_same_null):.3f}")
    print("Reading: if approved combinations land in different modules no more often than random pairs, "
          "'one drug per module' is not what clinical combination design actually does.")

    # Second test: where do the approved pairs rank among ALL coupled drugged pairs by the
    # 'vertical' score (bill_i + bill_j) * coupling_ij ?  (100% = top, 50% = chance)
    from .combo_designer import vertical_pairs
    print("\nRank of each approved pair among all coupled approved-drug pairs on its disease network (vertical score):")
    pct, seen2 = [], set()
    for key, x, y, note in PAIRS:
        if key not in cache or x not in lib or y not in lib:
            continue
        names, idx, lab, an = cache[key]
        tx = [t for t in lib[x].targets if t in idx]; ty = [t for t in lib[y].targets if t in idx]
        if not tx or not ty or set(tx) & set(ty):
            continue
        bx = max(tx, key=lambda t: an.bill[idx[t]]); by = max(ty, key=lambda t: an.bill[idx[t]])
        if (key, frozenset((bx, by))) in seen2:
            continue
        seen2.add((key, frozenset((bx, by))))
        allp = vertical_pairs(an, lib, a.modules, approved_only=True, same_module_only=False)
        pos = next((r for r, p in enumerate(allp) if {p["a"], p["b"]} == {bx, by}), None)
        if pos is None:
            print(f"  {key:20} {x}+{y:20} {bx}/{by}: NOT RANKABLE (pair absent from the candidate list; "
                  f"excluded from the percentile test)")
            continue
        pct.append(1 - pos / len(allp))
        print(f"  {key:20} {x}+{y:20} {bx}/{by}: rank {pos+1} of {len(allp)}  (top {100*(pos+1)/len(allp):.0f}%)")
    # Third test: are approved pairs DIRECTLY LINKED in the network more often than drugged pairs in
    # general?  (The coupling magnitude is ~1 for every STRING edge >= 0.9, so linkage is the part of
    # "coupling" that can carry signal.)
    linked, null_rates, seen3 = [], [], set()
    for key, x, y, note in PAIRS:
        if key not in cache or x not in lib or y not in lib:
            continue
        names, idx, lab, an = cache[key]
        tx = [t for t in lib[x].targets if t in idx]; ty = [t for t in lib[y].targets if t in idx]
        if not tx or not ty or set(tx) & set(ty):
            continue
        bx = max(tx, key=lambda t: an.bill[idx[t]]); by = max(ty, key=lambda t: an.bill[idx[t]])
        if (key, frozenset((bx, by))) in seen3:
            continue
        seen3.add((key, frozenset((bx, by))))
        linked.append(an.C[idx[bx], idx[by]] > 0)
        prots = [p for p in idx if any(t == p for d in lib.values() if d.status == "approved" for t in d.targets)]
        null_rates.append(np.mean([an.C[idx[prots[u]], idx[prots[v]]] > 0
                                   for u in range(len(prots)) for v in range(u + 1, len(prots))]))
    if linked:
        from scipy.stats import binom as _binom
        k, n, p0 = sum(linked), len(linked), float(np.mean(null_rates))
        print(f"\ndirectly linked in the network: {k}/{n} approved pairs vs {p0*100:.0f}% of all approved-drug-target pairs "
              f"(binomial P(>= {k}) = {_binom.sf(k - 1, n, p0):.3f})")
    if pct:
        from scipy.stats import wilcoxon
        print(f"median percentile {np.median(pct)*100:.0f}% (chance 50%); Wilcoxon vs 50%: p = "
              f"{wilcoxon(np.array(pct) - 0.5).pvalue:.3f}  (n={len(pct)})")


if __name__ == "__main__":
    main()
