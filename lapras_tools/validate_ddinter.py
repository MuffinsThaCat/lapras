"""Interaction tool vs DDInter (curated drug-drug interactions with severity), an INDEPENDENT ground truth.

validate_ddi.py scored the interaction tool against FDA label text, and the PK layer's CYP roles come
from that same text (circular).  DDInter (ddinter2.scbdd.com, 8 ATC-class CSV files) is curated
separately, so scoring the network score and the PK flag against it has no such circularity.

    python3 -m lapras_tools.validate_ddinter [--level major|moderate|any]
    (files: data/ddinter/ddinter_{A,B,D,H,L,P,R,V}.csv; download links on https://ddinter2.scbdd.com/download/)

Ground truth: a pair is positive when DDInter lists it (at or above --level); negatives are pairs of
drugs that BOTH appear somewhere in DDInter (so each has been curated) but are not listed together.
Only pairs where both drugs are in the library, have a network target and share no target are scored
for the network score; the PK flag is scored on pairs where both drugs have a label-derived enzyme role.

Caveats: DDInter lists interactions of any mechanism (mostly PK or class effects) and any absence is
only 'not curated as interacting'; severity 'Unknown' is excluded from the 'major'/'moderate' levels.
"""
import argparse, csv, glob, os
import numpy as np
from collections import Counter
from .druglib import default_library
from .validate_fdc import normalise
from .validate_ddi import auc

HERE = os.path.dirname(os.path.abspath(__file__))


def load_pairs(lib_names, level="any"):
    order = {"Minor": 1, "Moderate": 2, "Major": 3}
    cut = {"any": 0, "moderate": 2, "major": 3}[level]
    best, seen = {}, set()
    for f in sorted(glob.glob(os.path.join(HERE, "data", "ddinter", "ddinter_*.csv"))):
        for r in csv.DictReader(open(f, encoding="utf-8", errors="replace")):
            a, b, lv = r.get("Drug_A"), r.get("Drug_B"), r.get("Level")
            if not a or not b or not lv:
                continue                                        # malformed row
            na, nb = normalise(a.strip().lower(), lib_names), normalise(b.strip().lower(), lib_names)
            seen.update(x for x in (na, nb) if x)
            if na and nb and na != nb:
                k = frozenset((na, nb))
                best[k] = max(best.get(k, 0), order.get(lv, 0))
    pos = {k for k, v in best.items() if v >= max(cut, 1)} if cut else set(best)
    return pos, seen, best


def fetch():
    """Download the eight DDInter ATC-class files (not redistributed with this package: licence terms
    could not be confirmed, see the data-sources section of the README)."""
    import urllib.request
    d = os.path.join(HERE, "data", "ddinter")
    os.makedirs(d, exist_ok=True)
    for c in "ABDHLPRV":
        url = f"https://ddinter2.scbdd.com/static/media/download/ddinter_downloads_code_{c}.csv"
        for attempt in range(3):
            try:
                open(os.path.join(d, f"ddinter_{c}.csv"), "wb").write(urllib.request.urlopen(url, timeout=180).read())
                print(f"  {c}: ok"); break
            except Exception as e:
                print(f"  {c}: attempt {attempt+1} failed ({e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["any", "moderate", "major"], default="any")
    ap.add_argument("--modules", type=int, default=40)
    ap.add_argument("--fetch", action="store_true", help="download the DDInter CSVs into data/ddinter/ and exit")
    a = ap.parse_args()
    if a.fetch:
        fetch()
        return
    if not glob.glob(os.path.join(HERE, "data", "ddinter", "ddinter_*.csv")):
        raise SystemExit("no DDInter files; run with --fetch first")
    from .interactions import Context, pair_flag
    from .pk_layer import load_roles, pk_pair
    lib = default_library()
    pos, seen, best = load_pairs(set(lib), a.level)
    print(f"DDInter pairs with both drugs in the library: {len(best)} (level >= {a.level}: {len(pos)}); "
          f"{len(seen)} library drugs appear in DDInter")
    lv = Counter({0: "Unknown", 1: "Minor", 2: "Moderate", 3: "Major"}[v] for v in best.values())
    print("  severity mix (all levels):", dict(lv))
    ctx = Context(None, modules=a.modules)
    drugs = sorted(d for d in seen if any(t in ctx.idx for t in lib[d].targets))
    print(f"{len(drugs)} DDInter drugs have a network target")

    y, s, pairs = [], [], []
    st_pos = st_neg = 0                                          # same-target pairs (excluded from the scores)
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            f = pair_flag(ctx, lib[drugs[i]], lib[drugs[j]])
            if f is not None and f["same_target"]:
                if frozenset((drugs[i], drugs[j])) in pos:
                    st_pos += 1
                else:
                    st_neg += 1
            if f is None or f["same_target"]:
                continue
            k = frozenset((drugs[i], drugs[j]))
            if k in best and k not in pos:
                continue                                        # listed below the chosen severity: neither positive nor negative
            y.append(k in pos); s.append(f["score"]); pairs.append((drugs[i], drugs[j]))
    y, s = np.array(y), np.array(s)
    print(f"\nsame-target pairs (excluded from the scores below): {st_pos + st_neg}, of which {st_pos} DDInter-listed "
          f"({st_pos/max(st_pos+st_neg,1)*100:.1f}% vs {y.mean()*100:.1f}% for different-target pairs)")
    print(f"NETWORK score: {len(y)} scorable pairs, {int(y.sum())} DDInter-listed ({y.mean()*100:.1f}%)")
    print(f"  AUC {auc(s[y], s[~y]):.3f}")
    for thr, name in ((0.95, "OVERLAP+"), (1.2, "TIGHT")):
        fl = s >= thr
        print(f"  {name:9s} flagged {fl.mean()*100:4.1f}% of pairs; precision {y[fl].mean()*100:4.1f}% "
              f"(base {y.mean()*100:.1f}%); recall {fl[y].mean()*100:4.1f}%")
    cnt = Counter(d for p, yy in zip(pairs, y) if yy for d in p)
    dsum = np.array([cnt[p[0]] + cnt[p[1]] for p in pairs])
    q = np.quantile(dsum, [0.5, 0.8])
    for lo, hi, tag in zip([-1, q[0], q[1]], [q[0], q[1], dsum.max() + 1], ("low", "mid", "high")):
        m = (dsum > lo) & (dsum <= hi)
        if y[m].sum() >= 10 and (~y[m]).sum() >= 10:
            print(f"  popularity stratum {tag:5s}: n={m.sum():6d} positives={int(y[m].sum()):5d} AUC {auc(s[m][y[m]], s[m][~y[m]]):.3f}")

    # PK layer (roles from FDA labels; ground truth from DDInter: not circular)
    roles = load_roles(set(lib))
    pk_drugs = sorted(d for d in seen if d in roles and any(roles[d].values()))
    py, ps = [], []
    for i in range(len(pk_drugs)):
        for j in range(i + 1, len(pk_drugs)):
            k = frozenset((pk_drugs[i], pk_drugs[j]))
            if k in best and k not in pos:
                continue
            py.append(k in pos); ps.append(len(pk_pair(roles, pk_drugs[i], pk_drugs[j])))
    py, ps = np.array(py), np.array(ps)
    print(f"\nPK layer: {len(pk_drugs)} drugs with enzyme roles, {len(py)} pairs, {int(py.sum())} DDInter-listed ({py.mean()*100:.1f}%)")
    fl = ps > 0
    if fl.any() and py.any():
        print(f"  PK flag precision {py[fl].mean()*100:.1f}% (base {py.mean()*100:.1f}%, lift {py[fl].mean()/py.mean():.2f}x), "
              f"recall {fl[py].mean()*100:.1f}%; AUC of PK count {auc(ps[py], ps[~py]):.3f}")

    both = [k for k, (x, z) in enumerate(pairs) if x in roles and z in roles and any(roles[x].values()) and any(roles[z].values())]
    if len(both) > 500 and y[both].sum() >= 30:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        yb = y[both]
        net = s[both].reshape(-1, 1)
        pk = np.array([len(pk_pair(roles, *pairs[k])) for k in both], float).reshape(-1, 1)
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        out = {}
        for nm, X in (("network only", net), ("PK count only", pk), ("network + PK", np.hstack([net, pk, pk > 0]))):
            p = cross_val_predict(LogisticRegression(max_iter=1000, class_weight="balanced"), X, yb, cv=skf, method="predict_proba")[:, 1]
            out[nm] = auc(p[yb], p[~yb])
        print(f"  on {len(both)} pairs scorable by both (positives {int(yb.sum())}), 5-fold CV AUC: " +
              ", ".join(f"{k} {v:.3f}" for k, v in out.items()))


if __name__ == "__main__":
    main()
