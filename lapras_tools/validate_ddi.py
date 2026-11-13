"""Interaction predictor vs FDA label drug-interaction sections (a systematic, noisy ground truth).

Replaces the 7 hand-picked known interactions in interactions.py with every pair of library
drugs where one FDA label's "Drug Interactions" section names the other.

    python3 -m lapras_tools.validate_ddi --fetch     # one openFDA label query per library drug (~10 min)
    python3 -m lapras_tools.validate_ddi             # analysis

Ground truth caveats (read before trusting a number):
  * "named in the other drug's interaction section" includes pharmacokinetic interactions (CYP
    inhibition, absorption), which the Lapras interaction predictor does NOT model at all, and
    mentions that are about absence of interaction.  So the ceiling on AUC is low by construction.
  * Drugs with long labels name many partners; the analysis also reports a degree-matched
    comparison so a drug's popularity is not mistaken for signal.
  * Only pairs where both drugs have a target on the union network and share none are scored.
"""
import argparse, json, os, re, time, urllib.parse, urllib.request
import numpy as np
from .druglib import default_library

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "fda_label_interactions.json")
API = "https://api.fda.gov/drug/label.json"


def fetch(lib):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [n for n in lib if n not in cache]
    print(f"{len(cache)} cached, {len(todo)} to fetch", flush=True)
    from concurrent.futures import ThreadPoolExecutor

    def one(name):
        q = urllib.parse.urlencode({"search": f'openfda.generic_name:"{name}"', "limit": 3})
        for attempt in range(3):
            try:
                d = json.load(urllib.request.urlopen(f"{API}?{q}", timeout=60))
                parts = []
                for r in d.get("results", []):
                    parts += r.get("drug_interactions", [])
                return name, (" ".join(parts).lower() if parts else "")
            except urllib.error.HTTPError as e:
                if e.code == 404:                      # no label for this name
                    return name, ""
                time.sleep(2 * (attempt + 1))
            except Exception:
                time.sleep(2 * (attempt + 1))
        return name, None                              # failed: leave uncached so a rerun retries it

    # 4 workers x ~2 s per call stays well under openFDA's 240 requests/minute limit
    with ThreadPoolExecutor(4) as ex:
        for k, (name, txt) in enumerate(ex.map(one, todo)):
            if txt is not None:
                cache[name] = txt
            if k % 100 == 0:
                json.dump(cache, open(CACHE, "w"))
                print(f"  {k}/{len(todo)}", flush=True)
    json.dump(cache, open(CACHE, "w"))
    failed = [n for n in lib if n not in cache]
    if failed:
        print(f"  {len(failed)} drugs failed and are uncached (rerun --fetch): {failed[:8]}")
    print(f"saved {len(cache)} labels ({sum(1 for v in cache.values() if v)} with interaction text)")


def mentions(text, name):
    return len(name) >= 5 and re.search(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])", text) is not None


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    order = np.concatenate([pos, neg]).argsort(kind="mergesort")
    ranks = np.empty(len(order)); allv = np.concatenate([pos, neg])
    # average ranks for ties
    srt = allv[order]; r = np.arange(1, len(allv) + 1, dtype=float)
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        r[i:j + 1] = (i + 1 + j + 1) / 2
        i = j + 1
    ranks[order] = r
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--modules", type=int, default=40)
    a = ap.parse_args()
    lib = default_library()
    if a.fetch:
        fetch(lib)
        return
    if not os.path.exists(CACHE):
        raise SystemExit("no cache; run with --fetch first")
    from .interactions import Context, pair_flag
    labels = json.load(open(CACHE))
    ctx = Context(None, modules=a.modules)
    have = [n for n, t in labels.items() if t and n in lib and any(x in ctx.idx for x in lib[n].targets)]
    print(f"{len(have)} drugs with an FDA interaction section and a network target")

    positives = set()
    for n in have:
        for m in have:
            if m != n and mentions(labels[n], m):
                positives.add(frozenset((n, m)))
    pairs_scored, y, s, link = [], [], [], []
    hs = sorted(have)
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            f = pair_flag(ctx, lib[hs[i]], lib[hs[j]])
            if f is None or f["same_target"]:
                continue
            pairs_scored.append((hs[i], hs[j]))
            y.append(frozenset((hs[i], hs[j])) in positives); s.append(f["score"]); link.append(f["prox"])
    y = np.array(y); s = np.array(s)
    print(f"{len(y)} scorable pairs; {int(y.sum())} named in an interaction section ({y.mean()*100:.1f}%)")
    print(f"AUC of the interaction score for 'named as interacting': {auc(s[y], s[~y]):.3f}")
    for thr, lab in ((0.95, "OVERLAP or higher"), (1.2, "TIGHT")):
        fl = s >= thr
        print(f"  flagged {lab:18s}: {fl.mean()*100:4.1f}% of pairs; precision {y[fl].mean()*100:4.1f}% "
              f"(base rate {y.mean()*100:.1f}%); recall {fl[y].mean()*100:4.1f}%")

    # degree-matched check: drugs with long label sections name many partners
    cnt = {n: 0 for n in hs}
    for pr in positives:
        for n in pr:
            if n in cnt:
                cnt[n] += 1
    dsum = np.array([cnt[x] + cnt[z] for x, z in pairs_scored])
    q = np.quantile(dsum, [0.5, 0.8])
    print("AUC within strata of (partner mentions of drug A + drug B), to remove label-length/popularity effects:")
    edges = [-1, q[0], q[1], dsum.max() + 1]
    for lo, hi, tag in zip(edges[:-1], edges[1:], ("low", "mid", "high")):
        m = (dsum > lo) & (dsum <= hi)
        if y[m].sum() >= 5 and (~y[m]).sum() >= 5:
            print(f"  {tag:5s} n={m.sum():6d} positives={int(y[m].sum()):4d}  AUC {auc(s[m][y[m]], s[m][~y[m]]):.3f}")
    # permutation null on drug labels (keeps popularity): shuffle which drug is which
    rng = np.random.default_rng(0)
    obs = auc(s[y], s[~y])
    idx = {n: i for i, n in enumerate(hs)}
    null = []
    for _ in range(200):
        perm = rng.permutation(len(hs))
        yy = np.array([frozenset((hs[perm[idx[x]]], hs[perm[idx[z]]])) in positives for x, z in pairs_scored])
        if yy.sum() >= 5:
            null.append(auc(s[yy], s[~yy]))
    print(f"drug-label permutation null (popularity preserved): AUC {np.mean(null):.3f} +/- {np.std(null):.3f}; "
          f"observed {obs:.3f}, z = {(obs - np.mean(null)) / np.std(null):+.1f}")

    # ---------------------------------------------------------------- pharmacokinetic layer
    from .pk_layer import load_roles, pk_pair
    roles = load_roles(set(lib))
    pk_drugs = sorted(n for n, r in roles.items() if r["substrate"] or r["inhibitor"] or r["inducer"])
    lab_all = sorted(n for n, t in labels.items() if t and n in lib)
    pos_all = set()
    for n in lab_all:
        for m in lab_all:
            if m != n and mentions(labels[n], m):
                pos_all.add(frozenset((n, m)))
    py, ps, pairs_pk = [], [], []
    for i in range(len(pk_drugs)):
        for j in range(i + 1, len(pk_drugs)):
            a_, b_ = pk_drugs[i], pk_drugs[j]
            py.append(frozenset((a_, b_)) in pos_all)
            ps.append(len(pk_pair(roles, a_, b_)))
            pairs_pk.append((a_, b_))
    py = np.array(py); ps = np.array(ps)
    print(f"\nPK layer (CYP/P-gp roles mined from the labels themselves; see pk_layer.py for the circularity caveat):")
    print(f"  {len(pk_drugs)} drugs with >= 1 enzyme role, {len(py)} pairs, {int(py.sum())} named ({py.mean()*100:.1f}%); "
          f"{int((ps > 0).sum())} pairs flagged by a perpetrator->victim enzyme match")
    if (ps > 0).any() and py.any():
        fl = ps > 0
        print(f"  PK flag: precision {py[fl].mean()*100:.1f}% (base rate {py.mean()*100:.1f}%, lift {py[fl].mean()/py.mean():.1f}x), "
              f"recall {fl[py].mean()*100:.1f}%;  AUC of PK count {auc(ps[py], ps[~py]):.3f}")
    # combined with the network score on pairs that are scorable by both
    both =[k for k, (x, z) in enumerate(pairs_scored) if x in roles and z in roles
            and (roles[x]["substrate"] | roles[x]["inhibitor"] | roles[x]["inducer"])
            and (roles[z]["substrate"] | roles[z]["inhibitor"] | roles[z]["inducer"])]
    if len(both) > 500:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        X_net = s[both].reshape(-1, 1)
        pk = np.array([len(pk_pair(roles, *pairs_scored[k])) for k in both], float).reshape(-1, 1)
        yb = y[both]
        if yb.sum() >= 20:
            skf = StratifiedKFold(5, shuffle=True, random_state=0)
            out = {}
            for nm_, X in (("network score only", X_net), ("PK count only", pk), ("network + PK", np.hstack([X_net, pk, (pk > 0)]))):
                p_ = cross_val_predict(LogisticRegression(max_iter=1000, class_weight="balanced"), X, yb, cv=skf, method="predict_proba")[:, 1]
                out[nm_] = auc(p_[yb], p_[~yb])
            print(f"  on {len(both)} pairs scorable by both (positives {int(yb.sum())}), 5-fold CV AUC: " +
                  ", ".join(f"{k} {v:.3f}" for k, v in out.items()))


if __name__ == "__main__":
    main()
