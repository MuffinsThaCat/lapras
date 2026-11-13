"""Large-scale combination validation from FDA-approved indications.

Systematically mines openFDA drug labels for approved "in combination with" regimens,
maps both partners to the Lapras drug library and tests enrichment for direct network
linkage and bill-sum vs random drug pairs on the union network.  Merged with the FDC
(fixed-dose combination) pairs from validate_fdc.py for the most comprehensive set.

The hand-picked validate_combos.py (n=8, p=0.014 on linkage) is left as-is; this script
replaces the "n is too small" limitation with a systematic, reproducible dataset.

    python3 -m lapras_tools.validate_combos_large --fetch    # mine openFDA labels (~2 min)
    python3 -m lapras_tools.validate_combos_large            # analysis
    python3 -m lapras_tools.validate_combos_large --time-split 2020
"""
import argparse, json, os, re, time, urllib.parse, urllib.request
import numpy as np
from .core import analyze_network
from .druglib import default_library

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "fda_combination_indications.json")
API = "https://api.fda.gov/drug/label.json"


def fetch():
    results = {}
    for sort in ("effective_time:asc", "effective_time:desc"):
        for skip in range(0, 25001, 100):
            search = 'indications_and_usage:"in combination with"'
            q = urllib.parse.urlencode({"search": search, "limit": 100, "skip": skip, "sort": sort})
            try:
                d = json.load(urllib.request.urlopen(f"{API}?{q}", timeout=120))
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break
                print(f"  HTTP {e.code} at {sort} skip {skip}")
                time.sleep(5)
                continue
            except Exception as e:
                print(f"  error at {sort} skip {skip}: {e}")
                break
            for r in d.get("results", []):
                spl = r.get("id", "")
                if spl in results:
                    continue
                results[spl] = {
                    "generics": [g.lower() for g in r.get("openfda", {}).get("generic_name", [])],
                    "text": " ".join(r.get("indications_and_usage", [])),
                    "eff": r.get("effective_time", ""),
                }
            batch = len(d.get("results", []))
            if skip % 500 == 0:
                print(f"  {sort} skip {skip}: {len(results)} labels", flush=True)
            if batch < 100:
                break
            time.sleep(0.25)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(list(results.values()), f)
    print(f"saved {len(results)} labels -> {CACHE}")


_SALT = re.compile(r"\s+(hydrochloride|hcl|mesylate|maleate|tartrate|fumarate|succinate|"
                   r"besylate|tosylate|acetate|citrate|sulfate|phosphate|sodium|potassium|"
                   r"calcium|dihydrate|monohydrate|decanoate|dimeglumine|disodium|tromethamine|"
                   r"hemifumarate|esylate|bromide|nitrate|oxalate|malonate|lactate|xinafoate)$")


def _norm(name, lib_names):
    name = name.strip().lower()
    for cand in (name, _SALT.sub("", name), name.split()[0] if name else ""):
        if cand in lib_names:
            return cand
    return None


_COMBO_RE = re.compile(
    r"(?:in combination with|combined with|coadministered with|"
    r"administered (?:with|in conjunction with)|used (?:with|together with))\b", re.I)

_NOT_DRUGS = {"angiotensin ii", "angiotensin", "dopamine", "serotonin", "epinephrine",
              "norepinephrine", "acetylcholine", "histamine", "prostaglandin"}


def _is_same_drug(a, b):
    """True if one name is a prefix/substring of the other (prodrug/salt forms)."""
    return a in b or b in a


def _extract(text, lib_names, exclude):
    """Drug names mentioned near combination phrases in text."""
    lower = text.lower()
    hits = set()
    for m in _COMBO_RE.finditer(lower):
        pre = lower[max(0, m.start() - 40):m.start()]
        if re.search(r"\b(?:not|no longer|without|avoid|prior to|instead of|should not|"
                     r"must not|do not|is not|are not|was not|were not|not be)\b", pre):
            continue
        window = lower[m.end():m.end() + 250]
        stop = re.search(r"\.\s+[A-Z]|\n\n|;\s+[a-z]", text[m.end():m.end() + 250])
        if stop:
            window = window[:stop.start()]
        for name in lib_names:
            if len(name) < 4 or name in exclude or name in _NOT_DRUGS:
                continue
            if re.search(r"(?<![a-z-])" + re.escape(name) + r"(?![a-z-])", window):
                hits.add(name)
    return hits


def parse(lib):
    with open(CACHE) as f:
        labels = json.load(f)
    names = set(lib)
    pairs = {}
    for lab in labels:
        own = set()
        for g in lab["generics"]:
            for part in re.split(r"\s+and\s+|[;,/]", g):
                mapped = _norm(part, names)
                if mapped:
                    own.add(mapped)
        if not own:
            continue
        partners = _extract(lab["text"], names, own)
        for d in own:
            for p in partners:
                if d == p or _is_same_drug(d, p):
                    continue
                key = frozenset((d, p))
                if key not in pairs:
                    pairs[key] = lab["eff"]
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--modules", type=int, default=40)
    ap.add_argument("--time-split", type=int, default=0)
    a = ap.parse_args()
    if a.fetch:
        fetch()
        return
    if not os.path.exists(CACHE):
        raise SystemExit("no cache; run with --fetch first")

    lib = default_library()
    pairs = parse(lib)
    print(f"FDA label mining: {len(pairs)} drug pairs (both in library)")

    fdc_path = os.path.join(HERE, "data", "fda_fdc_products.json")
    if os.path.exists(fdc_path):
        from .validate_fdc import pairs_from_cache
        n_before = len(pairs)
        for k in pairs_from_cache(lib):
            pairs.setdefault(k, "FDC")
        print(f"merged FDC: +{len(pairs) - n_before} new (total {len(pairs)})")

    from .interactions import union_network
    C, names = union_network()
    idx = {n: i for i, n in enumerate(names)}
    an = analyze_network(C, names)
    lab = an.label_vector(an.level_with_at_most(a.modules))

    ev = []
    skip_no_target, skip_shared = 0, 0
    for pr, eff in pairs.items():
        x, y = sorted(pr)
        tx = [t for t in lib[x].targets if t in idx]
        ty = [t for t in lib[y].targets if t in idx]
        if not tx or not ty:
            skip_no_target += 1; continue
        if set(tx) & set(ty):
            skip_shared += 1; continue
        bx = max(tx, key=lambda t: an.bill[idx[t]])
        by = max(ty, key=lambda t: an.bill[idx[t]])
        ix, iy = idx[bx], idx[by]
        linked = C[ix, iy] > 0
        twohop = (not linked) and float((C[ix] * C[iy]).max()) > 0
        same_mod = lab[ix] == lab[iy]
        bill_sum = float(max(an.bill[ix], 0) + max(an.bill[iy], 0))
        ev.append(dict(x=x, y=y, bx=bx, by=by, linked=linked, twohop=twohop,
                       same_mod=same_mod, bill_sum=bill_sum,
                       eff=eff if isinstance(eff, str) else ""))

    drugs = sorted({e["x"] for e in ev} | {e["y"] for e in ev})
    print(f"\nevaluable: {len(ev)} pairs over {len(drugs)} drugs "
          f"(skipped: {skip_no_target} off-network, {skip_shared} shared-target)")
    if len(ev) < 10:
        raise SystemExit("too few evaluable pairs")

    lk = np.array([e["linked"] for e in ev])
    th = np.array([e["twohop"] for e in ev])
    reachable = lk | th
    sm = np.array([e["same_mod"] for e in ev])
    bs = np.array([e["bill_sum"] for e in ev])

    rng = np.random.default_rng(42)
    null_lk, null_th, null_sm, null_bs = [], [], [], []
    while len(null_lk) < 40000:
        i, j = rng.choice(len(drugs), 2, replace=False)
        da, db = lib[drugs[i]], lib[drugs[j]]
        ta = [t for t in da.targets if t in idx]
        tb = [t for t in db.targets if t in idx]
        if not ta or not tb or set(ta) & set(tb):
            continue
        ba = max(ta, key=lambda t: an.bill[idx[t]])
        bb = max(tb, key=lambda t: an.bill[idx[t]])
        ia, ib = idx[ba], idx[bb]
        direct = C[ia, ib] > 0
        hop2 = (not direct) and float((C[ia] * C[ib]).max()) > 0
        null_lk.append(direct)
        null_th.append(direct or hop2)
        null_sm.append(lab[ia] == lab[ib])
        null_bs.append(float(max(an.bill[ia], 0) + max(an.bill[ib], 0)))

    from scipy.stats import binom, mannwhitneyu
    p_lk = float(np.mean(null_lk))
    p_th = float(np.mean(null_th))
    p_sm = float(np.mean(null_sm))
    n = len(ev)
    k_lk, k_sm = int(lk.sum()), int(sm.sum())
    k_reach = int(reachable.sum())

    print(f"\n{'Signal':24s} {'approved':>15s} {'random':>10s} {'p-value':>12s}")
    print("-" * 65)
    print(f"{'directly linked':24s} {k_lk}/{n} = {k_lk/n*100:4.0f}%  {p_lk*100:7.0f}%  "
          f"{binom.sf(k_lk - 1, n, p_lk):.2g}")
    print(f"{'reachable (1-2 hop)':24s} {k_reach}/{n} = {k_reach/n*100:4.0f}%  {p_th*100:7.0f}%  "
          f"{binom.sf(k_reach - 1, n, p_th):.2g}")
    print(f"{'same module (k=' + str(a.modules) + ')':24s} {k_sm}/{n} = {k_sm/n*100:4.0f}%  {p_sm*100:7.0f}%  "
          f"{binom.sf(k_sm - 1, n, p_sm):.2g}")
    u_stat, p_u = mannwhitneyu(bs, np.array(null_bs), alternative="greater")
    print(f"{'bill sum (mean)':24s} {np.mean(bs):10.3f}  {np.mean(null_bs):10.3f}  {p_u:.2g}")

    # pseudo-replication check
    from collections import Counter
    freq = Counter(d for e in ev for d in (e["x"], e["y"]))
    print(f"\nmost-used drugs: " + ", ".join(f"{d} ({c})" for d, c in freq.most_common(5)))
    ind_l, ind_n = [], []
    for _ in range(2000):
        order = rng.permutation(len(ev)); used = set(); sel = []
        for k in order:
            e = ev[k]
            if e["x"] not in used and e["y"] not in used:
                used |= {e["x"], e["y"]}; sel.append(e["linked"])
        ind_n.append(len(sel)); ind_l.append(np.mean(sel) if sel else 0)
    mn, ml = int(np.mean(ind_n)), float(np.mean(ind_l))
    k_eq = int(round(ml * mn))
    print(f"independent-set resamples (~{mn} pairs): linked {ml*100:.0f}% "
          f"(5-95%: {np.percentile(ind_l, 5)*100:.0f}-{np.percentile(ind_l, 95)*100:.0f}%) "
          f"vs {p_lk*100:.0f}%")
    print(f"  binomial: P(>= {k_eq} of {mn} | {p_lk*100:.0f}%) = {binom.sf(k_eq - 1, mn, p_lk):.2g}")

    # partner-shuffle permutation
    xs, ys = [e["x"] for e in ev], [e["y"] for e in ev]
    obs_rate = float(lk.mean())
    cnt = 0; trials = 3000
    for _ in range(trials):
        perm = rng.permutation(len(ys)); s = []
        for ii in range(len(xs)):
            da, db = lib[xs[ii]], lib[ys[perm[ii]]]
            ta = [t for t in da.targets if t in idx]
            tb = [t for t in db.targets if t in idx]
            if not ta or not tb or set(ta) & set(tb):
                continue
            ba = max(ta, key=lambda t: an.bill[idx[t]])
            bb = max(tb, key=lambda t: an.bill[idx[t]])
            s.append(C[idx[ba], idx[bb]] > 0)
        if s and np.mean(s) >= obs_rate:
            cnt += 1
    print(f"partner-shuffle permutation: P(linked >= {obs_rate:.2f}) = {(cnt + 1) / (trials + 1):.4f}")

    # hormone-receptor exclusion (same check as validate_fdc)
    horm = {"ESR1", "PGR", "AR", "ESR2", "NR3C1"}
    ev_nh = [e for e in ev if not ((set(lib[e["x"]].targets) | set(lib[e["y"]].targets)) & horm)]
    if len(ev_nh) >= 10:
        k_nh = sum(e["linked"] for e in ev_nh)
        drugs_nh = sorted({e["x"] for e in ev_nh} | {e["y"] for e in ev_nh})
        nl_nh = []
        while len(nl_nh) < 30000:
            i, j = rng.choice(len(drugs_nh), 2, replace=False)
            da, db = lib[drugs_nh[i]], lib[drugs_nh[j]]
            ta = [t for t in da.targets if t in idx]
            tb = [t for t in db.targets if t in idx]
            if not ta or not tb or set(ta) & set(tb):
                continue
            ba = max(ta, key=lambda t: an.bill[idx[t]])
            bb = max(tb, key=lambda t: an.bill[idx[t]])
            nl_nh.append(C[idx[ba], idx[bb]] > 0)
        p_nh = float(np.mean(nl_nh))
        print(f"\nexcluding hormone-receptor drugs: {k_nh}/{len(ev_nh)} linked = {k_nh/len(ev_nh)*100:.0f}% "
              f"vs {p_nh*100:.0f}%; binomial p = {binom.sf(k_nh - 1, len(ev_nh), p_nh):.2g}")

    # time split
    if a.time_split:
        cut = str(a.time_split)
        before = [e for e in ev if e["eff"] and e["eff"][:4] < cut]
        after = [e for e in ev if e["eff"] and e["eff"][:4] >= cut]
        undated = len(ev) - len(before) - len(after)
        print(f"\ntime split at {a.time_split}: {len(before)} before, {len(after)} after, {undated} undated "
              "(effective_time is the label date, not the approval date; approximate)")
        for tag, sub in (("before", before), ("after", after)):
            if len(sub) < 5:
                print(f"  {tag}: too few ({len(sub)})"); continue
            k = sum(e["linked"] for e in sub)
            print(f"  {tag} {a.time_split}: linked {k}/{len(sub)} = {k/len(sub)*100:.0f}%  "
                  f"p = {binom.sf(k - 1, len(sub), p_lk):.2g}")

    # examples
    print(f"\nlinked approved pairs (top by bill sum):")
    shown = 0
    for e in sorted(ev, key=lambda e: -e["bill_sum"]):
        if e["linked"] and shown < 20:
            print(f"  {e['x']:20s} + {e['y']:20s}  {e['bx']}—{e['by']}  bill {e['bill_sum']:.3f}")
            shown += 1
    print(f"\n2-hop connected (not directly linked):")
    shown = 0
    for e in sorted(ev, key=lambda e: -e["bill_sum"]):
        if e["twohop"] and shown < 10:
            print(f"  {e['x']:20s} + {e['y']:20s}  {e['bx']}—{e['by']}  bill {e['bill_sum']:.3f}")
            shown += 1
    print(f"\nno path within 2 hops (sample):")
    shown = 0
    for e in ev:
        if not e["linked"] and not e["twohop"] and shown < 10:
            print(f"  {e['x']:20s} + {e['y']:20s}  {e['bx']}—{e['by']}")
            shown += 1


if __name__ == "__main__":
    main()
