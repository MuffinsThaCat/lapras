"""Tool 4 - Drug interaction predictor (pharmacodynamic overlap).

Maps each drug on a medication list to its targets on the Lapras coupling
network and flags pairs whose targets sit in the same tightly-coupled module or
are directly/two-hop linked.  Overlap means the drugs push on the same part of
the network; whether that is a synergy or a hazard depends on the disease and
on the direction of the effect, which the network alone cannot give.  The
output labels direction (both inhibit / opposing) and leaves the judgement to
the clinician.

    python3 -m lapras_tools.interactions lisinopril losartan spironolactone atorvastatin
    python3 -m lapras_tools.interactions dabrafenib trametinib --context melanoma
    python3 -m lapras_tools.interactions --validate

MEASURED PERFORMANCE against DDInter, an independent curated database (validate_ddinter.py): network
score AUC 0.52 for any listed interaction and 0.60 for Major ones (4,435 positives); PK flag precision
20.5% vs a 10.3% base rate for Major interactions (2x lift, recall 18%); network + PK cross-validated AUC
0.60 for Major.  Weak but real, and complementary.  Earlier, against FDA label text (validate_ddi.py,
152,345 drug pairs; circular for the PK layer): AUC 0.529,
precision of the strictest flag 3.8% vs a 1.2% base rate.  Treat flags as hypotheses about shared
pathway position, NOT as predictions that two drugs interact.  (The 7-pair --validate sanity check
below, AUC ~0.89, uses pairs chosen by hand and overstates this.)

NOT covered: pharmacokinetic interactions (CYP450 inhibition/induction,
transporters, protein binding) - these cause a large share of real adverse
interactions and have nothing to do with target coupling.  Not a clinical tool.

Context: default is the union of all 31 disease networks (STRING >= 0.9, 599
proteins); --context <disease> uses that disease's network only.
"""
import argparse, os
import numpy as np
from .core import analyze_network, load_disease, disease_keys, DATA_DIR
from .druglib import default_library

_union_cache = {}


def union_network(score=0.9):
    if score in _union_cache:
        return _union_cache[score]
    edges = {}
    for k in disease_keys():
        with open(os.path.join(DATA_DIR, f"{k}.tsv")) as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 6 or p[2] == p[3] or float(p[5]) < score:
                    continue
                key = tuple(sorted((p[2], p[3])))
                edges[key] = max(edges.get(key, 0), float(p[5]))
    names = sorted({x for e in edges for x in e})
    idx = {n: i for i, n in enumerate(names)}
    C = np.zeros((len(names), len(names)))
    for (a, b), s in edges.items():
        C[idx[a], idx[b]] = C[idx[b], idx[a]] = s
    _union_cache[score] = (C, names)
    return C, names


class Context:
    def __init__(self, disease=None, score=0.9, modules=40):
        if disease:
            C, names, _, self.label = load_disease(disease, score)
        else:
            C, names = union_network(score)
            self.label = "union of 31 disease networks"
        self.an = analyze_network(C, names)
        self.C, self.names = self.an.C, names
        self.idx = {n: i for i, n in enumerate(names)}
        self.lab = self.an.label_vector(self.an.level_with_at_most(modules))
        self.n_modules = len(set(self.lab))

    def proximity(self, i, j):
        """Direct coupling if linked, else best two-hop path strength."""
        if i == j:
            return 1.0
        if self.C[i, j] > 0:
            return float(self.C[i, j])
        return float((self.C[i] * self.C[j]).max())


_SUPPRESS = ("inhibit", "antagon", "block", "degrad", "antibody", "negative", "suppress", "disrupt", "sequest", "cleav")
_ACTIVATE = ("agonist", "activat", "opener", "positive", "stimulat", "inducer", "releas")


def direction(action):
    """-1 suppress, +1 activate, 0 unclear; handles ChEMBL action types
    ('INHIBITOR', 'BLOCKER', 'POSITIVE ALLOSTERIC MODULATOR', ...)."""
    a = (action or "").lower()
    if any(k in a for k in _ACTIVATE) and not any(k in a for k in ("antagon", "inhibit", "negative")):
        return +1
    if any(k in a for k in _SUPPRESS):
        return -1
    return 0


def pair_flag(ctx, d1, d2):
    """Strongest overlap over all target pairs of two drugs; None if no target is on the network."""
    best = None
    for t1 in d1.targets:
        for t2 in d2.targets:
            if t1 not in ctx.idx or t2 not in ctx.idx:
                continue
            i, j = ctx.idx[t1], ctx.idx[t2]
            prox = ctx.proximity(i, j)
            same_mod = ctx.lab[i] == ctx.lab[j]
            score = prox + (0.25 if same_mod else 0.0)
            if best is None or score > best["score"]:
                best = dict(score=score, t1=t1, t2=t2, prox=prox, same_module=bool(same_mod),
                            same_target=(i == j))
    if best is None:
        return None
    s1, s2 = direction(d1.action), direction(d2.action)
    best["direction"] = ("both suppress" if s1 < 0 and s2 < 0 else
                         "both activate" if s1 > 0 and s2 > 0 else
                         "opposing" if s1 * s2 < 0 else "mixed/unclear")
    return best


def level_of(f):
    if f is None:
        return "no data"
    if f["same_target"]:
        return "SAME TARGET"
    if f["same_module"] and f["prox"] >= 0.7:
        return "TIGHT (same module, strongly linked)"
    if f["same_module"] or f["prox"] >= 0.7:
        return "OVERLAP"
    return "independent"


def check(meds, ctx, lib):
    drugs, unknown = [], []
    for m in meds:
        d = lib.get(m.lower())
        (drugs if d else unknown).append(d or m)
    for u in unknown:
        print(f"  ? '{u}' not in the drug library")
    for d in drugs:
        on = [t for t in d.targets if t in ctx.idx]
        if not on:
            print(f"  ? {d.name}: targets {'/'.join(d.targets)} not on this network - cannot assess")
    out = []
    for a in range(len(drugs)):
        for b in range(a + 1, len(drugs)):
            f = pair_flag(ctx, drugs[a], drugs[b])
            out.append((drugs[a], drugs[b], f))
    out.sort(key=lambda x: -(x[2]["score"] if x[2] else -1))
    return out


def report(meds, ctx, lib):
    print(f"Network: {ctx.label} ({len(ctx.names)} proteins, {ctx.n_modules} modules)")
    pairs = check(meds, ctx, lib)
    for d1, d2, f in pairs:
        lv = level_of(f)
        if f is None or lv == "independent":
            continue
        print(f"  {d1.name} + {d2.name}: {lv}; {d1.name}->{f['t1']}, {d2.name}->{f['t2']} "
              f"(coupling {f['prox']:.2f}); direction: {f['direction']}")
    # pharmacokinetic flags (CYP / P-gp roles mined from FDA labels; needs data/fda_label_interactions.json)
    from .pk_layer import load_roles, pk_pair
    roles = load_roles({d.name.lower() for d in lib.values()})
    if roles:
        names = [m.lower() for m in meds]
        hits = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                for e, perp, vic, kind in pk_pair(roles, names[i], names[j]):
                    hits.append(f"{perp} ({kind}s {e.upper()}) -> {vic} (substrate)")
        if hits:
            print("  PK flags (enzyme-mediated; vs DDInter: ~1.5x lift for any listed interaction, ~2x for Major, "
                  "recall ~15-18%): " + "; ".join(hits))
        else:
            print("  PK flags: none found (roles known for "
                  f"{sum(1 for n in names if n in roles)}/{len(names)} of these drugs)")
    quiet = [f"{a.name}+{b.name}" for a, b, f in pairs if f is not None and level_of(f) == "independent"]
    if quiet:
        print(f"  independent (different modules, weak coupling): {', '.join(quiet)}")


# ---------------------------------------------------------------- validation

KNOWN_PD = [  # documented pharmacodynamic interactions or deliberate synergistic regimens, DIFFERENT targets
    ("lisinopril", "losartan"),        # dual RAAS blockade: hyperkalaemia, AKI
    ("lisinopril", "spironolactone"),  # RAAS + MR antagonist: hyperkalaemia
    ("losartan", "spironolactone"),
    ("dabrafenib", "trametinib"),      # BRAF + MEK: approved synergistic combination
    ("alpelisib", "fulvestrant"),      # PI3K + ER: approved combination
    ("alpelisib", "capivasertib"),     # same PI3K/AKT axis
    ("palbociclib", "fulvestrant"),    # CDK4/6 + ER: approved combination
    ("trastuzumab", "pertuzumab"),     # same-family ERBB2/ERBB2: dual HER2 blockade (same target -> excluded below)
    ("aspirin", "rivaroxaban"),        # bleeding: platelet COX vs factor Xa - expected MISS (different pathways)
    ("fluoxetine", "olanzapine"),      # serotonergic + dopaminergic/5HT2A antipsychotic
    ("methotrexate", "tofacitinib"),   # immunosuppression stack
]


def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def validate(ctx, lib, n_neg=3000, seed=0):
    rng = np.random.default_rng(seed)
    pos, skipped = [], []
    for a, b in KNOWN_PD:
        if a not in lib or b not in lib:
            skipped.append((a, b, "drug not in library"))
            continue
        f = pair_flag(ctx, lib[a], lib[b])
        if f is None or f["same_target"]:
            skipped.append((a, b, "target not on network" if f is None else "same target (trivial)"))
            continue
        pos.append((a, b, f))
    names = list(lib)
    known = {frozenset(p) for p in KNOWN_PD}
    neg = []
    while len(neg) < n_neg:
        a, b = rng.choice(len(names), 2, replace=False)
        if frozenset((names[a], names[b])) in known:
            continue
        f = pair_flag(ctx, lib[names[a]], lib[names[b]])
        if f is not None and not f["same_target"]:
            neg.append(f["score"])
    print(f"known interactions evaluable: {len(pos)} of {len(KNOWN_PD)}  "
          f"(skipped: {', '.join(f'{a}+{b} [{why}]' for a, b, why in skipped) or 'none'})")
    for a, b, f in pos:
        print(f"  {a}+{b}: score {f['score']:.2f} ({level_of(f)})  "
              f"percentile vs random pairs: {(np.array(neg) < f['score']).mean()*100:.0f}")
    print(f"AUC (known vs {len(neg)} random library pairs): {auc([f['score'] for _, _, f in pos], neg):.2f}   "
          f"[n_positive = {len(pos)}: a handful, treat as a sanity check, not a benchmark]")
    neg = np.array(neg)
    print(f"false-flag base rate on random library pairs: {np.mean(neg >= 0.95)*100:.1f}% reach OVERLAP-or-higher, "
          f"{np.mean(neg >= 1.2)*100:.1f}% reach TIGHT-like scores (>=1.2); "
          f"most 'flags' on a real med list will be noise unless the pair is in the top tier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meds", nargs="*")
    ap.add_argument("--context")
    ap.add_argument("--modules", type=int, default=40)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    ctx = Context(a.context, modules=a.modules)
    lib = default_library()
    if a.validate:
        validate(ctx, lib)
    elif len(a.meds) >= 2:
        report(a.meds, ctx, lib)
    else:
        ap.error("give 2+ drug names or --validate")


if __name__ == "__main__":
    main()
