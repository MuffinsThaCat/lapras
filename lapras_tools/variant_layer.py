"""Tool 5 - Variant-aware network reweighting.

Takes a patient's known genetic variants (gene + effect type) and reweights
the disease coupling matrix to reflect structural consequences.  Integrates
with patient.py for expression data and combo_designer.py for drug selection.

    python3 -m lapras_tools.variant_layer breast_cancer variants.tsv
    python3 -m lapras_tools.variant_layer breast_cancer --demo brca1_lof
    python3 -m lapras_tools.variant_layer breast_cancer --demo brca1_lof --then-drug
    python3 -m lapras_tools.variant_layer breast_cancer variants.tsv --expression expr.tsv
    python3 -m lapras_tools.variant_layer --list-demos

Variant effects and their network-structural meaning:
  lof      homozygous loss-of-function   weight 0.05  NOT targetable
  het_lof  heterozygous LoF              weight 0.50  targetable (some protein remains)
  gof      gain-of-function              weight 2.00  PRIORITY target
  dn       dominant negative             weight 0.10  NOT targetable
  vus      variant of uncertain sig.     weight 1.00  targetable (flagged)

LoF/dn proteins are marked NOT ELIGIBLE for drug targeting because the protein
is broken — a drug that modulates the protein has nothing to modulate.  However,
drugs targeting PARTNERS of the broken protein (synthetic lethality) remain
relevant; the combo_designer's indirect-target mechanism handles this (e.g.
olaparib targets PARP1 but has BRCA1 as an indirect target).

GoF proteins are PRIORITY targets: the protein is hyperactive and structurally
central, so inhibiting it has the biggest effect on the network.

When both variants and expression data are available, the weights multiply:
GoF × high expression → strongly amplified coupling; LoF × any expression →
still near-zero (the broken protein dominates regardless of transcript level).
"""
import argparse, csv
import numpy as np
from .core import load_disease, analyze_network, LINE
from .patient import reweight as _reweight_base
from .druglib import default_library, DISEASE_KEYWORDS
from .combo_designer import design, report as combo_report


EFFECT_WEIGHTS = {
    "lof":     0.05,
    "het_lof": 0.50,
    "gof":     2.00,
    "dn":      0.10,
    "vus":     1.00,
}

EFFECT_TARGETABLE = {
    "lof":     False,
    "het_lof": True,
    "gof":     True,
    "dn":      False,
    "vus":     True,
}


def read_variants(path):
    """Read gene,effect[,score] from TSV/CSV.  Returns {GENE: {effect, score}}."""
    with open(path) as f:
        sniff = f.read(2048); f.seek(0)
        delim = "\t" if sniff.count("\t") >= sniff.count(",") else ","
        rows = list(csv.reader(f, delimiter=delim))
    out = {}
    for r in rows:
        if len(r) < 2:
            continue
        gene = r[0].strip().upper()
        effect = r[1].strip().lower()
        if effect not in EFFECT_WEIGHTS:
            continue
        score = float(r[2]) if len(r) > 2 and r[2].strip() else None
        out[gene] = {"effect": effect, "score": score}
    return out


def variant_weights(variants, names):
    """Produce weight vector and eligibility mask from variant calls.

    Returns (w, eligible, annotations):
      w:           float array, coupling weight per network node
      eligible:    bool array, can this protein be drug-targeted?
      annotations: list of dicts for variant-carrying genes
    """
    w = np.ones(len(names))
    eligible = np.ones(len(names), dtype=bool)
    annotations = []
    for i, name in enumerate(names):
        if name in variants:
            v = variants[name]
            w[i] = EFFECT_WEIGHTS[v["effect"]]
            eligible[i] = EFFECT_TARGETABLE[v["effect"]]
            annotations.append(dict(
                gene=name, effect=v["effect"], weight=w[i],
                targetable=eligible[i], score=v.get("score"), index=i,
            ))
    return w, eligible, annotations


def reweight(C, w):
    """C' = C * sqrt(w_i * w_j), capped at 1.0 (GoF weights can exceed 1)."""
    return np.minimum(_reweight_base(C, w), 1.0)


def combine_weights(variant_w, expr_w):
    """Multiply variant and expression weights for joint reweighting."""
    return np.asarray(variant_w) * np.asarray(expr_w)


def null_D(C, names, w, n=30, seed=0):
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        ds.append(analyze_network(reweight(C, rng.permutation(w)), names).D)
    return np.array(ds)


def personalize_with_variants(C, names, variants, lib=None, modules=5,
                               keywords=None, expr_w=None, n_null=200):
    """Full variant-aware analysis with optional expression integration."""
    vw, eligible, annot = variant_weights(variants, names)
    if expr_w is not None:
        w = combine_weights(vw, expr_w)
        eligible = eligible & (np.asarray(expr_w) >= 0.25)
    else:
        w = vw

    base = analyze_network(C, names)
    var_an = analyze_network(reweight(C, w), names)
    nulls = null_D(C, names, w, n_null) if n_null else np.array([])

    rb = {names[i]: r for r, i in enumerate(base.ranked())}
    rv = {names[i]: r for r, i in enumerate(var_an.ranked())}
    moves = sorted(names, key=lambda g: rv[g] - rb[g])

    res = dict(
        base=base, variant=var_an, weights=w, eligible=eligible,
        annotations=annot, nulls=nulls, rank_base=rb, rank_var=rv,
        risers=[(g, rb[g] + 1, rv[g] + 1) for g in moves[:6]],
        fallers=[(g, rb[g] + 1, rv[g] + 1) for g in moves[::-1][:6]],
    )
    if lib is not None:
        res["plan"] = design(var_an, lib, modules, keywords=keywords, eligible=eligible)
    return res


def report(label, res, variants):
    b, v, nl = res["base"], res["variant"], res["nulls"]
    print(f"\n{label}: generic D = {b.D:.2f} ({b.verdict})   "
          f"variant D = {v.D:.2f} ({v.verdict})")

    if len(nl):
        lo, hi = np.percentile(nl, [2.5, 97.5])
        med = np.median(nl)
        pval = (1 + np.sum(np.abs(nl - med) >= abs(v.D - med))) / (len(nl) + 1)
        print(f"  null (shuffled weights, n={len(nl)}): D median {med:.2f}, "
              f"95% band [{lo:.2f}, {hi:.2f}]; variant D empirical p = {pval:.2f}")

    print("  variant effects on network:")
    for a in res["annotations"]:
        if not a["targetable"]:
            status = "NOT TARGETABLE (protein is non-functional)"
        elif a["effect"] == "gof":
            status = "PRIORITY TARGET (hyperactive)"
        else:
            status = "targetable"
        print(f"    {a['gene']:10s} {a['effect']:8s} weight={a['weight']:.2f}  {status}")

    gof_genes = [a["gene"] for a in res["annotations"] if a["effect"] == "gof"]
    lof_genes = [a["gene"] for a in res["annotations"]
                 if a["effect"] in ("lof", "dn")]
    het_genes = [a["gene"] for a in res["annotations"]
                 if a["effect"] == "het_lof"]
    if lof_genes:
        print(f"  synthetic lethality: drugs targeting partners of "
              f"{', '.join(lof_genes)} may exploit the loss "
              f"(see indirect targets in druglib)")
    if gof_genes:
        print(f"  direct inhibition: {', '.join(gof_genes)} "
              f"(hyperactive — drug to suppress)")
    if het_genes:
        print(f"  haploinsufficient: {', '.join(het_genes)} "
              f"(reduced but present — still druggable)")

    print("  top-10 by bill, generic -> variant:")
    print("   generic:", ", ".join(b.names[i] for i in b.ranked()[:10]))
    print("   variant:", ", ".join(v.names[i] for i in v.ranked()[:10]))
    print("  biggest risers :",
          ", ".join(f"{g} {a}->{c}" for g, a, c in res["risers"]))
    print("  biggest fallers:",
          ", ".join(f"{g} {a}->{c}" for g, a, c in res["fallers"]))

    if "plan" in res:
        combo_report(label + " (variant-adjusted)", res["variant"], res["plan"])


# ---------------------------------------------------------------- demos

_DEMOS = {
    "brca1_lof": dict(
        disease="breast_cancer",
        label="BRCA1 carrier (homozygous LoF)",
        variants={"BRCA1": {"effect": "lof"}},
        notes="BRCA1 is non-functional.  PARP inhibitors (olaparib) work via "
              "synthetic lethality — they target PARP1/2, exploiting BRCA1's "
              "absence.  The DNA repair module weakens; other repair genes rise.",
    ),
    "pik3ca_gof": dict(
        disease="breast_cancer",
        label="PIK3CA activating mutation (E545K / H1047R)",
        variants={"PIK3CA": {"effect": "gof"}},
        notes="PIK3CA gain-of-function drives PI3K/AKT/mTOR.  Alpelisib (PI3K "
              "inhibitor) is first-line.  GoF amplifies PIK3CA's coupling, "
              "making it the dominant bill contributor.",
    ),
    "egfr_gof": dict(
        disease="lung_cancer",
        label="EGFR-driven NSCLC (exon 19 del / L858R)",
        variants={"EGFR": {"effect": "gof"}},
        notes="EGFR gain-of-function is the classic NSCLC driver.  Osimertinib "
              "is first-line.  GoF makes EGFR even more structurally central.",
    ),
    "tp53_pik3ca": dict(
        disease="breast_cancer",
        label="Multi-hit: TP53 LoF + PIK3CA GoF",
        variants={
            "TP53": {"effect": "lof"},
            "PIK3CA": {"effect": "gof"},
        },
        notes="TP53 loss removes the checkpoint; PIK3CA GoF drives "
              "proliferation.  TP53 is NOT targetable (broken).  Combo "
              "designer should focus on PIK3CA and other functional nodes.",
    ),
    "brca2_het": dict(
        disease="ovarian_cancer",
        label="BRCA2 heterozygous LoF carrier",
        variants={"BRCA2": {"effect": "het_lof"}},
        notes="Heterozygous BRCA2 — protein is reduced but not absent.  "
              "Still a PARP inhibitor candidate (niraparib, olaparib).  "
              "The DNA repair module is partially weakened.",
    ),
    "kras_gof": dict(
        disease="lung_cancer",
        label="KRAS G12C NSCLC",
        variants={"KRAS": {"effect": "gof"}},
        notes="KRAS G12C is targetable by sotorasib/adagrasib (covalent "
              "inhibitors).  GoF amplifies the RAS/MAPK axis coupling.",
    ),
    "braf_gof_tp53_lof": dict(
        disease="melanoma",
        label="BRAF V600E + TP53 LoF melanoma",
        variants={
            "BRAF": {"effect": "gof"},
            "TP53": {"effect": "lof"},
        },
        notes="BRAF V600E is the dominant driver (dabrafenib + trametinib).  "
              "TP53 loss is common but not directly druggable.  The network "
              "should show BRAF/MAPK as the priority axis.",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("disease", nargs="?")
    ap.add_argument("variants", nargs="?", help="TSV/CSV: gene,effect[,score]")
    ap.add_argument("--demo", choices=list(_DEMOS),
                    help="predefined scenario")
    ap.add_argument("--list-demos", action="store_true")
    ap.add_argument("--expression", metavar="PATH",
                    help="expression file for joint reweighting")
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--modules", type=int, default=5)
    ap.add_argument("--score", type=float, default=0.9)
    ap.add_argument("--then-drug", action="store_true",
                    help="run combo_designer on the variant-reweighted network")
    a = ap.parse_args()

    if a.list_demos:
        for k, d in _DEMOS.items():
            vlist = ", ".join(f"{g}={v['effect']}" for g, v in d["variants"].items())
            print(f"  {k:22s} {d['disease']:18s} {vlist:30s} {d['label']}")
        return

    if not a.disease:
        ap.error("give a disease key (or --list-demos)")

    if a.demo:
        demo = _DEMOS[a.demo]
        if a.disease != demo["disease"]:
            ap.error(f"--demo {a.demo} is defined for {demo['disease']}, "
                     f"not {a.disease}")
        variants = demo["variants"]
        print(f"Demo: {demo['label']}")
        print(f"  {demo['notes']}")
    elif a.variants:
        variants = read_variants(a.variants)
        if not variants:
            ap.error("no valid variants in file (need: gene,effect where "
                     f"effect is one of {list(EFFECT_WEIGHTS)})")
    else:
        ap.error("give a variants file or --demo")

    C, names, _, label = load_disease(a.disease, a.score)

    on_net = [g for g in variants if g in set(names)]
    off_net = [g for g in variants if g not in set(names)]
    if off_net:
        print(f"  variants not on network (ignored): {', '.join(off_net)}")
    if not on_net:
        print("  WARNING: no variant genes found on the network — "
              "result is identical to the generic analysis")

    expr_w = None
    if a.expression:
        from .patient import read_expression, weights_from_expression
        expr = read_expression(a.expression)
        expr_w, miss = weights_from_expression(expr, names)
        print(f"expression: {len(miss)}/{len(names)} genes missing "
              "from expression file")

    lib = default_library() if a.then_drug else None
    kw = (DISEASE_KEYWORDS.get(a.disease)
          if lib and any(d.verified for d in lib.values()) else None)

    res = personalize_with_variants(
        C, names, variants, lib=lib, modules=a.modules,
        keywords=kw, expr_w=expr_w, n_null=a.null)
    report(label, res, variants)


if __name__ == "__main__":
    main()
