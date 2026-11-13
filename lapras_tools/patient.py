"""Tool 2 - Patient-specific Lapras.

Reweights a disease coupling matrix by what a patient's tissue expresses, then
recomputes D, the bill ranking and the combination plan.

    C'_ij = C_ij * sqrt(w_i * w_j),   w_i in [floor, 1] from the patient's expression

so an edge between two genes the tumour has switched off is down-weighted and an
edge between two highly expressed genes keeps its full STRING weight.

Input: TSV/CSV with columns gene, value (TPM/FPKM/counts - only ranks are used).
Genes must be HGNC symbols (case-insensitive); Ensembl/Entrez IDs will not match.
w_i is the gene's percentile among ALL genes in the file (use the whole
transcriptome, not just the network genes, or the percentiles are meaningless).

    python3 -m lapras_tools.patient breast_cancer patient.tsv
    python3 -m lapras_tools.patient breast_cancer --demo      # 3 SIMULATED subtypes

Significance: D is a step-like output of a greedy compiler.  Each run also draws
--null shuffled-expression controls (same expression values, assigned to random
network genes) so you can see whether the patient's D shift exceeds what random
reweighting produces.  A shift inside the null band is not a patient effect.
"""
import argparse, csv
import numpy as np
from .core import load_disease, analyze_network
from .druglib import default_library, DISEASE_KEYWORDS
from .combo_designer import design


_KEYWORDS = None      # disease indication keywords, set by main() when the ChEMBL library is loaded


def read_expression(path):
    with open(path) as f:
        sniff = f.read(2048); f.seek(0)
        delim = "\t" if sniff.count("\t") >= sniff.count(",") else ","
        rows = list(csv.reader(f, delimiter=delim))
    out = {}
    for r in rows:
        if len(r) < 2:
            continue
        try:
            out[r[0].strip().upper()] = float(r[1])
        except ValueError:
            continue                      # header
    return out


def weights_from_expression(expr, names, floor=0.05):
    """Percentile rank of each network gene within the whole expression table."""
    vals = np.array(sorted(expr.values()))
    w = np.full(len(names), np.nan)
    for i, n in enumerate(names):
        if n in expr:
            w[i] = np.searchsorted(vals, expr[n], side="right") / len(vals)
    missing = np.isnan(w)
    w[missing] = np.nanmedian(w) if (~missing).any() else 0.5
    return np.clip(w, floor, 1.0), [names[i] for i in np.where(missing)[0]]


def weights_relative(expr, names, ref_genes, ref_tpm, floor=0.05, exclude=None):
    """Cohort-relative weights: for each network gene, the patient's percentile among the
    REFERENCE patients (mid-rank, log scale).  ref_tpm is samples x genes aligned with
    ref_genes; `exclude` drops one reference row (leave-one-out when the patient is in it).

    Why: a whole-transcriptome percentile (weights_from_expression) puts nearly every gene
    in a narrow band, so patients barely differ (TCGA-BRCA top-10 bill sets overlap 0.91
    across ER subtypes).  Ranking against a reference cohort asks the useful question:
    is this gene high or low FOR A TUMOUR LIKE THIS?"""
    gi = {g: i for i, g in enumerate(ref_genes)}
    keep = np.ones(ref_tpm.shape[0], bool)
    if exclude is not None:
        keep[exclude] = False
    w = np.full(len(names), np.nan)
    for k, n in enumerate(names):
        if n in expr and n in gi:
            col = np.log2(ref_tpm[keep, gi[n]] + 1)
            v = np.log2(expr[n] + 1)
            w[k] = ((col < v).sum() + 0.5 * (col == v).sum()) / len(col)
    missing = [names[i] for i in np.where(np.isnan(w))[0]]
    w[np.isnan(w)] = 0.5
    return np.clip(w, floor, 1.0), missing


def load_reference(spec):
    """'tcga-brca' -> cached TCGA-BRCA table; otherwise a path to an .npz with genes + tpm."""
    import os
    path = (os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tcga_brca_tpm.npz")
            if spec == "tcga-brca" else spec)
    if not os.path.exists(path):
        raise SystemExit(f"reference cohort not found: {path}"
                         + ("  (run: python3 -m lapras_tools.validate_tcga --fetch 260)" if spec == "tcga-brca" else ""))
    z = np.load(path)
    return list(z["genes"]), z["tpm"].astype(float)


def reweight(C, w):
    s = np.sqrt(w)
    return C * np.outer(s, s)


def null_D(C, names, w, n=30, seed=0):
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        ds.append(analyze_network(reweight(C, rng.permutation(w)), names).D)
    return np.array(ds)


def personalize(C, names, w, n_null=200, lib=None, modules=5, min_expr=0.25):
    base = analyze_network(C, names)
    pat = analyze_network(reweight(C, w), names)
    nulls = null_D(C, names, w, n_null) if n_null else np.array([])
    rb = {names[i]: r for r, i in enumerate(base.ranked())}
    rp = {names[i]: r for r, i in enumerate(pat.ranked())}
    moves = sorted(names, key=lambda g: rp[g] - rb[g])
    res = dict(base=base, pat=pat, nulls=nulls, rank_base=rb, rank_pat=rp,
               risers=[(g, rb[g] + 1, rp[g] + 1) for g in moves[:6]],
               fallers=[(g, rb[g] + 1, rp[g] + 1) for g in moves[::-1][:6]])
    if lib is not None:
        # never propose a drug whose target this tumour barely expresses (e.g. an ER antagonist
        # for an ER-negative tumour): require expression weight >= min_expr
        res["plan"] = design(pat, lib, modules, keywords=_KEYWORDS, eligible=(np.asarray(w) >= min_expr))
    return res


def report(label, res, missing=()):
    b, p, nl = res["base"], res["pat"], res["nulls"]
    print(f"\n{label}: generic D = {b.D:.2f} ({b.verdict})   patient D = {p.D:.2f} ({p.verdict})")
    if len(nl):
        lo, hi = np.percentile(nl, [2.5, 97.5])
        # two-sided empirical p-value: how often a random reweighting lands at least this far from the null median
        med = np.median(nl)
        pval = (1 + np.sum(np.abs(nl - med) >= abs(p.D - med))) / (len(nl) + 1)
        print(f"  null (shuffled expression, n={len(nl)}): D median {med:.2f}, "
              f"95% band [{lo:.2f}, {hi:.2f}]; patient D empirical p = {pval:.2f} -> "
              f"{'distinguishable from random reweighting' if pval < 0.05 else 'NOT distinguishable from random reweighting'}")
        if len(nl) < 100:
            print(f"  (only {len(nl)} null draws: p cannot be below {1/(len(nl)+1):.3f}; use --null 200 for a real test)")
    if missing:
        frac = len(missing) / len(b.names)
        print(f"  {len(missing)}/{len(b.names)} network genes absent from the expression file (given the median weight)")
        if frac > 0.3:
            print(f"  WARNING: {frac*100:.0f}% of network genes missing - are the IDs HGNC symbols? "
                  f"(Ensembl/Entrez IDs will not match); this result is not meaningful")
    print("  top-10 by bill, generic -> patient:")
    top = [b.names[i] for i in b.ranked()[:10]]
    print("   generic:", ", ".join(top))
    print("   patient:", ", ".join(p.names[i] for i in p.ranked()[:10]))
    print("  biggest risers :", ", ".join(f"{g} {a}->{c}" for g, a, c in res["risers"]))
    print("  biggest fallers:", ", ".join(f"{g} {a}->{c}" for g, a, c in res["fallers"]))
    if "plan" in res:
        print("  patient combination:",
              "; ".join(f"{x['target']}:{x['drug'].name}" for x in res["plan"]["picks"]),
              "| modules with no eligible drug (undrugged, or target not expressed):", ", ".join(g["top_target"] for g in res["plan"]["gaps"]) or "none")


# ---------------------------------------------------------------- demo data

_SUBTYPES = {   # SIMULATED expression percentiles for named genes (everything else ~ U(0,1))
    "ER+ luminal (simulated)": dict(ESR1=.97, PGR=.9, GATA3=.97, FOXA1=.97, CCND1=.9, ERBB2=.4,
                                    EGFR=.3, MYC=.5, BRCA1=.4, TP53=.4, CDK4=.7, CDK6=.7),
    "HER2+ (simulated)":       dict(ERBB2=.99, ESR1=.3, PGR=.25, GATA3=.6, EGFR=.6, MYC=.7,
                                    PIK3CA=.7, TP53=.7, CDK12=.9),
    "triple-negative (simulated)": dict(ESR1=.05, PGR=.05, ERBB2=.2, GATA3=.1, FOXA1=.1,
                                        EGFR=.9, MYC=.95, BRCA1=.9, TP53=.95, RAD51=.95,
                                        CDK4=.6, PIK3CA=.4),
}


def simulated_expression(genes, subtype, seed=0, universe=20000):
    rng = np.random.default_rng(seed)
    expr = {f"BG{i}": float(rng.random()) for i in range(universe)}   # background transcriptome
    for g in genes:
        expr[g] = float(rng.random())
    for g, v in _SUBTYPES[subtype].items():
        expr[g] = float(np.clip(v + rng.normal(0, 0.03), 0, 1))
    return expr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("disease")
    ap.add_argument("expression", nargs="?")
    ap.add_argument("--demo", action="store_true", help="3 simulated breast-cancer subtypes")
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--reference", metavar="tcga-brca|PATH.npz",
                    help="weight each gene by the patient's percentile among a reference cohort "
                         "(recommended; whole-transcriptome percentiles barely differentiate patients). "
                         "npz needs 'genes' and 'tpm' (samples x genes).")
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--min-expression", type=float, default=0.25,
                    help="targets with expression weight below this are not proposed")
    ap.add_argument("--score", type=float, default=0.9)
    a = ap.parse_args()
    C, names, _, label = load_disease(a.disease, a.score)
    lib = default_library()
    global _KEYWORDS
    _KEYWORDS = DISEASE_KEYWORDS.get(a.disease) if any(d.verified for d in lib.values()) else None
    if a.demo:
        if a.disease != "breast_cancer":
            ap.error("--demo subtypes are defined for breast_cancer only")
        for st in _SUBTYPES:
            w, miss = weights_from_expression(simulated_expression(names, st), names, a.floor)
            report(f"{label} - {st}", personalize(C, names, w, n_null=a.null, lib=lib,
                                                  min_expr=a.min_expression), miss)
        return
    if not a.expression:
        ap.error("give an expression file or --demo")
    expr = read_expression(a.expression)
    if a.reference:
        rg, rt = load_reference(a.reference)
        w, miss = weights_relative(expr, names, rg, rt, a.floor)
        print(f"weights: percentile among {rt.shape[0]} reference samples ({a.reference})")
    else:
        w, miss = weights_from_expression(expr, names, a.floor)
        print("weights: percentile within this patient's transcriptome (no --reference: patients "
              "will look alike; a reference cohort is recommended)")
    report(label, personalize(C, names, w, n_null=a.null, lib=lib, min_expr=a.min_expression), miss)


if __name__ == "__main__":
    main()
