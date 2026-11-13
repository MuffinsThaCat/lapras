"""Retrospective test of the D = 3.03 line against standard of care.

Claim under test (README): below the line one drug can contain the disease, above it
combinations are needed.  Ground truth used here: does the FIRST-LINE guideline regimen for a
typical newly diagnosed patient combine >= 2 drugs of different mechanism at the outset
(label 1) or is it a single agent (label 0)?

HOW THE LABELS WERE MADE (read before trusting the result): written by Claude from general
knowledge of treatment guidelines in one pass, before any D value was compared to them, and
not adjudicated by a clinician.  Diseases whose first-line regimen is genuinely ambiguous are
EXCLUDED rather than forced (reason given).  A labelling dispute on a few diseases can move
the numbers, so the per-disease table is printed; change LABELS and rerun to test other calls.

    python3 -m lapras_tools.validate_line
"""
import numpy as np
from .core import load_disease, analyze_network, LINE
from .uncertainty import d_uncertainty

# key: (label, rationale).  label None = excluded.
LABELS = {
    "epilepsy":            (0, "antiepileptic monotherapy is first-line"),
    "depression":          (0, "single antidepressant first-line"),
    "schizophrenia":       (0, "single antipsychotic first-line"),
    "sickle_cell":         (0, "hydroxyurea single agent"),
    "obesity":             (0, "single GLP-1 class agent (+ lifestyle)"),
    "nafld":               (0, "resmetirom single agent"),
    "ipf":                 (0, "nintedanib OR pirfenidone, one agent"),
    "huntingtons":         (0, "symptomatic single agent; no disease-modifying combination"),
    "osteoporosis":        (0, "bisphosphonate single agent first-line"),
    "type2_diabetes":      (0, "metformin first-line (combinations added later)"),
    "type1_diabetes":      (0, "insulin (one drug class)"),
    "multiple_sclerosis":  (0, "single disease-modifying therapy first-line"),
    "heart_failure":       (1, "guideline-directed quadruple therapy (RASi/ARNI, beta-blocker, MRA, SGLT2i)"),
    "chronic_kidney":      (1, "RAS blocker + SGLT2 inhibitor (+ MRA) as standard"),
    "lupus":               (1, "hydroxychloroquine + glucocorticoid +/- immunosuppressant"),
    "hiv":                 (1, "three-drug antiretroviral regimens"),
    "tuberculosis":        (1, "four-drug RIPE regimen"),
    "breast_cancer":       (1, "endocrine + CDK4/6 or multi-agent chemotherapy"),
    "ovarian_cancer":      (1, "carboplatin + paclitaxel"),
    "colorectal_cancer":   (1, "FOLFOX/FOLFIRI multi-agent"),
    "melanoma":            (1, "nivolumab+ipilimumab or BRAF+MEK"),
    "prostate_cancer":     (1, "ADT + androgen-receptor pathway inhibitor"),
    "lung_cancer":         (1, "platinum doublet +/- immunotherapy"),
    "liver_cancer":        (1, "atezolizumab + bevacizumab"),
    "atherosclerosis":     (None, "statin first-line but antiplatelet/ACEi routinely added: ambiguous"),
    "parkinsons":          (None, "levodopa+carbidopa is one mechanism class: ambiguous"),
    "fibrosis_systemic":   (None, "organ-specific regimens: ambiguous"),
    "crohns":              (None, "anti-TNF alone or with immunomodulator: ambiguous"),
    "asthma":              (None, "stepwise; single combination inhaler: ambiguous"),
    "glioblastoma":        (None, "one drug (temozolomide) + radiotherapy: ambiguous"),
    "rheumatoid_arthritis": (None, "methotrexate first, combinations by treat-to-target: ambiguous"),
}
CANCERS = {"breast_cancer", "ovarian_cancer", "colorectal_cancer", "melanoma", "prostate_cancer",
           "lung_cancer", "liver_cancer", "glioblastoma"}


def auc(score, label):
    pos = [s for s, l in zip(score, label) if l == 1]
    neg = [s for s, l in zip(score, label) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def fisher(a, b, c, d):
    from scipy.stats import fisher_exact
    return fisher_exact([[a, b], [c, d]])[1]


def perm_p(score, label, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    obs = auc(score, label)
    lab = np.array(label)
    hits = sum(auc(score, rng.permutation(lab).tolist()) >= obs for _ in range(n))
    return (hits + 1) / (n + 1)


def report(rows, title):
    lab = [r["label"] for r in rows]
    D = [r["D"] for r in rows]
    n1, n0 = sum(lab), len(lab) - sum(lab)
    pred = [1 if d > LINE else 0 for d in D]
    tp = sum(p == 1 and l == 1 for p, l in zip(pred, lab)); tn = sum(p == 0 and l == 0 for p, l in zip(pred, lab))
    fp = sum(p == 1 and l == 0 for p, l in zip(pred, lab)); fn = sum(p == 0 and l == 1 for p, l in zip(pred, lab))
    print(f"\n{title}: n={len(rows)} ({n1} combination, {n0} single-agent)")
    print(f"   D > 3.03 predicts 'combination':  TP {tp}  FP {fp}  FN {fn}  TN {tn}   accuracy {(tp+tn)/len(rows):.2f}")
    print(f"   majority-class accuracy {max(n1, n0)/len(rows):.2f};  Fisher exact p = {fisher(tp, fp, fn, tn):.3f}")
    print(f"   AUC of D for 'combination' = {auc(D, lab):.2f}  (permutation p = {perm_p(D, lab):.3f})")
    miss = [(r["key"], r["D"], r["label"]) for r in rows if (r["D"] > LINE) != (r["label"] == 1)]
    print("   misclassified: " + ", ".join(f"{k} (D={d:.2f}, {'combo' if l else 'single'})" for k, d, l in miss))


def fdc_crosscheck(rows):
    """An independent, data-driven 'combination-ness' measure that does not depend on my labels:
    for each disease, the number of distinct approved fixed-dose products (Drugs@FDA, two active
    ingredients) in which BOTH ingredients have a phase-4 ChEMBL indication for the disease.
    Crude (co-formulation reflects pharma packaging as well as guidelines, and oncology
    combinations are co-administered), so read it as a second opinion, not ground truth."""
    import os
    from scipy.stats import spearmanr
    from .druglib import default_library, DISEASE_KEYWORDS
    from .validate_fdc import CACHE, pairs_from_cache
    if not os.path.exists(CACHE):
        print("\n(FDC cross-check skipped: run `python3 -m lapras_tools.validate_fdc --fetch` first)")
        return
    lib = default_library()
    pairs = [tuple(sorted(p)) for p in pairs_from_cache(lib)]
    for r in rows:
        kw = DISEASE_KEYWORDS[r["key"]]
        r["n_fdc"] = sum(1 for x, y in pairs
                         if lib[x].indicated_for(kw, 4) and lib[y].indicated_for(kw, 4))
    print("\nIndependent cross-check: approved fixed-dose products with BOTH ingredients indicated for the disease")
    for r in sorted(rows, key=lambda r: -r["n_fdc"])[:10]:
        print(f"   {r['n_fdc']:3d}  {r['name']:32s} D={r['D']:.2f}  my label: "
              f"{ {1:'combo',0:'single',None:'excluded'}[r['label']] }")
    allr = rows
    rho, p = spearmanr([r["D"] for r in allr], [r["n_fdc"] for r in allr])
    print(f"   Spearman(D, FDC count) over all 31 diseases: {rho:+.2f} (p = {p:.2f})")
    nc = [r for r in allr if not r["cancer"]]
    rho2, p2 = spearmanr([r["D"] for r in nc], [r["n_fdc"] for r in nc])
    print(f"   non-cancer only (n={len(nc)}): {rho2:+.2f} (p = {p2:.2f})")
    lab = [r for r in allr if r["label"] is not None]
    a = auc([r["n_fdc"] for r in lab], [r["label"] for r in lab])
    print(f"   agreement with MY labels: AUC of FDC count for 'combination' = {a:.2f} (n={len(lab)})")
    any_fdc = [1 if r["n_fdc"] > 0 else 0 for r in allr]
    print(f"   diseases with >= 1 such product: {sum(any_fdc)}/{len(allr)}; D > 3.03 among them: "
          f"{sum(1 for r in allr if r['n_fdc'] > 0 and r['D'] > LINE)}/{sum(any_fdc)}")


def combo_crosscheck(rows):
    """Second independent 'combination-ness' measure, this time INCLUDING co-administered regimens:
    for each disease, the number of distinct drug pairs (mined from FDA label 'in combination with'
    indications, validate_combos_large.parse, plus fixed-dose products) in which BOTH drugs have a
    phase-4 ChEMBL indication for the disease.  Covers oncology, which fixed-dose products do not."""
    import os
    from scipy.stats import spearmanr
    from .druglib import default_library, DISEASE_KEYWORDS
    from .validate_fdc import CACHE as FDC_CACHE
    from .validate_combos_large import CACHE as LARGE_CACHE
    from .pair_set import get_pairs
    if not (os.path.exists(FDC_CACHE) and os.path.exists(LARGE_CACHE)):
        print("\n(label-mined combination cross-check skipped: run validate_fdc --fetch and validate_combos_large --fetch)")
        return
    lib = default_library()
    pairs = get_pairs(lib)
    for r in rows:
        kw = DISEASE_KEYWORDS[r["key"]]
        r["n_combo"] = sum(1 for x, y in pairs if lib[x].indicated_for(kw, 4) and lib[y].indicated_for(kw, 4))
    print(f"\nSecond independent cross-check ({len(pairs)} label-mined + fixed-dose pairs): pairs with BOTH drugs approved for the disease")
    for r in sorted(rows, key=lambda r: -r["n_combo"])[:12]:
        print(f"   {r['n_combo']:3d}  {r['name']:32s} D={r['D']:.2f}  my label: "
              f"{ {1:'combo',0:'single',None:'excluded'}[r['label']] }")
    rho, p = spearmanr([r["D"] for r in rows], [r["n_combo"] for r in rows])
    print(f"   Spearman(D, pair count), all 31 diseases: {rho:+.2f} (p = {p:.3f})")
    nc = [r for r in rows if not r["cancer"]]
    rho2, p2 = spearmanr([r["D"] for r in nc], [r["n_combo"] for r in nc])
    print(f"   non-cancer only (n={len(nc)}): {rho2:+.2f} (p = {p2:.2f})")
    lab = [r for r in rows if r["label"] is not None]
    print(f"   agreement with MY labels: AUC of pair count for 'combination' = "
          f"{auc([r['n_combo'] for r in lab], [r['label'] for r in lab]):.2f} (n={len(lab)})")
    # does the count merely track how many drugs the disease has?  normalise by approved-drug count
    for r in rows:
        kw = DISEASE_KEYWORDS[r["key"]]
        r["n_drugs"] = sum(1 for d in lib.values() if d.indicated_for(kw, 4))
        r["frac"] = r["n_combo"] / max(r["n_drugs"] * (r["n_drugs"] - 1) / 2, 1)
    rho3, p3 = spearmanr([r["D"] for r in rows], [r["frac"] for r in rows])
    rho4, p4 = spearmanr([r["n_drugs"] for r in rows], [r["n_combo"] for r in rows])
    print(f"   pair count vs number of approved drugs for the disease: Spearman {rho4:+.2f} (it mostly tracks how well-drugged the disease is); "
          f"D vs the pair FRACTION (pairs / possible pairs): {rho3:+.2f} (p = {p3:.2f})")
    nc2 = [r for r in rows if not r["cancer"]]
    rho5, p5 = spearmanr([r["D"] for r in nc2], [r["frac"] for r in nc2])
    nc3 = [r for r in nc2 if r["n_drugs"] >= 10]     # fractions of tiny drug sets are noise
    rho6, p6 = spearmanr([r["D"] for r in nc3], [r["frac"] for r in nc3])
    print(f"   same fraction, non-cancer only: {rho5:+.2f} (p = {p5:.2f}); non-cancer with >= 10 approved drugs (n={len(nc3)}): "
          f"{rho6:+.2f} (p = {p6:.2f})")


def main():
    rows = []
    for key, (label, why) in LABELS.items():
        C, names, _, disp = load_disease(key)
        an = analyze_network(C, names)
        u = d_uncertainty(C, names, n=100)      # same setting as the README / batch.py --uncertainty table
        rows.append(dict(key=key, name=disp, D=an.D, tc=an.tc, edges=int((C > 0).sum() // 2), n=len(names),
                         p_global=u["p_global"], verdict=u["verdict"], label=label, why=why,
                         cancer=key in CANCERS))
    print(f"{'Disease':32} {'D':>5} {'P(global)':>9} {'label':>9}  rationale")
    for r in sorted(rows, key=lambda r: r["D"]):
        lab = {1: "COMBO", 0: "single", None: "excluded"}[r["label"]]
        print(f"{r['name'][:32]:32} {r['D']:5.2f} {r['p_global']:9.2f} {lab:>9}  {r['why']}")

    used = [r for r in rows if r["label"] is not None]
    report(used, "All labelled diseases")
    report([r for r in used if not r["cancer"]], "Non-cancer only (cancers are all 'global' AND all 'combination': an easy win)")
    report([r for r in used if r["verdict"] != "UNCERTAIN"], "Only diseases whose D verdict is robust to data noise")

    # --- how much do individual label disputes matter?
    print("\nSensitivity to my labels:")
    for name, subset in (("all labelled", used), ("non-cancer", [r for r in used if not r["cancer"]])):
        base = auc([r["D"] for r in subset], [r["label"] for r in subset])
        flips = []
        for i in range(len(subset)):
            lab_f = [r["label"] if j != i else 1 - r["label"] for j, r in enumerate(subset)]
            flips.append(auc([r["D"] for r in subset], lab_f))
        print(f"   {name:13s} AUC {base:.2f}; flipping any single label: range {min(flips):.2f}-{max(flips):.2f}")
    # ambiguous diseases included under the label that would favour D (and the one that hurts it)
    amb = [r for r in rows if r["label"] is None]
    for tag, rule in (("ambiguous labelled in D's favour", lambda r: 1 if r["D"] > LINE else 0),
                      ("ambiguous labelled against D", lambda r: 0 if r["D"] > LINE else 1)):
        ext = [dict(r, label=rule(r)) for r in amb] + used
        print(f"   {tag}: n={len(ext)}, AUC {auc([r['D'] for r in ext], [r['label'] for r in ext]):.2f}, "
              f"non-cancer AUC {auc([r['D'] for r in ext if not r['cancer']], [r['label'] for r in ext if not r['cancer']]):.2f}")

    lab = [r["label"] for r in used]
    print("\nBaselines on all labelled diseases (AUC for 'combination'):")
    print(f"   D                          {auc([r['D'] for r in used], lab):.2f}")
    print(f"   is a cancer                {auc([1.0 if r['cancer'] else 0.0 for r in used], lab):.2f}")
    print(f"   total correlation (TC)     {auc([r['tc'] for r in used], lab):.2f}")
    print(f"   edges in network           {auc([r['edges'] for r in used], lab):.2f}")
    print(f"   edge density               {auc([r['edges'] / (r['n']*(r['n']-1)/2) for r in used], lab):.2f}")
    nc = [r for r in used if not r["cancer"]]
    nl = [r["label"] for r in nc]
    print(f"   (non-cancer subset)  D {auc([r['D'] for r in nc], nl):.2f}   TC {auc([r['tc'] for r in nc], nl):.2f}   "
          f"edges {auc([r['edges'] for r in nc], nl):.2f}")
    fdc_crosscheck(rows)
    combo_crosscheck(rows)


if __name__ == "__main__":
    main()
