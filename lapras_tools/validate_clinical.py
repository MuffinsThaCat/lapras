"""Clinical validation of the Lapras coupling framework.

Three tests plus a confound analysis.  The result is a THERAPY TAXONOMY,
not a prediction score: the framework identifies when monotherapy works
by BYPASSING the protein network vs when treatment must MODULATE it.

Tests:
  1. D vs monotherapy/combination standard of care (43 diseases)
  2. Bill rank vs approved-drug targets (43 diseases)
  3. D vs number of drugs in first-line regimen (43 diseases)

Confound analysis:
  - Edge count (raw number of STRING interactions) beats D on every test
  - Residual D after regressing out edge count: AUC = 0.48, p = 0.81
  - D adds zero predictive information beyond network size

Novel finding (the therapy taxonomy):
  - CLASS 1  Network bypass (monogenic/replacement): 15/16 use monotherapy
    regardless of D.  Therapy replaces the missing protein or supplies the
    missing product — the coupled network exists but is irrelevant.
  - CLASS 2  Network modulation (polygenic, drug targets proteins): D and
    edge count both predict mono vs combo (AUC ~0.84).
  - CLASS 3  Multi-system combination: low-D diseases where combination
    addresses different physiological systems, not different protein modules.

HOW LABELS WERE MADE: by Claude from general knowledge of treatment
guidelines, written in one pass before any D value was compared.  Not
adjudicated by a clinician.  The per-disease table is printed; change
LABELS and rerun.

    python3 -m lapras_tools.validate_clinical
"""
import numpy as np
from math import erfc, comb
from .core import load_disease, analyze_network, LINE, disease_keys, spearman
from .druglib import default_library, by_target

try:
    from scipy.stats import t as _tdist
    def _spearman_p(rho, n):
        t = rho * np.sqrt((n - 2) / (1 - rho**2 + 1e-15))
        return float(_tdist.sf(abs(t), n - 2) * 2)
except ImportError:
    def _spearman_p(rho, n):
        t = rho * np.sqrt((n - 2) / (1 - rho**2 + 1e-15))
        return float(erfc(abs(t) / np.sqrt(2)))


# ---------------------------------------------------------------- labels

LABELS = {
    "epilepsy": 0, "depression": 0, "schizophrenia": 0,
    "sickle_cell": 0, "phenylketonuria": 0, "fabry": 0,
    "gaucher": 0, "rett_syndrome": 0, "obesity": 0,
    "nafld": 0, "familial_hypercholesterolemia": 0,
    "spinal_muscular_atrophy": 0, "parkinsons": 0, "ipf": 0,
    "duchenne_md": 0, "wilson_disease": 0, "marfan": 0,
    "huntingtons": 0, "osteoporosis": 0,
    "polycystic_kidney": 0, "type1_diabetes": 0,
    "multiple_sclerosis": 0, "hemophilia_a": 0,
    "heart_failure": 1, "cystic_fibrosis": 1,
    "atherosclerosis": 1, "chronic_kidney": 1,
    "lupus": 1, "hiv": 1,
    "fibrosis_systemic": 1, "crohns": 1,
    "type2_diabetes": 1,  # validate_line.py excludes this (ambiguous); here 1 (current guidelines: early combo)
    "tuberculosis": 1,
    "asthma": 1, "rheumatoid_arthritis": 1,
    "breast_cancer": 1, "ovarian_cancer": 1,
    "glioblastoma": 1, "colorectal_cancer": 1,
    "melanoma": 1, "prostate_cancer": 1,
    "lung_cancer": 1, "liver_cancer": 1,
}

N_DRUGS = {
    "epilepsy": 1, "depression": 1, "schizophrenia": 1,
    "heart_failure": 3, "sickle_cell": 1, "phenylketonuria": 1,
    "fabry": 1, "gaucher": 1, "cystic_fibrosis": 3,
    "atherosclerosis": 3, "rett_syndrome": 1, "obesity": 1,
    "nafld": 1, "familial_hypercholesterolemia": 1,
    "chronic_kidney": 2, "spinal_muscular_atrophy": 1,
    "parkinsons": 1, "ipf": 1, "duchenne_md": 1,
    "wilson_disease": 1, "lupus": 3, "marfan": 1,
    "huntingtons": 1, "osteoporosis": 1, "hiv": 3,
    "fibrosis_systemic": 2, "crohns": 2, "hemophilia_a": 1,
    "type2_diabetes": 2, "polycystic_kidney": 1, "type1_diabetes": 1,
    "breast_cancer": 3, "tuberculosis": 4, "asthma": 2,
    "ovarian_cancer": 2, "glioblastoma": 2, "multiple_sclerosis": 1,
    "colorectal_cancer": 3, "melanoma": 2, "prostate_cancer": 2,
    "lung_cancer": 2, "liver_cancer": 2, "rheumatoid_arthritis": 2,
}

CANCERS = {"breast_cancer", "ovarian_cancer", "glioblastoma",
           "colorectal_cancer", "melanoma", "prostate_cancer",
           "lung_cancer", "liver_cancer"}

MONOGENIC = {
    "wilson_disease", "hemophilia_a", "duchenne_md", "marfan",
    "polycystic_kidney", "huntingtons", "sickle_cell",
    "phenylketonuria", "fabry", "gaucher", "cystic_fibrosis",
    "rett_syndrome", "spinal_muscular_atrophy",
    "familial_hypercholesterolemia",
}

REPLACEMENT = {
    "type1_diabetes", "parkinsons", "hemophilia_a",
    "fabry", "gaucher", "phenylketonuria",
}


# ---------------------------------------------------------------- stats

def auc_mann_whitney(group0, group1):
    """AUC = P(group0 < group1).  Returns (auc, two-sided p)."""
    n0, n1 = len(group0), len(group1)
    if n0 == 0 or n1 == 0:
        return float("nan"), float("nan")
    U = sum(1 for a in group0 for b in group1 if a < b) + \
        0.5 * sum(1 for a in group0 for b in group1 if a == b)
    auc = U / (n0 * n1)
    mu = n0 * n1 / 2
    sigma = np.sqrt(n0 * n1 * (n0 + n1 + 1) / 12)
    z = (U - mu) / sigma if sigma > 0 else 0
    p = erfc(abs(z) / np.sqrt(2))
    return auc, p


def sign_test_p(k, n):
    """One-sided binomial p: at least k successes in n trials under p=0.5."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2**n


# ---------------------------------------------------------------- loader

def load_all():
    """Load all diseases with labels.  Returns list of dicts."""
    rows = []
    for k in disease_keys():
        if k not in LABELS:
            continue
        C, names, _, label = load_disease(k, 0.9)
        an = analyze_network(C, names)
        edges = int((C > 0).sum()) // 2
        density = edges / (len(names) * (len(names) - 1) / 2)
        rows.append(dict(
            key=k, label=label, D=an.D, tc=an.tc, n=len(names),
            edges=edges, density=density, y=LABELS[k],
            cancer=k in CANCERS,
            monogenic=k in MONOGENIC,
            replacement=k in REPLACEMENT,
            bypass=k in MONOGENIC or k in REPLACEMENT,
            n_drugs=N_DRUGS.get(k, 0),
            names=names, bill=an.bill, ranked=an.ranked(),
        ))
    return rows


# ---------------------------------------------------------------- tests

def test_mono_combo(rows):
    print("=" * 72)
    print("TEST 1: D vs monotherapy / combination therapy")
    print("=" * 72)

    D = np.array([r["D"] for r in rows])
    Y = np.array([r["y"] for r in rows])
    E = np.array([r["edges"] for r in rows])

    mono_D, combo_D = D[Y == 0], D[Y == 1]
    auc_d, p_d = auc_mann_whitney(mono_D, combo_D)
    auc_e, p_e = auc_mann_whitney(E[Y == 0], E[Y == 1])

    print(f"\n  All diseases (n={len(rows)}: {len(mono_D)} mono, {len(combo_D)} combo)")
    print(f"    Mono:  D mean={mono_D.mean():.2f}, median={np.median(mono_D):.2f}")
    print(f"    Combo: D mean={combo_D.mean():.2f}, median={np.median(combo_D):.2f}")
    print(f"    D:     AUC = {auc_d:.3f}, p = {p_d:.4f}")
    print(f"    edges: AUC = {auc_e:.3f}, p = {p_e:.4f}")

    correct = sum(1 for d, y in zip(D, Y)
                  if (d > LINE and y == 1) or (d <= LINE and y == 0))
    print(f"    Threshold (D > {LINE}): {correct}/{len(rows)} = {correct/len(rows):.1%}")

    nc = [r for r in rows if not r["cancer"]]
    D_nc = np.array([r["D"] for r in nc])
    Y_nc = np.array([r["y"] for r in nc])
    E_nc = np.array([r["edges"] for r in nc])
    auc_nc, p_nc = auc_mann_whitney(D_nc[Y_nc == 0], D_nc[Y_nc == 1])
    auc_enc, p_enc = auc_mann_whitney(E_nc[Y_nc == 0], E_nc[Y_nc == 1])
    print(f"\n  Non-cancer only (n={len(nc)}: {sum(Y_nc==0)} mono, {sum(Y_nc==1)} combo)")
    print(f"    D:     AUC = {auc_nc:.3f}, p = {p_nc:.4f}")
    print(f"    edges: AUC = {auc_enc:.3f}, p = {p_enc:.4f}")

    # residual D
    X = np.column_stack([np.ones(len(E)), E])
    beta = np.linalg.lstsq(X, D, rcond=None)[0]
    D_resid = D - X @ beta
    auc_r, p_r = auc_mann_whitney(D_resid[Y == 0], D_resid[Y == 1])
    print(f"\n  Residual D (edge-count regressed out):")
    print(f"    AUC = {auc_r:.3f}, p = {p_r:.4f}")
    print(f"    -> D adds {'signal' if p_r < 0.05 else 'NO signal'} beyond edge count")

    # per-disease table
    print(f"\n  {'Disease':42s} {'D':>5} {'edges':>5} {'label':>6}")
    print("  " + "-" * 62)
    for r in sorted(rows, key=lambda r: r["D"]):
        tag = "COMBO" if r["y"] else "mono"
        flag = ""
        if (r["D"] > LINE and r["y"] == 0) or (r["D"] <= LINE and r["y"] == 1):
            flag = " <-- MISS"
        print(f"  {r['label']:42s} {r['D']:5.2f} {r['edges']:5d} {tag:>6}{flag}")


def test_bill_vs_drugs(rows):
    print(f"\n{'=' * 72}")
    print("TEST 2: bill rank vs approved-drug targets")
    print("=" * 72)

    lib = default_library()
    bt = by_target(lib, approved_only=True)

    all_drugged, all_undrugged = [], []
    n_better = 0
    n_total = 0
    per_disease = []

    for r in rows:
        names = r["names"]
        bill_ranks = np.empty(len(names))
        bill_ranks[r["ranked"]] = np.arange(len(names))

        drug_r, undrug_r = [], []
        for i, name in enumerate(names):
            norm_rank = bill_ranks[i] / len(names)
            if bt.get(name):
                drug_r.append(norm_rank)
                all_drugged.append(norm_rank)
            else:
                undrug_r.append(norm_rank)
                all_undrugged.append(norm_rank)

        if drug_r and undrug_r:
            n_total += 1
            if np.mean(drug_r) < np.mean(undrug_r):
                n_better += 1
            per_disease.append((r["label"], len(names), len(drug_r),
                                np.mean(drug_r), np.mean(undrug_r)))

    d_arr = np.array(all_drugged)
    u_arr = np.array(all_undrugged)

    rng = np.random.default_rng(42)
    n_samp = 100000
    d_s = rng.choice(d_arr, n_samp)
    u_s = rng.choice(u_arr, n_samp)
    auc_pool = np.mean(d_s < u_s) + 0.5 * np.mean(d_s == u_s)

    p_sign = sign_test_p(n_better, n_total)

    print(f"\n  Drugged genes (n={len(d_arr)}):   mean normalized rank = {d_arr.mean():.3f}")
    print(f"  Undrugged genes (n={len(u_arr)}): mean normalized rank = {u_arr.mean():.3f}")
    print(f"  (0 = top of bill ranking, 1 = bottom)")
    print(f"\n  Pooled AUC = {auc_pool:.3f}")
    print(f"  Sign test: drugged genes rank higher in {n_better}/{n_total} diseases")
    print(f"    binomial p (one-sided) = {p_sign:.4f}")


def test_dose_response(rows):
    print(f"\n{'=' * 72}")
    print("TEST 3: D vs number of drugs in first-line regimen")
    print("=" * 72)

    ds = np.array([r["D"] for r in rows if r["n_drugs"]])
    ns = np.array([r["n_drugs"] for r in rows if r["n_drugs"]])

    rho = spearman(ds, ns)
    p = _spearman_p(rho, len(ds))

    print(f"\n  Spearman rho = {rho:.3f}, p = {p:.4f}, n = {len(ds)}")
    for nd in sorted(set(ns)):
        sub = ds[ns == nd]
        print(f"    {nd} drug(s): n={len(sub):2d}  "
              f"D mean={sub.mean():.2f}, median={np.median(sub):.2f}  "
              f"[{sub.min():.2f}, {sub.max():.2f}]")


def test_confounds(rows):
    print(f"\n{'=' * 72}")
    print("CONFOUND ANALYSIS: D vs baselines")
    print("=" * 72)

    Y = np.array([r["y"] for r in rows])

    predictors = [
        ("D",             [r["D"] for r in rows]),
        ("edges",         [r["edges"] for r in rows]),
        ("edge density",  [r["density"] for r in rows]),
        ("n (proteins)",  [r["n"] for r in rows]),
        ("is cancer",     [1 if r["cancer"] else 0 for r in rows]),
        ("TC",            [r["tc"] for r in rows]),
    ]

    nc_mask = np.array([not r["cancer"] for r in rows])

    print(f"\n  {'Predictor':20s} {'AUC (all)':>10} {'p':>8} {'AUC (non-ca)':>12} {'p':>8}")
    print("  " + "-" * 62)
    for name, vals in predictors:
        v = np.array(vals, float)
        a1, p1 = auc_mann_whitney(v[Y == 0], v[Y == 1])
        v_nc = v[nc_mask]; y_nc = Y[nc_mask]
        a2, p2 = auc_mann_whitney(v_nc[y_nc == 0], v_nc[y_nc == 1])
        print(f"  {name:20s} {a1:10.3f} {p1:8.4f} {a2:12.3f} {p2:8.4f}")


def test_taxonomy(rows):
    print(f"\n{'=' * 72}")
    print("THERAPY TAXONOMY (the novel finding)")
    print("=" * 72)

    bypass = [r for r in rows if r["bypass"]]
    modulate = [r for r in rows if not r["bypass"]]

    n_bypass_mono = sum(1 for r in bypass if r["y"] == 0)
    print(f"\n  CLASS 1 — Network bypass (monogenic / replacement therapy)")
    print(f"    {n_bypass_mono}/{len(bypass)} use monotherapy")
    print(f"    D range: {min(r['D'] for r in bypass):.2f} – "
          f"{max(r['D'] for r in bypass):.2f}")
    print(f"    {'Disease':40s} {'D':>5} {'therapy':>8}  mechanism")
    print("    " + "-" * 65)
    for r in sorted(bypass, key=lambda r: r["D"]):
        tag = "combo" if r["y"] else "mono"
        mech = []
        if r["monogenic"]: mech.append("monogenic")
        if r["replacement"]: mech.append("replacement")
        print(f"    {r['label']:40s} {r['D']:5.2f} {tag:>8}  {', '.join(mech)}")

    # modulation subset
    D_mod = np.array([r["D"] for r in modulate])
    Y_mod = np.array([r["y"] for r in modulate])
    E_mod = np.array([r["edges"] for r in modulate])

    auc_d, p_d = auc_mann_whitney(D_mod[Y_mod == 0], D_mod[Y_mod == 1])
    auc_e, p_e = auc_mann_whitney(E_mod[Y_mod == 0], E_mod[Y_mod == 1])

    print(f"\n  CLASS 2 — Network modulation (polygenic, drug targets proteins)")
    print(f"    n = {len(modulate)} ({sum(Y_mod == 0)} mono, {sum(Y_mod == 1)} combo)")
    print(f"    D:     AUC = {auc_d:.3f}, p = {p_d:.4f}")
    print(f"    edges: AUC = {auc_e:.3f}, p = {p_e:.4f}")

    # the GLOBAL mono diseases that aren't bypass
    global_mono_non_bypass = [r for r in modulate
                               if r["D"] > LINE and r["y"] == 0]
    if global_mono_non_bypass:
        print(f"\n    Remaining GLOBAL-but-mono (not bypass):")
        for r in global_mono_non_bypass:
            print(f"      {r['label']:40s} D={r['D']:.2f}")
        print("    (apex-targeting: one drug hits the top of the cascade)")

    combo_modular = [r for r in modulate
                     if r["D"] <= LINE and r["y"] == 1]
    if combo_modular:
        print(f"\n  CLASS 3 — Multi-system combination (MODULAR but combo)")
        for r in combo_modular:
            print(f"    {r['label']:40s} D={r['D']:.2f}")
        print("    (combination targets different organ systems, not "
              "different protein modules)")


# ---------------------------------------------------------------- main

def main():
    rows = load_all()
    test_mono_combo(rows)
    test_bill_vs_drugs(rows)
    test_dose_response(rows)
    test_confounds(rows)
    test_taxonomy(rows)

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print("=" * 72)
    print("""
  The D = 3.03 threshold predicts monotherapy vs combination with
  AUC = 0.80 (p = 0.0008) across 43 diseases.  However, edge count
  (the raw number of STRING interactions) achieves AUC = 0.84, and
  residual D after regressing out edge count has AUC = 0.48 — D adds
  no predictive information beyond network size.

  The novel finding is a THERAPY TAXONOMY:
    1. BYPASS diseases (monogenic / replacement): 15/16 monotherapy,
       regardless of D.  The network coupling is real but irrelevant —
       therapy acts upstream.
    2. MODULATION diseases (polygenic): D predicts mono vs combo
       (AUC = 0.84), though edge count still dominates.
    3. MULTI-SYSTEM combos: low-D diseases where combination addresses
       different organ systems, not protein modules.

  This taxonomy — not the threshold accuracy — is the clinically
  actionable output of the structural analysis.
""")


if __name__ == "__main__":
    main()
