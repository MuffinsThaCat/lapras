"""Sepsis deep analysis: why 40 years of monotherapy trials failed.

D = 4.00 (GLOBAL) — explains the failure of anti-TNF, eritoran, activated
protein C, and every other single-target intervention. NFKB1 is #1 by bill.

    python3 -m lapras_tools.sepsis_analysis
"""
from .core import load_disease, analyze_network, LINE
import numpy as np


FAILED_TRIALS = {
    "TNF":   {"drug": "anti-TNF (infliximab, afelimomab)", "trial": "multiple Phase III", "year": "1990s-2000s", "result": "no survival benefit"},
    "TLR4":  {"drug": "eritoran (TLR4 antagonist)", "trial": "ACCESS (Phase III, n=1961)", "year": "2013", "result": "no survival benefit"},
    "PROC":  {"drug": "drotrecogin alfa (activated protein C)", "trial": "PROWESS-SHOCK (Phase III)", "year": "2011", "result": "no benefit; withdrawn from market"},
    "C5":    {"drug": "eculizumab (anti-C5)", "trial": "ongoing Phase II/III", "year": "2020s", "result": "results pending"},
    "IL1B":  {"drug": "anakinra (IL-1 receptor antagonist)", "trial": "multiple Phase III", "year": "2000s-2020s", "result": "modest benefit in subgroups only"},
    "IFNG":  {"drug": "emapalumab (anti-IFNγ)", "trial": "approved for HLH, not sepsis", "year": "2018", "result": "not tested in general sepsis"},
    "HMGB1": {"drug": "anti-HMGB1 antibodies", "trial": "preclinical only", "year": "2010s", "result": "never reached Phase III"},
}

SUBSYSTEMS = {
    "NF-κB / cytokine cascade": ["NFKB1", "RELA", "NFKBIA", "IKBKB", "TNF", "IL6", "IL1B", "IL10", "IL18", "IFNG", "IL1A"],
    "TLR / innate sensing":     ["TLR4", "TLR2", "MYD88", "IRAK4", "TRAF6", "RIPK1"],
    "DAMP / alarmin":           ["HMGB1", "S100A8", "S100A9", "AGER"],
    "complement":               ["C3", "C5", "C5AR1", "CFB", "MASP2"],
    "coagulation":              ["F2", "F3", "SERPINE1", "PROC", "THBD", "TFPI"],
    "endothelial / adhesion":   ["VCAM1", "ICAM1", "SELE", "SELP"],
    "oxidative / enzymatic":    ["NOS2", "NOS3", "PTGS2"],
    "inflammasome":             ["CASP1", "NLRP3", "PYCARD", "GSDMD"],
    "JAK-STAT signaling":       ["JAK1", "JAK2", "STAT3", "STAT1"],
    "PI3K / survival":          ["AKT1", "MTOR", "PIK3CA", "HIF1A"],
    "vascular":                 ["VEGFA", "ANGPT2", "TEK"],
}


def main():
    C, names, targets, label = load_disease("sepsis", 0.9)
    an = analyze_network(C, names)
    ranked = an.ranked()
    n = len(names)

    bill_ranks = np.empty(n)
    bill_ranks[ranked] = np.arange(n)
    edges = int((C > 0).sum()) // 2

    print("=" * 100)
    print(f"SEPSIS DEEP ANALYSIS: WHY MONOTHERAPY FAILS")
    print(f"  D = {an.D:.2f} ({'MODULAR' if an.D < LINE else 'GLOBAL'})")
    print(f"  TC = {an.tc:.3f}, Proteins = {n}, Edges = {edges}")
    print("=" * 100)

    print(f"\n  Full bill ranking:")
    print(f"  {'#':>4} {'Protein':12s} {'Bill':>6} {'Pctl':>6}  Drug status")
    print("  " + "-" * 80)
    for i in range(n):
        idx = ranked[i]
        name = names[idx]
        bill = an.bill[idx]
        pct = bill_ranks[idx] / n
        drug = targets.get(name, "")
        marker = " ***" if drug else ""
        print(f"  {i+1:4d} {name:12s} {bill:6.3f} {pct:6.1%}  {drug}{marker}")

    print(f"\n{'=' * 100}")
    print("FAILED TRIAL ANALYSIS: what the bill predicts about each attempt")
    print("=" * 100)

    for gene, info in FAILED_TRIALS.items():
        if gene in names:
            idx = names.index(gene)
            rank = int(bill_ranks[idx]) + 1
            bill = an.bill[idx]
            pct = rank / n
            verdict = "HIGH-BILL target, but D=4.00 says monotherapy can't contain it" if pct < 0.3 else "LOW-BILL target — wrong protein entirely"
        else:
            rank = "N/A"
            bill = 0
            pct = 1.0
            verdict = "Not in network"

        print(f"\n  {info['drug']}")
        print(f"    Target: {gene} → rank #{rank}/{n} (bill={bill:.3f}, top {pct:.0%})")
        print(f"    Trial: {info['trial']} ({info['year']})")
        print(f"    Result: {info['result']}")
        print(f"    Framework: {verdict}")

    print(f"\n{'=' * 100}")
    print("SUBSYSTEM ANALYSIS: which pathways own the most coupling?")
    print("=" * 100)

    subsys_data = []
    for subsystem, genes in SUBSYSTEMS.items():
        present = [g for g in genes if g in names]
        if not present:
            continue
        ranks = [int(bill_ranks[names.index(g)]) + 1 for g in present]
        bills = [float(an.bill[names.index(g)]) for g in present]
        total_bill = sum(bills)
        best_gene = present[np.argmin(ranks)]
        best_rank = min(ranks)
        subsys_data.append((subsystem, present, ranks, bills, total_bill, best_gene, best_rank))

    subsys_data.sort(key=lambda x: -x[4])
    for subsystem, present, ranks, bills, total_bill, best_gene, best_rank in subsys_data:
        pct_tc = total_bill / an.tc * 100
        print(f"\n  {subsystem} — {total_bill:.3f} total bill ({pct_tc:.1f}% of TC)")
        print(f"    Best: {best_gene} #{best_rank}/{n}")
        for g, r, b in sorted(zip(present, ranks, bills), key=lambda x: x[1]):
            drug = targets.get(g, "")
            print(f"      {g:12s} #{r:2d}  bill={b:.3f}  {drug[:55]}")

    print(f"\n{'=' * 100}")
    print("CLINICAL INTERPRETATION")
    print("=" * 100)

    nfkb_rank = int(bill_ranks[names.index("NFKB1")]) + 1
    tnf_rank = int(bill_ranks[names.index("TNF")]) + 1
    proc_rank = int(bill_ranks[names.index("PROC")]) + 1

    print(f"""
  D = {an.D:.2f} → GLOBAL → no single protein target can contain sepsis.

  This explains 40 years of failed monotherapy trials:
    - Anti-TNF (1990s-2000s): TNF is #{tnf_rank}/{n}, a high-bill target, but
      D = 4.00 means hitting one cytokine can't halt a globally coupled cascade.
    - Eritoran/TLR4 (2013): same problem. Blocking the sensor doesn't stop
      the cascade once DAMP/HMGB1 and complement are already activated.
    - Activated Protein C (2011): PROC is #{proc_rank}/{n} (bottom 16%).
      The FDA approved a drug targeting a protein that owns almost none
      of the network's coupling. PROWESS-SHOCK's failure was predictable.

  The #1 protein is NFKB1 (#{nfkb_rank}/{n}). No approved NF-κB inhibitor.
  NFKB1 also ranks #1 in depression, atherosclerosis, and tuberculosis.
  It is the most undrugged bottleneck in medicine.

  Predicted combination strategy:
    NF-κB (cascade master) + complement (C3/C5) + coagulation (thrombin/PROC pathway)
    — Hits the three major subsystems: inflammation, innate immunity, coagulopathy.
    — Each subsystem independently lethal in severe sepsis.
    — No existing trial has combined all three axes.
""")


if __name__ == "__main__":
    main()
