"""Tool 6 - Genetic intervention simulator.

Simulates the network-structural effect of genetic interventions by modifying
the coupling matrix and recomputing D, bill rankings, and module hierarchy.

The key question: does fixing/removing gene X change the disease from
GLOBAL (D > 3.03, needs combination) to MODULAR (D <= 3.03, one target
might suffice)?

    python3 -m lapras_tools.genetic_intervention breast_cancer --knockout ERBB2
    python3 -m lapras_tools.genetic_intervention lung_cancer --silence EGFR
    python3 -m lapras_tools.genetic_intervention breast_cancer --correct BRCA1
    python3 -m lapras_tools.genetic_intervention breast_cancer --knockout ERBB2 --then-drug
    python3 -m lapras_tools.genetic_intervention glioblastoma --sweep
    python3 -m lapras_tools.genetic_intervention breast_cancer --correct BRCA1 --knockout ERBB2

Interventions:
  --knockout GENE    Remove the protein entirely (CRISPR-KO, complete silencing)
  --silence GENE     Reduce coupling by --knockdown fraction (ASO/siRNA, default 0.8)
  --activate GENE    Boost coupling by --boost factor (CRISPRa, gene therapy, default 1.5)
  --correct GENE     Restore a LoF gene to full coupling (CRISPR HDR, gene replacement);
                     requires --variant-effect to establish the 'diseased' baseline

Combined interventions (--correct BRCA1 --knockout ERBB2) are applied left to
right: first the correction establishes the starting point; then the knockout
operates on the corrected network.  This models 'fix this gene AND silence that
one' — the kind of combination intervention that multi-modal gene therapy enables.

  --then-drug        After intervention, run combo_designer on modified network
  --sweep            Try every gene as knockout; rank by delta-D
"""
import argparse
import numpy as np
from .core import load_disease, analyze_network, LINE
from .druglib import default_library, DISEASE_KEYWORDS
from .combo_designer import design, report as combo_report


# ---------------------------------------------------------------- operations

def knockout(C, idx):
    """Zero out the protein's row and column."""
    M = C.copy()
    M[idx, :] = 0.0
    M[:, idx] = 0.0
    return M


def silence(C, idx, fraction=0.8):
    """Reduce the protein's edges by `fraction` (clamped to [0, 1])."""
    fraction = max(0.0, min(1.0, fraction))
    M = C.copy()
    M[idx, :] *= (1.0 - fraction)
    M[:, idx] *= (1.0 - fraction)
    return M


def amplify(C, idx, factor=1.5):
    """Boost the protein's edges (capped at 1.0 per edge)."""
    M = C.copy()
    M[idx, :] = np.minimum(M[idx, :] * factor, 1.0)
    M[:, idx] = np.minimum(M[:, idx] * factor, 1.0)
    return M


# ---------------------------------------------------------------- sweep

def sweep_knockout(C, names):
    """Try knocking out every gene; rank by D-change."""
    base = analyze_network(C, names)
    results = []
    for i, gene in enumerate(names):
        M = knockout(C, i)
        if not (M > 0).any():
            continue
        try:
            a = analyze_network(M, names)
            results.append(dict(
                gene=gene, D_original=base.D, D_knockout=a.D,
                delta_D=a.D - base.D,
                bill_rank=int(np.where(base.ranked() == i)[0][0]) + 1,
                verdict_original=base.verdict, verdict_knockout=a.verdict,
                crosses_line=(base.D > LINE) != (a.D > LINE),
            ))
        except (ValueError, np.linalg.LinAlgError):
            continue
    results.sort(key=lambda r: r["delta_D"])
    return results, base


def report_sweep(label, results, base):
    n = len(results)
    crosses = [r for r in results if r["crosses_line"]]
    print(f"\n{label}: knockout sweep ({n} genes)")
    print(f"  original D = {base.D:.2f} ({base.verdict})")

    if crosses:
        print(f"\n  ** {len(crosses)} gene(s) cross D={LINE} on knockout: **")
        for r in crosses:
            d = "GLOBAL->MODULAR" if r["D_original"] > LINE else "MODULAR->GLOBAL"
            print(f"    {r['gene']:10s} D {r['D_original']:.2f} -> "
                  f"{r['D_knockout']:.2f} (delta {r['delta_D']:+.2f}, "
                  f"bill rank {r['bill_rank']})  [{d}]")

    dec = [r for r in results if r["delta_D"] < -0.01]
    if dec:
        print(f"\n  knockout DECREASES D (gene sustains global coupling — "
              f"silencing/KO target):")
        for r in dec[:12]:
            flag = " ** CROSSES LINE **" if r["crosses_line"] else ""
            print(f"    {r['gene']:10s} D {r['D_original']:.2f} -> "
                  f"{r['D_knockout']:.2f} (delta {r['delta_D']:+.2f}, "
                  f"bill rank {r['bill_rank']}){flag}")

    inc = sorted([r for r in results if r["delta_D"] > 0.01],
                 key=lambda r: -r["delta_D"])
    if inc:
        print(f"\n  knockout INCREASES D (gene was stabilizing — "
              f"correction/restoration target):")
        for r in inc[:12]:
            flag = " ** CROSSES LINE **" if r["crosses_line"] else ""
            print(f"    {r['gene']:10s} D {r['D_original']:.2f} -> "
                  f"{r['D_knockout']:.2f} (delta {r['delta_D']:+.2f}, "
                  f"bill rank {r['bill_rank']}){flag}")

    insens = [r for r in results if abs(r["delta_D"]) <= 0.01]
    print(f"\n  summary: {len(dec)} decrease D, {len(inc)} increase D, "
          f"{len(insens)} insensitive (|delta| <= 0.01)")

    if base.verdict == "GLOBAL" and not any(
            r["D_knockout"] <= LINE for r in results):
        print(f"  no single knockout brings D below {LINE}: gene therapy "
              f"alone is structurally insufficient — combination needed")
    elif base.verdict == "GLOBAL" and crosses:
        ko = [r["gene"] for r in crosses
              if r["D_knockout"] <= LINE]
        print(f"  ** single-gene targets whose KO would make the disease "
              f"MODULAR: {', '.join(ko)} **")

    print(f"\n  interpretation:")
    print(f"    'decreases D' = this gene holds the network together; "
          f"silencing it cools the disease")
    print(f"    'increases D' = this gene was buffering; losing it makes "
          f"remaining coupling more entangled")
    print(f"    knockout sweep == correction sweep in reverse: if KO of X "
          f"drops D by 0.3, then correcting a LoF in X raises D by ~0.3")


# ---------------------------------------------------------------- report

def report_intervention(label, C_base, names, C_modified, interventions_desc,
                        lib=None, modules=5, keywords=None, eligible=None):
    """Compare base and modified networks, optionally run combo_designer."""
    base = analyze_network(C_base, names)
    if not (C_modified > 0).any():
        print(f"\n{label}: {interventions_desc}")
        print(f"  before: D = {base.D:.2f} ({base.verdict})")
        print("  after:  network collapsed (no edges remaining)")
        return

    try:
        after = analyze_network(C_modified, names)
    except (ValueError, np.linalg.LinAlgError):
        print(f"\n{label}: {interventions_desc}")
        print(f"  before: D = {base.D:.2f} ({base.verdict})")
        print("  after:  analysis failed (degenerate network)")
        return

    delta = after.D - base.D
    print(f"\n{label}: {interventions_desc}")
    print(f"  before: D = {base.D:.2f} ({base.verdict})")
    print(f"  after:  D = {after.D:.2f} ({after.verdict})   "
          f"delta = {delta:+.2f}")

    if base.D > LINE and after.D <= LINE:
        print(f"  ** CROSSES LINE: disease goes from GLOBAL to MODULAR **")
        print(f"     implication: this intervention alone may be "
              f"structurally sufficient")
    elif base.D <= LINE and after.D > LINE:
        print(f"  ** WARNING: intervention makes disease MORE global "
              f"(D increased past {LINE}) **")
    elif base.D > LINE and after.D > LINE:
        print(f"  disease remains GLOBAL — combination still needed")
    elif base.D <= LINE and after.D <= LINE:
        if delta < -0.01:
            print(f"  disease stays MODULAR, coupling further reduced")
        else:
            print(f"  disease stays MODULAR")

    print("  top-10 by bill, before -> after:")
    print("   before:", ", ".join(base.names[i] for i in base.ranked()[:10]))
    print("   after: ", ", ".join(after.names[i] for i in after.ranked()[:10]))

    if lib is not None:
        plan = design(after, lib, modules, keywords=keywords, eligible=eligible)
        combo_report(f"{label} (after {interventions_desc})", after, plan)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("disease")
    ap.add_argument("--knockout", metavar="GENE",
                    help="remove protein (CRISPR-KO, full silencing)")
    ap.add_argument("--silence", metavar="GENE",
                    help="partial knockdown (ASO/siRNA)")
    ap.add_argument("--correct", metavar="GENE",
                    help="restore a LoF gene (CRISPR HDR, gene therapy)")
    ap.add_argument("--activate", metavar="GENE",
                    help="boost a gene (CRISPRa, overexpression)")
    ap.add_argument("--knockdown", type=float, default=0.8,
                    help="fraction for --silence (default 0.8)")
    ap.add_argument("--boost", type=float, default=1.5,
                    help="factor for --activate (default 1.5)")
    ap.add_argument("--variant-effect", default="lof",
                    choices=["lof", "het_lof", "dn"],
                    help="variant type to correct from (default lof)")
    ap.add_argument("--variants", metavar="PATH",
                    help="variant file (gene,effect TSV) for full multi-variant "
                         "baseline — the diseased state reflects ALL variants, "
                         "not just the corrected gene")
    ap.add_argument("--then-drug", action="store_true",
                    help="run combo_designer on the modified network")
    ap.add_argument("--modules", type=int, default=5)
    ap.add_argument("--sweep", action="store_true",
                    help="try every gene as knockout, rank by delta-D")
    ap.add_argument("--score", type=float, default=0.9)
    a = ap.parse_args()

    C, names, _, label = load_disease(a.disease, a.score)
    idx_map = {n: i for i, n in enumerate(names)}

    if a.sweep:
        results, base = sweep_knockout(C, names)
        report_sweep(label, results, base)
        return

    has_intervention = any([a.knockout, a.silence, a.correct, a.activate])
    if not has_intervention:
        ap.error("give --knockout, --silence, --correct, --activate, "
                 "or --sweep")

    # validate gene names
    for gene, flag in [(a.knockout, "--knockout"), (a.silence, "--silence"),
                       (a.correct, "--correct"), (a.activate, "--activate")]:
        if gene and gene not in idx_map:
            ap.error(f"{gene} not in {a.disease} network "
                     f"({len(names)} proteins); check the gene symbol")

    # build base weights (the patient's current state)
    from .variant_layer import EFFECT_WEIGHTS, read_variants, variant_weights, reweight as vl_reweight
    if a.variants:
        variants = read_variants(a.variants)
        if not variants:
            ap.error("no valid variants in file")
        vw, elig_v, annot = variant_weights(variants, names)
        on_net = [g for g in variants if g in set(names)]
        off_net = [g for g in variants if g not in set(names)]
        if off_net:
            print(f"  variants not on network (ignored): {', '.join(off_net)}")
        print(f"  variant baseline: {len(on_net)} variant(s) on network")
        for a_ in annot:
            print(f"    {a_['gene']:10s} {a_['effect']:8s} weight={a_['weight']:.2f}")
    else:
        vw = np.ones(len(names))
        elig_v = np.ones(len(names), dtype=bool)
        if a.correct:
            vw[idx_map[a.correct]] = EFFECT_WEIGHTS[a.variant_effect]

    # C_base = what the patient has now (all variants applied)
    C_base = vl_reweight(C, vw) if not np.all(vw == 1.0) else C.copy()

    # build C_modified: correct the target gene, keep other variants
    w_mod = vw.copy()
    if a.correct:
        w_mod[idx_map[a.correct]] = 1.0
    C_mod = vl_reweight(C, w_mod) if not np.all(w_mod == 1.0) else C.copy()

    desc_parts = []
    eligible = elig_v.copy()

    if a.correct:
        variant_src = f"variant file {a.variants}" if a.variants else a.variant_effect
        desc_parts.append(f"correct({a.correct}, from {variant_src})")
    if a.knockout:
        C_mod = knockout(C_mod, idx_map[a.knockout])
        eligible[idx_map[a.knockout]] = False
        desc_parts.append(f"knockout({a.knockout})")
    if a.silence:
        C_mod = silence(C_mod, idx_map[a.silence], a.knockdown)
        desc_parts.append(f"silence({a.silence}, {a.knockdown:.0%})")
    if a.activate:
        C_mod = amplify(C_mod, idx_map[a.activate], a.boost)
        desc_parts.append(f"activate({a.activate}, {a.boost:.1f}x)")

    desc = " + ".join(desc_parts)
    lib = default_library() if a.then_drug else None
    kw = (DISEASE_KEYWORDS.get(a.disease)
          if lib and any(d.verified for d in lib.values()) else None)

    report_intervention(label, C_base, names, C_mod, desc,
                        lib=lib, modules=a.modules, keywords=kw,
                        eligible=eligible if a.then_drug else None)


if __name__ == "__main__":
    main()
