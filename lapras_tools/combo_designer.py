"""Tool 1 - Combination Designer.

For a disease: cut the module hierarchy at a coarse level, and in each module
pick the highest-bill protein that has a drug in the library.  One drug per
module; a multi-target drug that already covers a module through another of its
targets is not added again.

    python3 -m lapras_tools.combo_designer glioblastoma
    python3 -m lapras_tools.combo_designer breast_cancer --modules 4 --approved-only
    python3 -m lapras_tools.combo_designer --all        # every disease, one line each

Why --modules: the Lapras leaf modules are <=3 proteins (15-20 of them); no
regimen can hit 18 modules.  The coarser levels of the compiler hierarchy
(10, 5, 3, 2 groups) are the realistic unit; default 5.
"""
import argparse
import numpy as np
from .core import load_disease, analyze_network, disease_keys, LINE
from .druglib import default_library, by_target, DISEASE_KEYWORDS


def design(analysis, lib, max_modules=5, approved_only=False, keywords=None, indicated_only=False,
           min_phase=3, eligible=None):
    """keywords: lower-case MeSH substrings for the disease (druglib.DISEASE_KEYWORDS).  When
    given, a module's pick prefers a protein that has a drug ChEMBL lists as indicated for the
    disease (phase >= 3); only if none exists does it fall back to a mechanism-only drug,
    flagged as such.  indicated_only=True forbids the fallback.
    eligible: optional boolean vector over analysis.names; only True proteins may be picked
    (the patient tool passes 'target is expressed in this tumour')."""
    if indicated_only and not keywords:
        raise ValueError("indicated_only needs disease keywords AND a library with ChEMBL indications "
                         "(run build_druglib); without them every module would silently come back empty")
    names = analysis.names
    idx = {n: i for i, n in enumerate(names)}
    part = analysis.level_with_at_most(max_modules)
    lab = analysis.label_vector(part)
    bt = by_target(lib, approved_only)
    bill = analysis.bill

    # modules by total bill mass (positive part), biggest first
    mass = [sum(max(bill[i], 0) for i in grp) for grp in part]
    order = np.argsort(mass)[::-1]

    covered = set()      # module ids already hit
    picks, gaps = [], []
    total_mass = sum(mass) or 1.0
    for m in order:
        if m in covered:
            continue
        grp = sorted(part[m], key=lambda i: -bill[i])
        choice = None

        def clash(d):          # drug also lands in a module we already hit
            return any(t in idx and int(lab[idx[t]]) in covered for t in d.targets)

        passes = [True] if indicated_only else ([True, False] if keywords else [None])
        for want in passes:
            for i in grp:
                if eligible is not None and not eligible[i]:
                    continue                       # target not expressed in this patient
                drugs = bt.get(names[i]) or []
                if want:
                    drugs = [d for d in drugs if d.indicated_for(keywords, min_phase)]
                if drugs:
                    clean = [d for d in drugs if not clash(d)]
                    pool = clean or drugs
                    if keywords:      # among equally eligible drugs prefer the broadest approved evidence (stable sort)
                        pool = sorted(pool, key=lambda d: -d.evidence_count(keywords))
                    drug = pool[0]
                    alts = [d.name for d in pool[1:6]]      # same target, equally eligible
                    ind = drug.best_indication(keywords, 1) if keywords else None   # best match at any phase
                    choice = (i, drug, ind if want is not None else None, want, alts, len(pool) - 1)
                    break
            if choice:
                break
        if choice is None:
            top = grp[0]
            gaps.append(dict(module=int(m), size=len(grp), mass=mass[m] / total_mass,
                             top_target=names[top], top_bill=float(bill[top])))
            continue
        i, drug, indicated, want, alts, n_alt = choice
        # modules this drug also reaches through its other targets
        reached = {int(m)}
        for t in drug.targets:
            if t in idx and lab[idx[t]] >= 0:
                reached.add(int(lab[idx[t]]))
        overlap = sorted((reached - {int(m)}) & covered)   # modules already hit by an earlier pick
        new = sorted(reached - {int(m)} - covered)
        covered |= reached
        picks.append(dict(module=int(m), size=len(grp), mass=mass[m] / total_mass,
                          target=names[i], bill=float(bill[i]), drug=drug,
                          also_reaches=new, overlaps=overlap, indicated=indicated,
                          indicated_pass=want, direct=names[i] in drug.targets,
                          alternatives=alts, n_alternatives=n_alt))
    return dict(part=part, picks=picks, gaps=gaps, n_modules=len(part), covered=sorted(covered),
                covered_mass=sum(mass[m] for m in covered) / total_mass)


def pair_coupling(C, i, j):
    """Direct coupling if linked, else the best two-hop path strength."""
    return float(C[i, j]) if C[i, j] > 0 else float((C[i] * C[j]).max())


def vertical_pairs(analysis, lib, max_modules=5, approved_only=False, keywords=None, min_phase=3,
                   eligible=None, same_module_only=True):
    """Tightly-coupled drugged pairs: the 'vertical' combination style (BRAF+MEK, CDK4/6+ER,
    KRAS+EGFR).  Every pair of drugged proteins (distinct drugs, no shared target) scored by
    (bill_i + bill_j) * coupling_ij, highest first.  By default only pairs inside one module.
    What validate_combos.py supports (n=8, small): approved pairs are DIRECTLY LINKED far more often
    than drugged pairs in general (6/8 vs 31%, p=0.014) and rank high on bill; the coupling
    MAGNITUDE factor is inert (STRING edges >= 0.9 are all ~1; dropping it, or even the link
    requirement, leaves the ranking unchanged), so the score is effectively 'bill sum of a linked
    pair'.  The shared-module evidence for this mode is weaker (5/8, p=0.18).
    CONTROLLED for same-disease pairing (README section 7; validate_combos_matched and
    validate_combos_string --sweep) the linkage enrichment of approved pairs over other same-disease
    drug pairs is ~2x on the union network, ABSENT on a disease-agnostic STRING network at scores
    0.4-0.7 and marginal at 0.9: there is no robust evidence that combination partners are closer
    than other drugs for the same disease.  Treat this mode as a heuristic, not a validated one."""
    names, idx = analysis.names, {n: i for i, n in enumerate(analysis.names)}
    lab = analysis.label_vector(analysis.level_with_at_most(max_modules))
    bt = by_target(lib, approved_only)
    cand = []                                       # (protein index, best drug, indicated?)
    for n, i in idx.items():
        if eligible is not None and not eligible[i]:
            continue
        ds = bt.get(n) or []
        if keywords:
            ind = [d for d in ds if d.indicated_for(keywords, min_phase)]
            ds = ind or ds
            flag = bool(ind)
        else:
            flag = None
        if ds:
            cand.append((i, ds[:4], flag))          # a few alternatives per protein, best first
    out = []
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            (i, dsi, fi), (j, dsj, fj) = cand[a], cand[b]
            if same_module_only and lab[i] != lab[j]:
                continue
            c = pair_coupling(analysis.C, i, j)
            if c <= 0:
                continue
            # first drug combination that is two different drugs with no shared target; trying
            # alternatives matters when both proteins' best drug is the same multi-target agent
            combo = next(((di, dj) for di in dsi for dj in dsj
                          if di.name != dj.name and not (set(di.targets) & set(dj.targets))), None)
            if combo is None:
                continue
            di, dj = combo
            out.append(dict(a=names[i], b=names[j], drug_a=di, drug_b=dj, module=int(lab[i]),
                            coupling=c, score=float((max(analysis.bill[i], 0) + max(analysis.bill[j], 0)) * c),
                            indicated=(fi, fj)))
    out.sort(key=lambda r: -r["score"])
    return out


def report_vertical(label, analysis, pairs, top=8):
    print(f"\n{label}: D = {analysis.D:.2f} ({analysis.verdict}); vertical (same-module, tightly coupled) pairs:")
    print(f"  {'Pair':22} {'Drugs':38} {'module':>6} {'coupling':>8} {'score':>6}")
    for p in pairs[:top]:
        tag = ""
        if p["indicated"][0] is not None:
            tag = " [both indicated]" if all(p["indicated"]) else (" [one indicated]" if any(p["indicated"]) else " [mechanism only]")
        print(f"  {p['a']+'+'+p['b']:22} {p['drug_a'].name+' + '+p['drug_b'].name:38} {p['module']:>6} "
              f"{p['coupling']:8.2f} {p['score']:6.3f}{tag}")


def report(label, analysis, plan):
    print(f"\n{label}: D = {analysis.D:.2f} ({analysis.verdict}, line {LINE}); "
          f"{len(analysis.names)} proteins; hierarchy cut into {plan['n_modules']} modules")
    if analysis.D <= LINE:
        print("  D below the line: the model says ONE well-chosen target can contain this "
              "disease; the combination below is optional.")
    print(f"  {'Module':>6} {'Share':>6}  {'Target':10} {'Drug':20} Mechanism / status")
    for p in plan["picks"]:
        d = p["drug"]
        extra = ""
        if p["indicated_pass"]:
            h, ph = p["indicated"]
            extra += (f"  [APPROVED for {h}]" if ph >= 4 else f"  [phase {ph} trials for {h}; not approved for this]")
        elif p["indicated_pass"] is False:
            extra += "  [MECHANISM ONLY: no late-phase indication for this disease in ChEMBL]"
        if d.marketed is False:
            extra += "  [NOT currently marketed in the US per Drugs@FDA (withdrawn/discontinued or ex-US)]"
        if p["target"] in d.seed_only:
            extra += "  [target from curated seed, NOT in ChEMBL mechanism data]"
        if not p["direct"]:
            extra += "  [indirect: acts through this gene, e.g. synthetic lethality]"
        if p["also_reaches"]:
            extra += f"  [also hits module {p['also_reaches']}]"
        if p["overlaps"]:
            extra += f"  [REDUNDANT with module {p['overlaps']} already hit]"
        if p["n_alternatives"]:
            more = f", +{p['n_alternatives'] - len(p['alternatives'])} more" if p["n_alternatives"] > len(p["alternatives"]) else ""
            extra += f"  [equally eligible on this target: {', '.join(p['alternatives'])}{more}; the pick is a tie-break, not a recommendation]"
        print(f"  {p['module']:>6} {p['mass']*100:5.1f}%  {p['target']:10} {d.name:20} "
              f"{d.action}, {d.status}{extra}")
    for g in plan["gaps"]:
        print(f"  {g['module']:>6} {g['mass']*100:5.1f}%  {g['top_target']:10} {'-- no drug in library --':20} "
              f"top-bill target of an undrugged module")
    print(f"  modules hit: {len(plan['covered'])}/{plan['n_modules']} ({len(plan['picks'])} drugs), "
          f"share of coupling mass covered: {plan['covered_mass']*100:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("disease", nargs="?")
    ap.add_argument("--modules", type=int, default=5)
    ap.add_argument("--score", type=float, default=0.9)
    ap.add_argument("--approved-only", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--indicated-only", action="store_true",
                    help="only drugs ChEMBL lists as indicated (phase>=3) for the disease")
    ap.add_argument("--mode", choices=["spread", "vertical"], default="spread",
                    help="spread = one drug per module; vertical = tightly coupled same-module pairs "
                         "(closer to how approved combinations are built, see validate_combos)")
    ap.add_argument("--min-phase", type=int, default=3, choices=[3, 4],
                    help="minimum ChEMBL phase for a drug to count as indicated (4 = approved for the disease)")
    a = ap.parse_args()
    lib = default_library()
    if not any(d.verified for d in lib.values()):
        print("NOTE: using the hand-written seed library (run `python3 -m lapras_tools.build_druglib` "
              "for the ChEMBL-verified one); no indication data.")
    keys = disease_keys() if a.all else [a.disease]
    if not keys[0]:
        ap.error("give a disease key or --all (keys: %s)" % ", ".join(disease_keys()))
    for k in keys:
        C, names, _, label = load_disease(k, a.score)
        an = analyze_network(C, names)
        kw = DISEASE_KEYWORDS.get(k) if any(d.verified for d in lib.values()) else None
        if a.indicated_only and not kw:
            ap.error("--indicated-only needs the ChEMBL library with indications (python3 -m lapras_tools.build_druglib)")
        if a.mode == "vertical":
            report_vertical(label, an, vertical_pairs(an, lib, a.modules, a.approved_only, kw, a.min_phase))
        else:
            report(label, an, design(an, lib, a.modules, a.approved_only, kw, a.indicated_only, a.min_phase))


if __name__ == "__main__":
    main()
