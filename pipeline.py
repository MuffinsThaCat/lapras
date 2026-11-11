"""Drug target discovery pipeline.

Ranks proteins by their contribution to total coupling (the "bill"),
decomposes disease networks into modules, predicts whether single-target
therapy can work (D < 3.03) or combination therapy is required (D > 3.03),
and identifies novel targets — high-bill proteins with no approved drug.

Usage:
    python3 pipeline.py                # all diseases
    python3 pipeline.py --disease aging
    python3 pipeline.py --disease cancer
    python3 pipeline.py --disease alzheimer
    python3 pipeline.py --disease als
    python3 pipeline.py --disease pancreatic
"""
import os, sys, math, argparse
import numpy as np

from coupling import (
    tc_gaussian, per_component_bill, cov_from_coupling, regularize,
    compile_gap, node_residuals, halving_depth, edge_sensitivity,
    degree_centrality,
)
from networks import (
    load_string_aging, construct_cancer_network,
    construct_alzheimer_network, construct_als_network,
    construct_pancreatic_cancer_network,
)

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTDIR, exist_ok=True)

LINE = 3.03


def analyze(C, names, known_targets, disease, q=0.9):
    """Full pipeline on one disease network."""
    n = C.shape[0]
    S = cov_from_coupling(C, q=q)
    S, shrink = regularize(S)
    tc = tc_gaussian(S)
    bill = per_component_bill(S)

    res = compile_gap(C)
    parts = [g for _, g in res if len(g) > 1]
    if not parts:
        parts = [[[i] for i in range(n)]]
    Rn, R = node_residuals(C, parts)
    D, D_exact = halving_depth(R)
    sens, edges = edge_sensitivity(C, parts, R)

    leaf_modules = parts[0]
    protein_module = {}
    for mi, mod in enumerate(leaf_modules):
        for idx in mod:
            protein_module[idx] = mi

    ranked = np.argsort(-bill)
    deg = degree_centrality(C)

    out = []
    def p(s=""):
        print(s)
        out.append(s)

    p(f"\n{'='*72}")
    p(f"  {disease.upper()}")
    p(f"{'='*72}")
    p(f"  Proteins: {n}   Edges: {(C > 0).sum() // 2}")
    p(f"  TC = {tc:.2f} nats   (e^TC = {math.exp(tc):.0f})")
    p(f"  D = {D:.2f} — {'ABOVE' if D > LINE else 'BELOW'} the capacity line ({LINE})")
    if D > LINE:
        p(f"  -> Globally entangled. Single-target therapy insufficient.")
    else:
        p(f"  -> Hierarchically containable. Modular targeting viable.")

    # Known targets
    p(f"\n  KNOWN TARGETS (validation):")
    p(f"  {'Rank':>4s}  {'Protein':12s} {'Bill':>8s} {'Module':>6s} {'Drug/status'}")
    count = 0
    for rank, i in enumerate(ranked):
        if names[i] in known_targets:
            count += 1
            p(f"  {rank+1:4d}  {names[i]:12s} {bill[i]:8.4f} "
              f"{protein_module.get(i,'?'):>6}  {known_targets[names[i]]}")
            if count >= 15:
                break

    # Novel targets
    p(f"\n  NOVEL TARGETS (high bill, no approved drug):")
    p(f"  {'Rank':>4s}  {'Protein':12s} {'Bill':>8s} {'Module':>6s} "
      f"{'Degree':>6s} {'Why interesting'}")
    novel_count = 0
    for rank, i in enumerate(ranked):
        if names[i] not in known_targets and bill[i] > np.median(bill):
            novel_count += 1
            mod_id = protein_module.get(i, -1)
            mod_members = ([names[j] for j in leaf_modules[mod_id]]
                           if 0 <= mod_id < len(leaf_modules) else [])
            mod_known = [m for m in mod_members if m in known_targets]
            if mod_known:
                reason = f"same module as {', '.join(mod_known[:2])}"
            elif bill[i] > bill.mean() + 2 * bill.std():
                reason = "exceptionally high bill (>2 sigma)"
            elif int(deg[i]) >= np.percentile(deg, 80):
                reason = "high connectivity hub"
            else:
                reason = "above-median bill, no drug exists"
            p(f"  {rank+1:4d}  {names[i]:12s} {bill[i]:8.4f} "
              f"{mod_id:>6d} {int(deg[i]):>6d}  {reason}")
            if novel_count >= 15:
                break

    # Combination proposals
    p(f"\n  COMBINATION PROPOSALS (one target per module, by bill):")
    used_modules = set()
    combo = []
    for rank, i in enumerate(ranked):
        mod_id = protein_module.get(i, -1)
        if mod_id in used_modules or mod_id < 0:
            continue
        used_modules.add(mod_id)
        drug = known_targets.get(names[i], "NO DRUG — novel target")
        combo.append((names[i], mod_id, bill[i], drug, names[i] not in known_targets))
        if len(combo) >= 5:
            break
    for ci, (nm, mid, b, drug, novel) in enumerate(combo):
        marker = " <- NOVEL" if novel else ""
        p(f"    {ci+1}. {nm:12s} (module {mid}, bill={b:.4f}){marker}")
        p(f"       {drug}")

    # Cross-module edge targets
    if sens is not None:
        mean_sens = np.abs(sens).mean(0)
        p(f"\n  INTERACTION TARGETS (cross-module, high sensitivity):")
        top_edges = np.argsort(-mean_sens)[:20]
        edge_count = 0
        for ei in top_edges:
            a, b_idx = edges[ei]
            ma = protein_module.get(a, -1)
            mb = protein_module.get(b_idx, -1)
            if ma == mb:
                continue
            na, nb = names[a], names[b_idx]
            a_known = na in known_targets
            b_known = nb in known_targets
            if not a_known or not b_known:
                edge_count += 1
                ms = mean_sens[ei]
                tag = "[both novel]" if not a_known and not b_known else "[one novel]"
                p(f"    {na:12s} <-> {nb:12s}  "
                  f"|sens|={ms:.6f}  modules {ma}<->{mb}  {tag}")
            if edge_count >= 8:
                break

    # Enrichment
    hub_set = set(known_targets.keys()) & set(names)
    total_hubs = len(hub_set)
    if total_hubs > 0:
        p(f"\n  ENRICHMENT:")
        for tn in [10, 20]:
            if tn > n:
                continue
            top_names = {names[i] for i in ranked[:tn]}
            hits = len(top_names & hub_set)
            expected = total_hubs * tn / n
            enrichment = hits / expected if expected > 0 else 0
            p(f"    Top {tn}: {hits}/{total_hubs} known targets "
              f"(expected {expected:.1f}, enrichment {enrichment:.1f}x)")

    safe = disease.lower().replace(' ','_').replace('/','_').replace("'","")
    path = os.path.join(OUTDIR, f"{safe}.txt")
    with open(path, "w") as f:
        f.write("\n".join(out))
    p(f"\n  [saved {path}]")
    return bill, D


def cross_disease(all_results):
    """Find proteins ranking high across multiple diseases."""
    out = []
    def p(s=""):
        print(s)
        out.append(s)

    p(f"\n{'='*72}")
    p(f"  CROSS-DISEASE REPURPOSING CANDIDATES")
    p(f"{'='*72}")
    p(f"  Proteins in the top 30% by bill across 2+ diseases\n")

    protein_scores = {}
    for disease, (C, names, targets, bill) in all_results.items():
        n = len(names)
        ranked = np.argsort(-bill)
        for rank, i in enumerate(ranked):
            pct = rank / n
            if pct <= 0.30:
                nm = names[i]
                if nm not in protein_scores:
                    protein_scores[nm] = []
                protein_scores[nm].append((disease, pct, bill[i]))

    multi = {k: v for k, v in protein_scores.items() if len(v) >= 2}
    multi_sorted = sorted(multi.items(), key=lambda x: -len(x[1]))

    p(f"  {'Protein':12s} {'#Diseases':>9s}  Diseases")
    p(f"  {'--------':12s} {'---------':>9s}  --------")
    for nm, entries in multi_sorted:
        disease_strs = [f"{d} (top {pct*100:.0f}%)" for d, pct, b in entries]
        p(f"  {nm:12s} {len(entries):>9d}  {'; '.join(disease_strs)}")

    p(f"\n  STRONGEST (3+ diseases):\n")
    for nm, entries in multi_sorted:
        if len(entries) >= 3:
            p(f"  {nm}:")
            for d, pct, b in entries:
                p(f"    {d:30s}: top {pct*100:.0f}%, bill={b:.4f}")
            p("")

    path = os.path.join(OUTDIR, "cross_disease.txt")
    with open(path, "w") as f:
        f.write("\n".join(out))
    p(f"  [saved {path}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", default="all",
                    choices=["all","aging","cancer","alzheimer","als","pancreatic"])
    args = ap.parse_args()

    loaders = {
        "aging":      ("Aging",              load_string_aging),
        "cancer":     ("Cancer",             construct_cancer_network),
        "alzheimer":  ("Alzheimer's",        construct_alzheimer_network),
        "als":        ("ALS",                construct_als_network),
        "pancreatic": ("Pancreatic Cancer",  construct_pancreatic_cancer_network),
    }

    if args.disease != "all":
        label, loader = loaders[args.disease]
        C, names, targets = loader()
        analyze(C, names, targets, label)
        return

    all_results = {}
    for key, (label, loader) in loaders.items():
        C, names, targets = loader()
        bill, D = analyze(C, names, targets, label)
        all_results[label] = (C, names, targets, bill)

    cross_disease(all_results)


if __name__ == "__main__":
    main()
