"""Batch runner: analyze all diseases using real STRING data.

Loads each disease's STRING TSV, builds the coupling matrix,
runs the full pipeline, and produces the periodic table of D values.
"""
import os, sys
import numpy as np

from coupling import (
    tc_gaussian, per_component_bill, cov_from_coupling, regularize,
    compile_gap, node_residuals, halving_depth, edge_sensitivity,
    degree_centrality,
)
from diseases import DISEASES
from pipeline import analyze, OUTDIR

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_string_network(tsv_path, known_targets, min_score=0.0):
    """Load a STRING TSV into (C, names, known_targets) like networks.py does.
    min_score: only include edges with combined score >= this threshold."""
    edges = {}
    all_proteins = set()
    with open(tsv_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            a = parts[2]  # preferredName_A
            b = parts[3]  # preferredName_B
            score = float(parts[5])
            if a == b or score < min_score:
                continue
            all_proteins.add(a)
            all_proteins.add(b)
            key = tuple(sorted([a, b]))
            edges[key] = max(edges.get(key, 0), score)

    names = sorted(all_proteins)
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    C = np.zeros((n, n))
    for (a, b), score in edges.items():
        i, j = idx[a], idx[b]
        C[i, j] = C[j, i] = score

    targets_in_network = {k: v for k, v in known_targets.items() if k in idx}
    return C, names, targets_in_network


def run_all():
    """Run the full pipeline on every disease and build the periodic table."""
    results = {}
    table = []

    # Parse --score threshold
    score_threshold = 0.7
    args = sys.argv[1:]
    if "--score" in args:
        si = args.index("--score")
        score_threshold = float(args[si + 1])
        args = args[:si] + args[si + 2:]
    diseases = args if args else list(DISEASES.keys())
    diseases = [d for d in diseases if d in DISEASES]

    print(f"  STRING score threshold: {score_threshold}\n")

    for key in diseases:
        d = DISEASES[key]
        tsv = os.path.join(DATA_DIR, f"{key}.tsv")
        if not os.path.exists(tsv):
            print(f"\n  SKIP {d['label']} — no STRING data ({tsv})")
            continue

        C, names, targets = load_string_network(tsv, d["known_targets"],
                                                 min_score=score_threshold)
        n = C.shape[0]
        edges = (C > 0).sum() // 2

        if n < 5 or edges < 3:
            print(f"\n  SKIP {d['label']} — too sparse ({n} proteins, {edges} edges)")
            continue

        bill, D = analyze(C, names, targets, d["label"])
        results[d["label"]] = (C, names, targets, bill, D)

        tc = tc_gaussian(cov_from_coupling(C, q=0.9))
        total_known = len(targets)
        ranked = np.argsort(-bill)
        top10_hits = len({names[i] for i in ranked[:10]} & set(targets.keys())) if total_known > 0 else 0
        expected = total_known * min(10, n) / n if n > 0 else 0
        enrichment = top10_hits / expected if expected > 0 else 0

        table.append({
            "key": key,
            "label": d["label"],
            "n": n,
            "edges": edges,
            "tc": tc,
            "D": D,
            "above_line": D > 3.03,
            "enrichment": enrichment,
            "top10_hits": top10_hits,
            "total_known": total_known,
        })

    # Print and save the periodic table
    print(f"\n\n{'='*100}")
    print(f"  THE PERIODIC TABLE: D VALUES FOR {len(table)} DISEASES")
    print(f"{'='*100}")
    print(f"  Capacity line: D = 3.03")
    print(f"  Below → modular (single-target viable)  |  Above → global (combinations required)")
    print(f"{'='*100}\n")

    table.sort(key=lambda x: x["D"])

    print(f"  {'Disease':40s} {'D':>6s}  {'Verdict':12s} {'Proteins':>8s} {'Edges':>7s} "
          f"{'TC':>7s} {'Enrich':>7s} {'Hits':>5s}")
    print(f"  {'-'*40} {'-'*6}  {'-'*12} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*5}")

    for row in table:
        d = row["D"]
        if d == float("inf"):
            d_str = "  inf"
        else:
            d_str = f"{d:6.2f}"
        verdict = "GLOBAL" if row["above_line"] else "MODULAR"
        enrich_str = f"{row['enrichment']:.1f}x" if row["enrichment"] > 0 else "  n/a"
        hit_str = f"{row['top10_hits']}/{row['total_known']}" if row["total_known"] > 0 else "n/a"
        print(f"  {row['label']:40s} {d_str}  {verdict:12s} {row['n']:8d} {row['edges']:7d} "
              f"{row['tc']:7.2f} {enrich_str:>7s} {hit_str:>5s}")

    # Separator at the line
    below = [r for r in table if not r["above_line"] and r["D"] != float("inf")]
    above = [r for r in table if r["above_line"]]
    print(f"\n  Below the line (modular):  {len(below)}")
    print(f"  Above the line (global):  {len(above)}")

    # Save
    out = []
    out.append(f"THE PERIODIC TABLE: D VALUES FOR {len(table)} DISEASES")
    out.append(f"Capacity line: D = 3.03")
    out.append(f"Below → modular (single-target viable)  |  Above → global (combinations required)\n")
    out.append(f"{'Disease':40s} {'D':>6s}  {'Verdict':12s} {'Proteins':>8s} {'Edges':>7s} "
               f"{'TC':>7s} {'Enrich':>7s}")
    out.append(f"{'-'*40} {'-'*6}  {'-'*12} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
    for row in table:
        d = row["D"]
        d_str = f"{d:6.2f}" if d != float("inf") else "   inf"
        verdict = "GLOBAL" if row["above_line"] else "MODULAR"
        enrich_str = f"{row['enrichment']:.1f}x" if row["enrichment"] > 0 else "  n/a"
        out.append(f"{row['label']:40s} {d_str}  {verdict:12s} {row['n']:8d} {row['edges']:7d} "
                   f"{row['tc']:7.2f} {enrich_str:>7s}")
    out.append(f"\nBelow the line (modular):  {len(below)}")
    out.append(f"Above the line (global):  {len(above)}")

    # Cross-disease analysis
    from pipeline import cross_disease
    all_for_cross = {label: (C, names, targets, bill)
                     for label, (C, names, targets, bill, D) in results.items()}
    if len(all_for_cross) >= 2:
        cross_disease(all_for_cross)

    path = os.path.join(OUTDIR, "periodic_table.txt")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"\n  [saved {path}]")


if __name__ == "__main__":
    run_all()
