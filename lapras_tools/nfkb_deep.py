"""Deep NF-κB analysis: pathway components, druggability, and target nomination.

NF-κB (NFKB1) is the most undrugged coupling bottleneck in medicine, ranking
in the top 30% of bill across many of the 46 diseases. This script goes deeper: which
NF-κB pathway components are most structurally important, which are most
druggable, and what does the coupling structure suggest about where to intervene?

    python3 -m lapras_tools.nfkb_deep
"""
import numpy as np
from .core import load_disease, analyze_network, LINE, disease_keys


NFKB_PATHWAY = {
    "NFKB1":  {"role": "transcription factor (p50/p105)", "druggability": "low (TF)", "pocket": "no known binding site"},
    "NFKB2":  {"role": "transcription factor (p52/p100)", "druggability": "low (TF)", "pocket": "no known binding site"},
    "RELA":   {"role": "transcription factor (p65)", "druggability": "low (TF)", "pocket": "no known binding site"},
    "RELB":   {"role": "transcription factor", "druggability": "low (TF)", "pocket": "no known binding site"},
    "REL":    {"role": "transcription factor (c-Rel)", "druggability": "low (TF)", "pocket": "no known binding site"},
    "IKBKB":  {"role": "IKKβ kinase (canonical pathway)", "druggability": "HIGH (kinase)", "pocket": "ATP-binding site, allosteric site"},
    "CHUK":   {"role": "IKKα kinase", "druggability": "HIGH (kinase)", "pocket": "ATP-binding site"},
    "IKBKG":  {"role": "NEMO/IKKγ scaffold", "druggability": "medium (PPI)", "pocket": "ubiquitin-binding domain, CC2-LZ interface"},
    "NFKBIA": {"role": "IκBα inhibitor", "druggability": "medium (stabilizer)", "pocket": "phospho-degron site"},
    "NFKBIB": {"role": "IκBβ inhibitor", "druggability": "low", "pocket": "none established"},
    "TNF":    {"role": "upstream activator", "druggability": "APPROVED (infliximab, adalimumab, etc.)", "pocket": "receptor interface (antibody)"},
    "TNFRSF1A": {"role": "TNF receptor 1", "druggability": "APPROVED (etanercept)", "pocket": "extracellular domain"},
    "IL1B":   {"role": "upstream activator (inflammasome)", "druggability": "APPROVED (canakinumab)", "pocket": "receptor interface (antibody)"},
    "IL1R1":  {"role": "IL-1 receptor", "druggability": "APPROVED (anakinra)", "pocket": "extracellular domain"},
    "MYD88":  {"role": "adaptor (TLR/IL-1R signaling)", "druggability": "medium (PPI/TIR domain)", "pocket": "TIR-TIR interface, death domain"},
    "IRAK4":  {"role": "kinase (TLR/IL-1R cascade)", "druggability": "HIGH (kinase)", "pocket": "ATP-binding site — emavusertib in trials"},
    "IRAK1":  {"role": "kinase (TLR signaling)", "druggability": "HIGH (kinase)", "pocket": "ATP-binding site"},
    "TRAF6":  {"role": "E3 ubiquitin ligase", "druggability": "medium (PPI)", "pocket": "TRAF domain, RING domain"},
    "TRAF2":  {"role": "E3 ubiquitin ligase (non-canonical)", "druggability": "medium", "pocket": "TRAF domain"},
    "RIPK1":  {"role": "kinase (necroptosis / NF-κB)", "druggability": "HIGH (kinase)", "pocket": "ATP site, allosteric (GSK'963-type)"},
    "RIPK2":  {"role": "kinase (NOD signaling)", "druggability": "HIGH (kinase)", "pocket": "ATP-binding site"},
    "TLR4":   {"role": "pattern recognition receptor", "druggability": "medium (antibody/antagonist)", "pocket": "MD-2 interface"},
    "TLR2":   {"role": "pattern recognition receptor", "druggability": "medium", "pocket": "lipopeptide-binding groove"},
    "BCL10":  {"role": "CARD–MALT1 signaling (lymphocyte NF-κB)", "druggability": "medium", "pocket": "CARD domain"},
    "MALT1":  {"role": "paracaspase (lymphocyte NF-κB)", "druggability": "HIGH (protease)", "pocket": "active site — MLT-748 class"},
    "CARD11": {"role": "scaffold (CBM complex)", "druggability": "medium (PPI)", "pocket": "CC domain, CARD interface"},
}


def main():
    print("=" * 100)
    print("NF-κB DEEP ANALYSIS: pathway components, coupling contribution, and target nomination")
    print("=" * 100)

    all_rows = []
    for k in disease_keys():
        try:
            C, names, _, label = load_disease(k, 0.9)
        except SystemExit:
            continue
        an = analyze_network(C, names)
        ranked = an.ranked()
        n = len(names)
        name_set = set(names)

        nfkb_genes = [g for g in NFKB_PATHWAY if g in name_set]
        if not nfkb_genes:
            continue

        bill_ranks = np.empty(n)
        bill_ranks[ranked] = np.arange(n)

        gene_data = []
        for g in nfkb_genes:
            idx = names.index(g)
            gene_data.append({
                "gene": g,
                "bill_rank": int(bill_ranks[idx]) + 1,
                "bill_pct": float(bill_ranks[idx]) / n,
                "bill_value": float(an.bill[idx]),
            })

        best = min(gene_data, key=lambda x: x["bill_rank"])

        all_rows.append({
            "key": k,
            "label": label,
            "D": an.D,
            "verdict": an.verdict,
            "n": n,
            "nfkb_genes": gene_data,
            "best_nfkb": best,
            "n_pathway": len(nfkb_genes),
        })

    # diseases where NF-κB pathway is most structurally important
    all_rows.sort(key=lambda r: r["best_nfkb"]["bill_pct"])

    print(f"\n  NF-κB pathway present in {len(all_rows)} of 46 diseases")

    top30 = [r for r in all_rows if r["best_nfkb"]["bill_pct"] <= 0.30]
    print(f"  Top 30% by bill in {len(top30)} diseases")

    print(f"\n  {'Disease':35s} {'D':>5} {'best NF-κB gene':>18} {'rank':>5} {'pctl':>6} {'#path':>6}")
    print("  " + "-" * 80)
    for r in all_rows:
        b = r["best_nfkb"]
        in_top = " **" if b["bill_pct"] <= 0.30 else ""
        print(f"  {r['label'][:35]:35s} {r['D']:5.2f} {b['gene']:>18} {b['bill_rank']:5d} "
              f"{b['bill_pct']:6.1%} {r['n_pathway']:6d}{in_top}")

    # target nomination: rank NF-κB pathway components by coupling importance × druggability
    print(f"\n{'=' * 100}")
    print("TARGET NOMINATION: NF-κB pathway components ranked by coupling × druggability")
    print("=" * 100)

    gene_scores = {}
    for r in all_rows:
        for gd in r["nfkb_genes"]:
            g = gd["gene"]
            if g not in gene_scores:
                gene_scores[g] = {"appearances": 0, "top30_count": 0,
                                  "total_bill": 0.0, "diseases": []}
            gene_scores[g]["appearances"] += 1
            if gd["bill_pct"] <= 0.30:
                gene_scores[g]["top30_count"] += 1
            gene_scores[g]["total_bill"] += gd["bill_value"]
            gene_scores[g]["diseases"].append(
                (r["label"], gd["bill_rank"], r["n"]))

    druggability_score = {
        "APPROVED": 1.0, "HIGH": 0.8, "medium": 0.5, "low": 0.2
    }

    print(f"\n  {'Gene':10s} {'#diseases':>10} {'#top30%':>8} {'druggability':>14} "
          f"{'drug_score':>11} {'priority':>9}  status")
    print("  " + "-" * 95)

    nominations = []
    for gene, data in sorted(gene_scores.items(), key=lambda x: -x[1]["top30_count"]):
        info = NFKB_PATHWAY.get(gene, {})
        drug_text = info.get("druggability", "unknown")
        drug_key = drug_text.split("(")[0].strip().split(" ")[0]
        ds = druggability_score.get(drug_key, 0.3)
        priority = data["top30_count"] * ds
        nominations.append((gene, data, ds, priority, info))

    nominations.sort(key=lambda x: -x[3])
    for gene, data, ds, priority, info in nominations:
        drug_text = info.get("druggability", "unknown")
        pocket = info.get("pocket", "unknown")
        role = info.get("role", "")
        print(f"  {gene:10s} {data['appearances']:10d} {data['top30_count']:8d} "
              f"{drug_text[:14]:>14} {ds:11.1f} {priority:9.1f}  {role}")

    # top nominations
    print(f"\n{'=' * 100}")
    print("ACTIONABLE TARGETS (high coupling × high druggability)")
    print("=" * 100)

    for gene, data, ds, priority, info in nominations[:8]:
        if priority < 0.5:
            break
        print(f"\n  {gene} — {info.get('role', '')}")
        print(f"    Present in {data['appearances']} diseases, "
              f"top 30% in {data['top30_count']}")
        print(f"    Druggability: {info.get('druggability', 'unknown')}")
        print(f"    Binding site: {info.get('pocket', 'unknown')}")
        top_diseases = sorted(data["diseases"], key=lambda x: x[1])[:5]
        disease_str = ", ".join(f"{d[0][:20]} (#{d[1]}/{d[2]})"
                                for d in top_diseases)
        print(f"    Best-ranked in: {disease_str}")


if __name__ == "__main__":
    main()
