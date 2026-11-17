"""Rare disease triage: which rare diseases are most likely to respond to single-target therapy?

Runs the Lapras pipeline on rare diseases to rank them by tractability.
Diseases with D < 3.03 are predicted to be controllable by a single well-chosen
target.  The output is a prioritized list for rare disease foundations and
drug developers.

    python3 -m lapras_tools.rare_disease_triage
    python3 -m lapras_tools.rare_disease_triage --download   # fetch STRING data first
    python3 -m lapras_tools.rare_disease_triage --full       # include already-analyzed diseases

Gene lists are curated from OMIM, Orphanet, and disease-specific literature.
Each disease needs >= 10 interacting proteins for a meaningful D value.
"""
import os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from .core import load_disease, analyze_network, LINE, DATA_DIR, disease_keys
from .uncertainty import d_uncertainty

RARE_DISEASES = {
    "pompe": {
        "label": "Pompe Disease (Glycogen Storage Disease II)",
        "proteins": [
            "GAA", "LAMP1", "LAMP2", "TFEB", "MTOR", "AKT1", "PIK3CA",
            "AMPK", "PRKAA1", "PRKAA2", "ULK1", "ATG5", "ATG7", "BECN1",
            "LC3B", "SQSTM1", "CTSD", "CTSB", "CTSL", "GYS1", "PYGM",
            "GBE1", "AGL", "PHKA2", "PHKB", "GSK3B", "IGF1R", "IGF1",
            "GH1", "FOXO3", "TSC1", "TSC2",
        ],
        "known_targets": {
            "GAA": "alglucosidase alfa (Lumizyme), avalglucosidase alfa (Nexviazyme)",
        },
        "mechanism": "enzyme replacement",
        "current_therapy": "mono (ERT)",
    },

    "niemann_pick_c": {
        "label": "Niemann-Pick Disease Type C",
        "proteins": [
            "NPC1", "NPC2", "SMPD1", "LIPA", "ABCA1", "ABCG1", "LDLR",
            "PCSK9", "HMGCR", "SREBF1", "SREBF2", "INSIG1", "SCAP",
            "OSBPL1A", "STARD3", "STARD4", "LAMP1", "LAMP2", "TFEB",
            "MTOR", "AKT1", "CTSD", "GBA", "PSAP", "CLN3", "CLN5",
        ],
        "known_targets": {
            "NPC1": "miglustat (substrate reduction, off-label)",
        },
        "mechanism": "substrate reduction / chaperone",
        "current_therapy": "mono (miglustat)",
    },

    "mps_i": {
        "label": "MPS I (Hurler/Scheie Syndrome)",
        "proteins": [
            "IDUA", "GUSB", "GALNS", "HGSNAT", "SGSH", "NAGLU", "GNS",
            "ARSB", "GLB1", "HEXB", "HEXA", "LAMP1", "LAMP2", "TFEB",
            "GNPTAB", "GNPTG", "MCOLN1", "CLN3", "CTSD", "CTSB",
            "SUMF1", "IGF2R", "M6PR", "GAA", "GBA",
        ],
        "known_targets": {
            "IDUA": "laronidase (Aldurazyme)",
        },
        "mechanism": "enzyme replacement",
        "current_therapy": "mono (ERT)",
    },

    "mps_ii": {
        "label": "MPS II (Hunter Syndrome)",
        "proteins": [
            "IDS", "IDUA", "GUSB", "GALNS", "SGSH", "NAGLU", "HGSNAT",
            "ARSB", "GLB1", "HEXB", "LAMP1", "LAMP2", "TFEB", "GNPTAB",
            "GNPTG", "MCOLN1", "CLN3", "CTSD", "SUMF1", "IGF2R",
            "M6PR", "GAA", "GBA", "PSAP", "HEXA",
        ],
        "known_targets": {
            "IDS": "idursulfase (Elaprase)",
        },
        "mechanism": "enzyme replacement",
        "current_therapy": "mono (ERT)",
    },

    "friedreich_ataxia": {
        "label": "Friedreich's Ataxia",
        "proteins": [
            "FXN", "ISCU", "NFS1", "LYRM4", "HSPA9", "HSCB", "FDXR",
            "FDX2", "BOLA3", "NFU1", "GLRX5", "ABCB7", "ALAS2",
            "ACO2", "SDHB", "SDHC", "SDHD", "SDHA", "SOD2", "GPX4",
            "NRF2", "KEAP1", "HMOX1", "NQO1", "GCLC", "GCLM", "GSR",
            "TXN2", "TXNRD2", "PRDX3", "CAT", "PPARGC1A",
        ],
        "known_targets": {
            "FXN": "omaveloxolone (Skyclarys, NRF2 activator — approved 2023)",
        },
        "mechanism": "NRF2 pathway activation (indirect)",
        "current_therapy": "mono (omaveloxolone)",
    },

    "tuberous_sclerosis": {
        "label": "Tuberous Sclerosis Complex",
        "proteins": [
            "TSC1", "TSC2", "MTOR", "RPTOR", "RICTOR", "RPS6KB1",
            "EIF4EBP1", "RHEB", "AKT1", "PIK3CA", "PTEN", "AMPK",
            "PRKAA1", "ULK1", "TFEB", "HIF1A", "VEGFA", "CCND1",
            "RB1", "TP53", "CDK4", "CDK6", "CDKN1A", "CDKN2A",
            "MYC", "STAT3", "JAK2", "EGFR", "ERBB2", "RAF1",
        ],
        "known_targets": {
            "MTOR": "everolimus (Afinitor), sirolimus",
        },
        "mechanism": "mTOR inhibition",
        "current_therapy": "mono (mTOR inhibitor)",
    },

    "neurofibromatosis_1": {
        "label": "Neurofibromatosis Type 1",
        "proteins": [
            "NF1", "KRAS", "HRAS", "NRAS", "BRAF", "RAF1", "MAP2K1",
            "MAP2K2", "MAPK1", "MAPK3", "SOS1", "GRB2", "GAB1",
            "SPRED1", "SPRED2", "PTPN11", "CBL", "EGFR", "MET",
            "PDGFRA", "PDGFRB", "KIT", "RET", "ALK", "FGFR1",
            "PIK3CA", "AKT1", "MTOR", "TP53", "CDKN2A",
        ],
        "known_targets": {
            "MAP2K1": "selumetinib (Koselugo, MEK inhibitor — approved 2020)",
        },
        "mechanism": "RAS/MAPK pathway inhibition",
        "current_therapy": "mono (selumetinib for plexiform neurofibromas)",
    },

    "vhl_disease": {
        "label": "Von Hippel-Lindau Disease",
        "proteins": [
            "VHL", "HIF1A", "EPAS1", "HIF3A", "EGLN1", "EGLN2", "EGLN3",
            "EP300", "CREBBP", "ARNT", "VEGFA", "VEGFB", "VEGFC",
            "KDR", "FLT1", "FLT4", "PDGFB", "PDGFRA", "PDGFRB",
            "TGFA", "EGFR", "ERBB2", "MET", "SRC", "AKT1", "MTOR",
            "PIK3CA", "PTEN", "RB1", "TP53",
        ],
        "known_targets": {
            "HIF2A": "belzutifan (Welireg, HIF-2α inhibitor — approved 2021)",
        },
        "mechanism": "HIF-2α inhibition",
        "current_therapy": "mono (belzutifan for RCC)",
    },

    "hereditary_angioedema": {
        "label": "Hereditary Angioedema",
        "proteins": [
            "SERPING1", "F12", "KLKB1", "KNG1", "BDKRB1", "BDKRB2",
            "F11", "PROC", "PROS1", "SERPINC1", "SERPINA1", "SERPINE1",
            "PLG", "PLAT", "PLAU", "ACE", "ACE2", "AGT", "REN",
            "AGTR1", "MAS1", "NOS3", "EDN1", "EDNRA", "EDNRB",
        ],
        "known_targets": {
            "KLKB1": "lanadelumab (Takhzyro, anti-kallikrein — approved 2018)",
            "F12": "garadacimab (anti-FXIIa — approved 2023)",
            "BDKRB2": "icatibant (Firazyr, B2R antagonist)",
            "SERPING1": "C1-INH replacement (Cinryze, Haegarda)",
        },
        "mechanism": "kallikrein-kinin pathway inhibition",
        "current_therapy": "mono (lanadelumab or C1-INH prophylaxis)",
    },

    "alpha1_antitrypsin": {
        "label": "Alpha-1 Antitrypsin Deficiency",
        "proteins": [
            "SERPINA1", "ELANE", "CTSG", "PRTN3", "MMP12", "MMP9",
            "MMP2", "TIMP1", "TIMP2", "TGFB1", "SMAD2", "SMAD3",
            "SMAD4", "TNF", "IL1B", "IL6", "IL8", "NFKB1", "RELA",
            "IKBKB", "IKBKG", "CHUK", "CASP3", "CASP8", "BCL2",
            "BAX", "TP53", "NFE2L2", "KEAP1", "SOD1",
        ],
        "known_targets": {
            "SERPINA1": "augmentation therapy (Prolastin, Zemaira, Glassia)",
        },
        "mechanism": "protein replacement",
        "current_therapy": "mono (augmentation therapy)",
    },

    "familial_mediterranean_fever": {
        "label": "Familial Mediterranean Fever",
        "proteins": [
            "MEFV", "NLRP3", "PYCARD", "CASP1", "IL1B", "IL1R1",
            "IL1RN", "IL18", "IL18R1", "IL6", "TNF", "TNFRSF1A",
            "NFKB1", "RELA", "IKBKB", "CHUK", "IRAK4", "MYD88",
            "TRAF6", "RIPK2", "NOD2", "TLR4", "TLR2", "HMGB1",
            "S100A8", "S100A9", "SAA1", "CRP", "STAT3", "JAK2",
        ],
        "known_targets": {
            "tubulin": "colchicine (standard of care)",
            "IL1B": "canakinumab (Ilaris, for colchicine-resistant)",
            "IL1R1": "anakinra (Kineret, for colchicine-resistant)",
        },
        "mechanism": "inflammasome / IL-1 pathway",
        "current_therapy": "mono (colchicine)",
    },

    "myasthenia_gravis": {
        "label": "Myasthenia Gravis",
        "proteins": [
            "CHRNA1", "CHRNB1", "CHRND", "CHRNE", "CHRNG", "MUSK",
            "LRP4", "RAPSN", "DOK7", "AGRN", "ACHE", "CHAT",
            "SLC18A3", "SLC5A7", "VAMP1", "SNAP25", "SYT1",
            "C5", "C3", "C1QA", "C1QB", "C2", "C4A", "FCGR1A",
            "FCGR2A", "FCGR3A", "CD80", "CD86", "CTLA4", "ICOS",
        ],
        "known_targets": {
            "ACHE": "pyridostigmine (Mestinon)",
            "C5": "eculizumab (Soliris), ravulizumab (Ultomiris)",
            "FCGRT": "efgartigimod (Vyvgart, FcRn blocker)",
        },
        "mechanism": "neuromuscular junction / complement",
        "current_therapy": "mono (pyridostigmine) or combo (+ immunosuppressant)",
    },

    "pheochromocytoma": {
        "label": "Pheochromocytoma / Paraganglioma",
        "proteins": [
            "RET", "VHL", "SDHB", "SDHC", "SDHD", "SDHA", "SDHAF2",
            "NF1", "MAX", "TMEM127", "EPAS1", "HIF1A", "EGLN1",
            "FH", "MDH2", "IDH1", "IDH2", "KIF1B", "GDNF", "GFRA1",
            "HRAS", "BRAF", "FGFR1", "MET", "VEGFA", "KDR",
            "PDGFRA", "PDGFRB", "TH", "DBH", "PNMT",
        ],
        "known_targets": {
            "ADRA": "phenoxybenzamine (alpha-blocker, preoperative)",
        },
        "mechanism": "surgical; alpha-blockade preoperative",
        "current_therapy": "surgery (pharmacotherapy is supportive)",
    },

    "xla_agammaglobulinemia": {
        "label": "X-linked Agammaglobulinemia (Bruton's)",
        "proteins": [
            "BTK", "BLNK", "SYK", "LYN", "PIK3CA", "PIK3CD",
            "PLCG2", "PRKCB", "CARD11", "BCL10", "MALT1",
            "NFKB1", "RELA", "IKBKB", "TRAF6", "MYD88", "IRAK4",
            "CD19", "CD79A", "CD79B", "IGHM", "CD40", "CD40LG",
            "TNFRSF13B", "TNFRSF13C", "AICDA", "RAG1", "RAG2",
        ],
        "known_targets": {
            "BTK": "immunoglobulin replacement (IgG infusions)",
        },
        "mechanism": "immunoglobulin replacement",
        "current_therapy": "mono (IgG replacement)",
    },

    "als": {
        "label": "Amyotrophic Lateral Sclerosis",
        "proteins": [
            "SOD1", "TARDBP", "FUS", "C9orf72", "OPTN", "TBK1",
            "SQSTM1", "VCP", "UBQLN2", "PFN1", "TUBA4A", "KIF5A",
            "NEK1", "DNAJC7", "ANG", "ATXN2", "SETX", "ALS2",
            "VAPB", "SIGMAR1", "CHCHD10", "MATR3", "HNRNPA1",
            "EWSR1", "TAF15", "NEFH", "NEFL", "PRPH", "DCTN1",
            "ERBB4", "VEGFA", "IL6", "TNF", "NFKB1", "CASP3",
        ],
        "known_targets": {
            "SOD1": "tofersen (Qalsody, ASO — approved 2023)",
        },
        "mechanism": "antisense oligonucleotide / neuroprotection",
        "current_therapy": "mono (tofersen for SOD1-ALS; riluzole for general)",
    },

    "charcot_marie_tooth": {
        "label": "Charcot-Marie-Tooth Disease",
        "proteins": [
            "PMP22", "MPZ", "GJB1", "MFN2", "GDAP1", "SH3TC2",
            "NEFL", "HSPB1", "HSPB8", "LMNA", "PRX", "EGR2",
            "SOX10", "DNM2", "FIG4", "MTMR2", "MTMR13", "LITAF",
            "RAB7A", "TRPV4", "GARS1", "YARS1", "AARS1", "MARS1",
            "HINT1", "IGHMBP2", "DYNC1H1", "KIF1B",
        ],
        "known_targets": {},
        "mechanism": "none approved",
        "current_therapy": "none (supportive care only)",
    },
}


def download_rare_disease_data(diseases=None):
    """Download STRING data for rare diseases not yet in data/."""
    sys.path.insert(0, ROOT)
    from download_string import resolve_identifiers, download_network

    targets = diseases or list(RARE_DISEASES.keys())
    for key in targets:
        if key not in RARE_DISEASES:
            print(f"  unknown disease: {key}")
            continue
        outpath = os.path.join(DATA_DIR, f"{key}.tsv")
        if os.path.exists(outpath):
            print(f"  {key}: already downloaded")
            continue
        d = RARE_DISEASES[key]
        print(f"  downloading {key} ({d['label']}, {len(d['proteins'])} proteins)...")
        try:
            download_network(d["proteins"], outpath)
            print(f"    -> {outpath}")
        except Exception as e:
            print(f"    FAILED: {e}")
        import time
        time.sleep(1.5)


def analyze_rare_diseases(include_existing=False):
    """Run Lapras analysis on all rare diseases with STRING data."""
    rows = []
    all_diseases = dict(RARE_DISEASES)

    if include_existing:
        from diseases import DISEASES
        for k, v in DISEASES.items():
            if k not in all_diseases:
                all_diseases[k] = v
                all_diseases[k].setdefault("mechanism", "")
                all_diseases[k].setdefault("current_therapy", "")

    sys.path.insert(0, ROOT)
    from batch import load_string_network

    for key, info in sorted(all_diseases.items()):
        tsv = os.path.join(DATA_DIR, f"{key}.tsv")
        if not os.path.exists(tsv):
            continue
        try:
            C, names, _, label = load_disease(key, 0.9)
        except (SystemExit, Exception):
            try:
                known = info.get("known_targets", {})
                C, names, _ = load_string_network(tsv, known, min_score=0.9)
                label = info.get("label", key)
            except Exception as e:
                print(f"  {key}: load failed ({e})")
                continue

        if len(names) < 5 or not (C > 0).any():
            print(f"  {key}: too sparse ({len(names)} nodes)")
            continue

        try:
            an = analyze_network(C, names)
        except Exception as e:
            print(f"  {key}: analysis failed ({e})")
            continue

        try:
            u = d_uncertainty(C, names, n=50)
            p_global = u["p_global"]
            robust = u["verdict"]
        except Exception:
            p_global = float("nan")
            robust = "?"

        edges = int((C > 0).sum()) // 2
        rows.append({
            "key": key,
            "label": info.get("label", label),
            "D": an.D,
            "verdict": an.verdict,
            "p_global": p_global,
            "robust": robust,
            "n": len(names),
            "edges": edges,
            "tc": an.tc,
            "mechanism": info.get("mechanism", ""),
            "current_therapy": info.get("current_therapy", ""),
            "top5": [an.names[i] for i in an.ranked()[:5]],
            "is_rare": key in RARE_DISEASES,
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true",
                    help="download STRING data for rare diseases first")
    ap.add_argument("--full", action="store_true",
                    help="include already-analyzed diseases for comparison")
    ap.add_argument("--diseases", nargs="*",
                    help="specific diseases to analyze")
    a = ap.parse_args()

    if a.download:
        print("Downloading STRING data for rare diseases...")
        download_rare_disease_data(a.diseases)
        print()

    print("=" * 90)
    print("RARE DISEASE TRIAGE: which diseases are most likely to respond to single-target therapy?")
    print("=" * 90)

    rows = analyze_rare_diseases(include_existing=a.full)
    if not rows:
        print("\nNo diseases analyzed. Run with --download first to fetch STRING data.")
        return

    rows.sort(key=lambda r: r["D"])

    modular = [r for r in rows if r["D"] <= LINE]
    global_ = [r for r in rows if r["D"] > LINE]

    print(f"\n  Analyzed: {len(rows)} diseases")
    print(f"  MODULAR (D <= {LINE}): {len(modular)} — predicted to respond to single-target therapy")
    print(f"  GLOBAL  (D >  {LINE}): {len(global_)} — predicted to need combinations")

    print(f"\n  {'Disease':45s} {'D':>5} {'verdict':>8} {'robust':>8} {'n':>4} {'edges':>6}  top targets")
    print("  " + "-" * 100)

    for r in rows:
        marker = ""
        if r["is_rare"]:
            marker = " [RARE]"
        if r["D"] <= LINE:
            marker += " ** MODULAR — single-target candidate **"
        top = ", ".join(r["top5"][:3])
        print(f"  {r['label'][:45]:45s} {r['D']:5.2f} {r['verdict']:>8} {r['robust']:>8} "
              f"{r['n']:4d} {r['edges']:6d}  {top}{marker}")

    if modular:
        rare_modular = [r for r in modular if r["is_rare"]]
        if rare_modular:
            print(f"\n{'=' * 90}")
            print("PRIORITY RARE DISEASES FOR SINGLE-TARGET DRUG DEVELOPMENT")
            print("=" * 90)
            for r in rare_modular:
                print(f"\n  {r['label']}")
                print(f"    D = {r['D']:.2f} ({r['robust']}), "
                      f"{r['n']} proteins, {r['edges']} edges")
                print(f"    Current therapy: {r['current_therapy'] or 'none known'}")
                print(f"    Top targets by coupling bill: {', '.join(r['top5'])}")
                if r['mechanism']:
                    print(f"    Mechanism: {r['mechanism']}")

    # actionable summary
    print(f"\n{'=' * 90}")
    print("TRIAGE SUMMARY")
    print("=" * 90)

    no_therapy = [r for r in rows if r["is_rare"]
                  and (not r["current_therapy"] or "none" in r["current_therapy"].lower())]
    if no_therapy:
        print(f"\n  Rare diseases with NO approved therapy:")
        for r in sorted(no_therapy, key=lambda r: r["D"]):
            pred = "MODULAR (single-target candidate)" if r["D"] <= LINE else "GLOBAL (combination likely needed)"
            print(f"    {r['label']:45s} D={r['D']:.2f}  {pred}")

    print(f"\n  Diseases where the pipeline suggests a DIFFERENT approach than current standard:")
    for r in rows:
        if not r["is_rare"]:
            continue
        ct = r.get("current_therapy", "")
        if not ct:
            continue
        is_mono = "mono" in ct.lower() or "surgery" in ct.lower()
        if is_mono and r["D"] > LINE:
            print(f"    {r['label']:45s} D={r['D']:.2f} GLOBAL but current = {ct}")
        elif not is_mono and r["D"] <= LINE and "combo" in ct.lower():
            print(f"    {r['label']:45s} D={r['D']:.2f} MODULAR but current = {ct}")


if __name__ == "__main__":
    main()
