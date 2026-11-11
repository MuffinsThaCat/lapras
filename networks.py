"""Disease protein interaction networks.

Each function returns (C, names, known_targets) where:
  C            — n x n symmetric coupling matrix (weights 0-1)
  names        — list of n protein names
  known_targets — dict mapping protein name → approved/in-development drug info
"""
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_string_aging():
    """STRING aging network (97 proteins, real interaction data)."""
    path = os.path.join(DATA_DIR, "string_aging_full.tsv")
    edges = []
    proteins = set()
    with open(path) as f:
        f.readline()
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) >= 6:
                a, b = cols[2], cols[3]
                w = float(cols[5])
                proteins.add(a)
                proteins.add(b)
                edges.append((a, b, w))
    prot_list = sorted(proteins)
    idx = {p: i for i, p in enumerate(prot_list)}
    n = len(prot_list)
    C = np.zeros((n, n))
    for a, b, w in edges:
        i, j = idx[a], idx[b]
        C[i, j] = C[j, i] = w
    known_targets = {
        "TP53":   "p53 reactivators (APR-246)",
        "AKT1":   "AKT inhibitors (capivasertib, ipatasertib)",
        "MTOR":   "mTOR inhibitors (rapamycin, everolimus)",
        "PIK3CA": "PI3K inhibitors (alpelisib, idelalisib)",
        "SIRT1":  "SIRT1 activators (resveratrol, SRT1720)",
        "PTEN":   "no direct drug (tumor suppressor)",
        "BCL2":   "BCL2 inhibitors (venetoclax)",
        "EGFR":   "EGFR inhibitors (erlotinib, osimertinib)",
        "NFKB1":  "NF-kB pathway inhibitors (bortezomib indirectly)",
        "TNF":    "TNF inhibitors (infliximab, adalimumab)",
        "IL6":    "IL-6 inhibitors (tocilizumab, siltuximab)",
        "IGF1":   "IGF1R inhibitors (linsitinib)",
    }
    return C, prot_list, known_targets


def construct_cancer_network():
    """Cancer signaling network (65 proteins, 7 pathways)."""
    proteins = [
        "EGFR", "ERBB2", "ERBB3", "GRB2", "SOS1",
        "KRAS", "NRAS", "HRAS", "BRAF", "RAF1",
        "MAP2K1", "MAP2K2", "MAPK1", "MAPK3",
        "PIK3CA", "PIK3CB", "AKT1", "AKT2", "MTOR",
        "RPTOR", "RICTOR", "RPS6KB1", "PTEN", "TSC2",
        "TP53", "MDM2", "MDM4", "CDKN2A", "CDK4",
        "CDK6", "RB1", "CCND1", "CCNE1", "E2F1",
        "CTNNB1", "APC", "GSK3B", "AXIN1",
        "NOTCH1", "NOTCH2", "DLL1", "JAG1", "RBPJ",
        "HES1",
        "BRCA1", "BRCA2", "ATM", "ATR", "CHEK1",
        "CHEK2", "RAD51",
        "BCL2", "BAX", "BAK1", "CYCS", "CASP9",
        "CASP3",
        "WNT1", "FZD1", "DVL1", "LRP6",
        "MYC", "VEGFA", "HIF1A", "STAT3",
    ]
    n = len(proteins)
    idx = {p: i for i, p in enumerate(proteins)}
    C = np.zeros((n, n))

    def link(a, b, w):
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            C[i, j] = C[j, i] = max(C[i, j], w)

    # RTK/RAS
    for a, b in [("EGFR","GRB2"),("ERBB2","GRB2"),("ERBB3","GRB2"),
                 ("GRB2","SOS1"),("SOS1","KRAS"),("SOS1","NRAS"),("SOS1","HRAS"),
                 ("KRAS","BRAF"),("KRAS","RAF1"),("NRAS","BRAF"),
                 ("BRAF","MAP2K1"),("RAF1","MAP2K1"),("MAP2K1","MAPK1"),
                 ("MAP2K1","MAPK3"),("MAP2K2","MAPK1"),("MAP2K2","MAPK3"),
                 ("MAPK1","MYC"),("MAPK3","MYC")]:
        link(a, b, 0.9)
    # PI3K/AKT/mTOR
    for a, b in [("PIK3CA","AKT1"),("PIK3CB","AKT1"),("AKT1","MTOR"),
                 ("AKT1","TSC2"),("AKT2","MTOR"),("PTEN","PIK3CA"),
                 ("PTEN","PIK3CB"),("MTOR","RPTOR"),("MTOR","RICTOR"),
                 ("MTOR","RPS6KB1"),("AKT1","GSK3B"),("AKT1","MDM2")]:
        link(a, b, 0.9)
    # p53/apoptosis
    for a, b in [("TP53","MDM2"),("TP53","MDM4"),("MDM2","MDM4"),
                 ("TP53","BAX"),("TP53","CDKN2A"),("TP53","CHEK1"),
                 ("TP53","CHEK2"),("BCL2","BAX"),("BCL2","BAK1"),
                 ("BAX","CYCS"),("CYCS","CASP9"),("CASP9","CASP3")]:
        link(a, b, 0.9)
    # Rb/cell-cycle
    for a, b in [("CDKN2A","CDK4"),("CDKN2A","CDK6"),("CDK4","RB1"),
                 ("CDK6","RB1"),("RB1","E2F1"),("CCND1","CDK4"),
                 ("CCNE1","RB1"),("E2F1","MYC"),("MDM2","CDKN2A")]:
        link(a, b, 0.9)
    # Wnt
    for a, b in [("WNT1","FZD1"),("FZD1","DVL1"),("DVL1","GSK3B"),
                 ("LRP6","FZD1"),("GSK3B","CTNNB1"),("CTNNB1","APC"),
                 ("CTNNB1","AXIN1"),("APC","GSK3B"),("CTNNB1","MYC")]:
        link(a, b, 0.9)
    # Notch
    for a, b in [("NOTCH1","DLL1"),("NOTCH1","JAG1"),("NOTCH2","DLL1"),
                 ("NOTCH1","RBPJ"),("NOTCH2","RBPJ"),("RBPJ","HES1"),
                 ("NOTCH1","MYC")]:
        link(a, b, 0.9)
    # DNA damage
    for a, b in [("BRCA1","BRCA2"),("ATM","CHEK2"),("ATR","CHEK1"),
                 ("BRCA1","RAD51"),("BRCA2","RAD51"),("ATM","TP53"),
                 ("ATM","BRCA1"),("CHEK2","TP53")]:
        link(a, b, 0.9)
    # Cross-pathway
    cross = [
        ("KRAS","PIK3CA",0.7),("MAPK1","CCND1",0.5),("AKT1","BAX",0.5),
        ("VEGFA","HIF1A",0.9),("HIF1A","MYC",0.5),("STAT3","MYC",0.7),
        ("STAT3","BCL2",0.5),("EGFR","STAT3",0.7),("NOTCH1","VEGFA",0.5),
        ("TP53","BCL2",0.7),("MTOR","HIF1A",0.5),("PTEN","AKT2",0.7),
        ("HRAS","RAF1",0.7),("MAP2K2","MAP2K1",0.5),
    ]
    for a, b, w in cross:
        link(a, b, w)

    known_targets = {
        "EGFR":   "EGFR inhibitors (erlotinib, osimertinib)",
        "ERBB2":  "HER2 inhibitors (trastuzumab, pertuzumab, lapatinib)",
        "KRAS":   "KRAS G12C inhibitors (sotorasib, adagrasib)",
        "BRAF":   "BRAF inhibitors (vemurafenib, dabrafenib)",
        "MAP2K1": "MEK inhibitors (trametinib, cobimetinib)",
        "MAPK1":  "ERK inhibitors (ulixertinib)",
        "PIK3CA": "PI3K inhibitors (alpelisib, idelalisib, copanlisib)",
        "AKT1":   "AKT inhibitors (capivasertib, ipatasertib)",
        "MTOR":   "mTOR inhibitors (everolimus, temsirolimus)",
        "PTEN":   "no direct drug (tumor suppressor)",
        "TP53":   "p53 reactivators (APR-246/eprenetapopt); MDM2 inhibitors",
        "MDM2":   "MDM2 inhibitors (nutlin-3a, idasanutsan, navtemadlin)",
        "CDK4":   "CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib)",
        "CDK6":   "CDK4/6 inhibitors (palbociclib, ribociclib)",
        "RB1":    "loss-of-function; CDK4/6 inhibitors bypass",
        "BCL2":   "BCL2 inhibitors (venetoclax)",
        "NOTCH1": "gamma-secretase inhibitors (nirogacestat); anti-Notch antibodies",
        "CTNNB1": "beta-catenin inhibitors (PRI-724, in development)",
        "BRCA1":  "synthetic lethality with PARP inhibitors",
        "BRCA2":  "synthetic lethality with PARP inhibitors",
        "ATR":    "ATR inhibitors (berzosertib, ceralasertib)",
        "CHEK1":  "CHK1 inhibitors (prexasertib)",
        "VEGFA":  "anti-VEGF (bevacizumab)",
        "STAT3":  "STAT3 inhibitors (in development)",
        "HIF1A":  "HIF inhibitors (belzutifan — VHL disease)",
    }
    return C, proteins, known_targets


def construct_alzheimer_network():
    """Alzheimer's disease network (45 proteins, 6 pathways)."""
    proteins = [
        "APP", "BACE1", "PSEN1", "PSEN2", "NCSTN", "APH1A",
        "ADAM10", "ADAM17", "IDE", "NEP",
        "MAPT", "GSK3B", "CDK5", "PP2A", "DYRK1A",
        "FKBP52", "PIN1", "MARK4",
        "TREM2", "CD33", "TYROBP", "SYK", "IL1B",
        "NLRP3", "CASP1", "NFKB1",
        "SYP", "DLG4", "GRIN2B", "GRIA1", "CREB1",
        "CAMK2A", "BDNF", "NTRK2",
        "CHRNA7", "CHRNA4", "CHAT", "ACHE",
        "APOE", "CLU", "ABCA7", "BIN1", "PICALM",
        "SORL1", "LRP1",
    ]
    n = len(proteins)
    idx = {p: i for i, p in enumerate(proteins)}
    C = np.zeros((n, n))

    def link(a, b, w):
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            C[i, j] = C[j, i] = max(C[i, j], w)

    # Amyloid
    for a, b in [("APP","BACE1"),("APP","PSEN1"),("PSEN1","PSEN2"),
                 ("PSEN1","NCSTN"),("PSEN1","APH1A"),("APP","ADAM10"),
                 ("APP","ADAM17"),("APP","IDE"),("APP","NEP"),
                 ("APP","LRP1"),("BACE1","NCSTN")]:
        link(a, b, 0.9)
    # Tau
    for a, b in [("MAPT","GSK3B"),("MAPT","CDK5"),("MAPT","PP2A"),
                 ("MAPT","DYRK1A"),("MAPT","FKBP52"),("MAPT","PIN1"),
                 ("MAPT","MARK4"),("GSK3B","CDK5"),("CDK5","DYRK1A"),
                 ("MAPT","SYP")]:
        link(a, b, 0.9)
    # Neuroinflammation
    for a, b in [("TREM2","TYROBP"),("TREM2","CD33"),("TYROBP","SYK"),
                 ("SYK","IL1B"),("IL1B","NLRP3"),("NLRP3","CASP1"),
                 ("IL1B","NFKB1"),("TREM2","NFKB1")]:
        link(a, b, 0.9)
    # Synaptic
    for a, b in [("SYP","DLG4"),("DLG4","GRIN2B"),("DLG4","GRIA1"),
                 ("GRIN2B","CAMK2A"),("CAMK2A","CREB1"),("CREB1","BDNF"),
                 ("BDNF","NTRK2"),("CAMK2A","CHRNA7")]:
        link(a, b, 0.9)
    # Cholinergic
    for a, b in [("CHRNA7","CHRNA4"),("CHRNA7","CHAT"),("CHAT","ACHE"),
                 ("ACHE","CHRNA4")]:
        link(a, b, 0.9)
    # Lipid/ApoE
    for a, b in [("APOE","CLU"),("APOE","ABCA7"),("APOE","LRP1"),
                 ("BIN1","PICALM"),("SORL1","APP"),("SORL1","APOE"),
                 ("LRP1","APOE")]:
        link(a, b, 0.9)
    # Cross-pathway
    cross = [
        ("APP","MAPT",0.5),("GSK3B","APP",0.5),("TREM2","APP",0.3),
        ("IL1B","MAPT",0.3),("NFKB1","BACE1",0.3),("APOE","APP",0.5),
        ("APOE","MAPT",0.3),("BIN1","MAPT",0.3),("PICALM","APP",0.3),
        ("BDNF","MAPT",0.3),("CREB1","BDNF",0.7),("CDK5","APP",0.3),
        ("NLRP3","APP",0.3),("SYK","MAPT",0.3),("GRIN2B","MAPT",0.3),
        ("CHRNA7","NFKB1",0.3),("CAMK2A","MAPT",0.3),
    ]
    for a, b, w in cross:
        link(a, b, w)

    known_targets = {
        "APP":    "anti-amyloid antibodies (lecanemab, donanemab, aducanumab)",
        "BACE1":  "BACE inhibitors (verubecestat, atabecestat — failed Phase III)",
        "PSEN1":  "gamma-secretase modulators (tarenflurbil — failed; avagacestat)",
        "MAPT":   "anti-tau antibodies (semorinemab, zagotenemab); tau aggregation inhibitors",
        "GSK3B":  "GSK3 inhibitors (tideglusib, lithium — repurposed)",
        "CDK5":   "CDK5 inhibitors (roscovitine — preclinical for AD)",
        "DYRK1A": "DYRK1A inhibitors (leucettine L41 — preclinical)",
        "TREM2":  "TREM2 agonist antibodies (AL002 — Phase II)",
        "IL1B":   "IL-1 inhibitors (canakinumab — repurposing studies)",
        "NLRP3":  "NLRP3 inhibitors (MCC950/CRID3, inzomelid — early trials)",
        "GRIN2B": "NMDA modulators (memantine — approved)",
        "CHRNA7": "alpha7 nAChR agonists (encenicline — failed Phase III)",
        "ACHE":   "acetylcholinesterase inhibitors (donepezil, rivastigmine, galantamine)",
        "BDNF":   "BDNF mimetics (7,8-DHF — preclinical)",
        "NFKB1":  "NF-kB pathway inhibition (indirect targets)",
        "APOE":   "ApoE-targeted therapies (gene therapy, antisense — early stage)",
    }
    return C, proteins, known_targets


def construct_als_network():
    """ALS network (47 proteins, 6 pathways)."""
    proteins = [
        "TARDBP", "FUS", "C9orf72", "HNRNPA1", "HNRNPA2B1",
        "MATR3", "TIA1", "ATXN2", "UPF1", "SMN1",
        "SOD1", "UBQLN2", "OPTN", "SQSTM1", "VCP",
        "CHMP2B", "TBK1", "SIGMAR1", "VAPB", "DCTN1",
        "TUBA4A", "PFN1", "KIF5A", "NEFH", "PRPH",
        "STMN2", "UNC13A",
        "GRIA2", "GRIN1", "SLC1A2", "GRM5", "GRIK2",
        "TREM2", "C3", "TLR4", "IL1B", "TNF",
        "NFKB1", "NLRP3", "CCL2",
        "CHCHD10", "CHCHD2", "SOD2", "PINK1",
        "NEK1", "ERBB4", "SRC",
    ]
    n = len(proteins)
    idx = {p: i for i, p in enumerate(proteins)}
    C = np.zeros((n, n))

    def link(a, b, w):
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            C[i, j] = C[j, i] = max(C[i, j], w)

    # RNA processing
    for a, b in [("TARDBP","FUS"),("TARDBP","HNRNPA1"),("TARDBP","HNRNPA2B1"),
                 ("FUS","HNRNPA1"),("FUS","MATR3"),("TIA1","TARDBP"),
                 ("TIA1","FUS"),("ATXN2","TARDBP"),("UPF1","TARDBP"),
                 ("SMN1","FUS"),("C9orf72","TARDBP"),("C9orf72","UPF1")]:
        link(a, b, 0.9)
    link("HNRNPA1","HNRNPA2B1",0.9)
    link("ATXN2","TIA1",0.5)
    # Protein homeostasis
    for a, b in [("SOD1","UBQLN2"),("UBQLN2","SQSTM1"),("OPTN","TBK1"),
                 ("OPTN","SQSTM1"),("VCP","UBQLN2"),("VCP","SQSTM1"),
                 ("CHMP2B","VCP"),("SIGMAR1","VAPB"),("VAPB","TARDBP"),
                 ("SOD1","SOD2"),("DCTN1","VAPB")]:
        link(a, b, 0.9)
    for a, b in [("TBK1","SQSTM1"),("VCP","OPTN")]:
        link(a, b, 0.7)
    # Cytoskeletal
    for a, b in [("TUBA4A","PFN1"),("KIF5A","NEFH"),("KIF5A","DCTN1"),
                 ("NEFH","PRPH"),("STMN2","TARDBP"),("UNC13A","TARDBP")]:
        link(a, b, 0.9)
    link("PFN1","NEFH",0.5)
    # Glutamate
    for a, b in [("GRIA2","GRIN1"),("SLC1A2","GRIA2"),("GRM5","GRIA2"),
                 ("GRIK2","GRIN1"),("SLC1A2","GRIN1")]:
        link(a, b, 0.9)
    # Neuroinflammation
    for a, b in [("TREM2","C3"),("TLR4","NFKB1"),("NFKB1","TNF"),
                 ("NFKB1","IL1B"),("IL1B","NLRP3"),("NLRP3","C3"),
                 ("TNF","NFKB1"),("CCL2","TLR4")]:
        link(a, b, 0.9)
    link("TREM2","TLR4",0.5)
    # Mitochondrial
    for a, b in [("CHCHD10","CHCHD2"),("PINK1","SOD2"),("CHCHD10","SOD2")]:
        link(a, b, 0.9)
    # Kinases
    for a, b in [("NEK1","C9orf72"),("ERBB4","SRC")]:
        link(a, b, 0.7)
    # Cross-pathway
    cross = [
        ("TARDBP","SQSTM1",0.5),("TARDBP","STMN2",0.7),("TARDBP","UNC13A",0.7),
        ("SOD1","SLC1A2",0.4),("C9orf72","SQSTM1",0.5),("C9orf72","NLRP3",0.4),
        ("FUS","SOD2",0.3),("TARDBP","CHCHD10",0.3),("VCP","CHCHD10",0.4),
        ("TBK1","NFKB1",0.5),("SOD1","TNF",0.3),("OPTN","NFKB1",0.5),
        ("NEK1","TUBA4A",0.4),("ERBB4","GRIA2",0.3),("SRC","GRIN1",0.3),
        ("NEFH","SOD1",0.3),
    ]
    for a, b, w in cross:
        link(a, b, w)

    known_targets = {
        "SLC1A2":  "riluzole (approved)",
        "SOD1":    "tofersen (approved — antisense oligonucleotide)",
        "GRIN1":   "memantine (off-label NMDA antagonist)",
        "TNF":     "TNF inhibitors (epidemiological signal only)",
        "C9orf72": "antisense oligos in trials (WVE-004 discontinued)",
        "FUS":     "no approved drug (target for antisense)",
        "TARDBP":  "no approved drug (TDP-43 — central target, undruggable so far)",
        "ATXN2":   "antisense oligos (BIIB105/ION541 in Phase I)",
        "STMN2":   "no drug (stathmin-2 restoration is a leading strategy)",
        "UNC13A":  "no drug (UNC13A cryptic exon is a leading strategy)",
    }
    return C, proteins, known_targets


def construct_pancreatic_cancer_network():
    """Pancreatic ductal adenocarcinoma network (59 proteins, 8 pathways)."""
    proteins = [
        "KRAS", "BRAF", "MAP2K1", "MAPK1", "MAPK3",
        "RAF1", "SOS1", "NF1",
        "PIK3CA", "AKT1", "MTOR", "PTEN", "TSC2",
        "TP53", "MDM2", "CDKN2A", "CDK4", "CDK6",
        "RB1", "CCND1", "MYC",
        "TGFBR1", "TGFBR2", "SMAD4", "SMAD2", "SMAD3",
        "SHH", "PTCH1", "SMO", "GLI1",
        "CTNNB1", "APC", "GSK3B", "AXIN1", "RNF43",
        "BRCA1", "BRCA2", "ATM", "PALB2", "CHEK2",
        "FAP", "ACTA2", "COL1A1", "HAS2", "MMP9",
        "SPARC", "FN1",
        "CD274", "CTLA4", "LAG3", "TIGIT", "IDO1",
        "FOXP3", "IL10", "TGFB1",
        "HIF1A", "LDHA", "ATG5", "BECN1",
    ]
    n = len(proteins)
    idx = {p: i for i, p in enumerate(proteins)}
    C = np.zeros((n, n))

    def link(a, b, w):
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            C[i, j] = C[j, i] = max(C[i, j], w)

    # KRAS/MAPK
    for a, b in [("KRAS","BRAF"),("KRAS","RAF1"),("BRAF","MAP2K1"),
                 ("RAF1","MAP2K1"),("MAP2K1","MAPK1"),("MAP2K1","MAPK3"),
                 ("SOS1","KRAS"),("NF1","KRAS")]:
        link(a, b, 0.9)
    link("KRAS","MAPK1",0.5)
    # PI3K/mTOR
    for a, b in [("PIK3CA","AKT1"),("AKT1","MTOR"),("PTEN","PIK3CA"),
                 ("AKT1","TSC2"),("TSC2","MTOR")]:
        link(a, b, 0.9)
    # TP53 / cell cycle
    for a, b in [("TP53","MDM2"),("CDKN2A","CDK4"),("CDKN2A","CDK6"),
                 ("CDK4","RB1"),("CDK6","RB1"),("CCND1","CDK4"),
                 ("RB1","MYC"),("TP53","CDKN2A")]:
        link(a, b, 0.9)
    # TGF-beta
    for a, b in [("TGFBR1","TGFBR2"),("TGFBR1","SMAD2"),("TGFBR1","SMAD3"),
                 ("SMAD2","SMAD4"),("SMAD3","SMAD4"),("SMAD2","SMAD3")]:
        link(a, b, 0.9)
    # Hedgehog
    for a, b in [("SHH","PTCH1"),("PTCH1","SMO"),("SMO","GLI1")]:
        link(a, b, 0.9)
    # Wnt
    for a, b in [("CTNNB1","APC"),("CTNNB1","GSK3B"),("CTNNB1","AXIN1"),
                 ("GSK3B","AXIN1"),("RNF43","CTNNB1")]:
        link(a, b, 0.9)
    link("APC","GSK3B",0.5)
    # DNA damage
    for a, b in [("BRCA1","BRCA2"),("BRCA1","PALB2"),("BRCA2","PALB2"),
                 ("ATM","BRCA1"),("ATM","CHEK2"),("CHEK2","TP53")]:
        link(a, b, 0.9)
    # Stroma
    for a, b in [("FAP","ACTA2"),("ACTA2","COL1A1"),("COL1A1","FN1"),
                 ("HAS2","MMP9"),("SPARC","COL1A1"),("FAP","FN1")]:
        link(a, b, 0.9)
    link("MMP9","FN1",0.5)
    # Immune evasion
    for a, b in [("CD274","CTLA4"),("LAG3","TIGIT"),("IDO1","FOXP3"),
                 ("FOXP3","IL10"),("TGFB1","FOXP3"),("TGFB1","IL10")]:
        link(a, b, 0.9)
    link("CD274","LAG3",0.5)
    # Metabolism
    for a, b in [("HIF1A","LDHA"),("ATG5","BECN1")]:
        link(a, b, 0.9)
    # Cross-pathway
    cross = [
        ("KRAS","PIK3CA",0.7),("KRAS","MYC",0.5),("MAPK1","MYC",0.5),
        ("AKT1","MDM2",0.5),("SMAD4","MYC",0.4),("TGFB1","SMAD2",0.7),
        ("TGFB1","ACTA2",0.6),("TGFB1","CD274",0.4),("SHH","FAP",0.5),
        ("GLI1","CCND1",0.4),("MYC","HIF1A",0.5),("HIF1A","CD274",0.4),
        ("MMP9","TGFB1",0.4),("CTNNB1","MYC",0.5),("KRAS","GLI1",0.3),
        ("ATM","KRAS",0.3),("SPARC","PIK3CA",0.3),("FAP","CD274",0.3),
        ("MTOR","HIF1A",0.5),("BECN1","TP53",0.3),("AKT1","GSK3B",0.5),
        ("MAPK1","CCND1",0.4),
    ]
    for a, b, w in cross:
        link(a, b, w)

    known_targets = {
        "KRAS":   "sotorasib/adagrasib (G12C only — ~1% of PDAC); G12D inhibitors in trials",
        "MAP2K1": "trametinib (minimal single-agent activity in PDAC)",
        "MTOR":   "everolimus (failed in PDAC trials)",
        "CDK4":   "palbociclib (limited activity in PDAC)",
        "BRCA1":  "olaparib (approved for BRCA-mutant PDAC — ~5-7%)",
        "BRCA2":  "olaparib (approved for BRCA-mutant PDAC)",
        "CD274":  "pembrolizumab (MSI-high only — ~1% of PDAC)",
        "SMO":    "vismodegib (failed in PDAC — accelerated death)",
        "ATM":    "ATR inhibitors in early trials",
    }
    return C, proteins, known_targets
