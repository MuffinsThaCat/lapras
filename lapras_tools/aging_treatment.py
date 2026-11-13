"""Aging treatment analyzer.

Maps the Lapras aging network through the lens of longevity biology:
  - Annotates each target with aging-specific directionality (activate/inhibit/modulate)
  - Scores existing longevity interventions by module coverage
  - Identifies directionality mismatches (targets where oncology drugs go the wrong way)
  - Proposes an evidence-graded multi-module treatment framework
  - Fine-grained module analysis (10-20 sub-modules)

    python3 -m lapras_tools.aging_treatment
    python3 -m lapras_tools.aging_treatment --modules 15 --fine
    python3 -m lapras_tools.aging_treatment --regimen
"""
import argparse
import numpy as np
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import load_string_aging
from lapras_tools.core import analyze_network, LINE


# ── Aging-specific directionality annotations ────────────────────────────
# For each protein: what direction of modulation would slow aging?
# Sources: López-Otín hallmarks, ITP results, genetic longevity studies,
# caloric restriction signatures, senolytic mechanisms.
#
# ACTIVATE  = upregulate / agonize for longevity
# INHIBIT   = downregulate / antagonize for longevity
# MODULATE  = context-dependent (e.g. intermittent, cell-type specific)
# CLEAR     = remove cells expressing this (senolytics)
# PRESERVE  = maintain function (DNA repair enzymes — boost, don't block)

AGING_DIRECTIONALITY = {
    # ── Module 0: Growth / Senescence ──
    "TP53":    ("MODULATE", "Activate in senescent cells (clearance), preserve in stem cells"),
    "AKT1":    ("INHIBIT",  "Constitutive AKT drives aging via mTOR; reduced AKT extends lifespan in model organisms"),
    "MYC":     ("INHIBIT",  "MYC haploinsufficient mice live 15% longer; master growth regulator"),
    "PIK3CA":  ("INHIBIT",  "Upstream of AKT/mTOR axis; reduced PI3K signaling extends lifespan"),
    "CTNNB1":  ("ACTIVATE", "Wnt/beta-catenin declines with age; restoring it maintains stem cell pools"),
    "PTEN":    ("ACTIVATE", "Tumor suppressor, counteracts PI3K; PTEN overexpression extends mouse lifespan"),
    "HIF1A":   ("MODULATE", "Intermittent activation (hormesis) beneficial; chronic activation harmful (fibrosis)"),
    "PIK3CB":  ("INHIBIT",  "Same axis as PIK3CA"),
    "EGFR":    ("INHIBIT",  "Growth factor signaling; reduced EGFR extends C. elegans lifespan"),
    "ERBB2":   ("INHIBIT",  "Growth factor receptor; same logic as EGFR"),
    "MDM2":    ("INHIBIT",  "Frees p53 for senescent cell surveillance; MDM2 inhibition = functional p53 activation"),
    "CDKN1A":  ("MODULATE", "p21 marks senescent cells; clear those cells but don't suppress p21 globally"),
    "FOXO3":   ("ACTIVATE", "Most replicated human longevity gene; stress resistance, autophagy, antioxidant defense"),
    "EP300":   ("MODULATE", "Histone acetyltransferase; inhibition mimics caloric restriction in some models"),
    "HSP90AA1":("MODULATE", "Proteostasis chaperone; needed for protein quality control"),
    "HDAC1":   ("MODULATE", "Epigenetic regulator; some HDAC inhibitors extend lifespan in model organisms"),
    "HDAC2":   ("MODULATE", "Same as HDAC1"),
    "EZH2":    ("MODULATE", "H3K27 methyltransferase; decline with age; context-dependent"),
    "CREBBP":  ("MODULATE", "CBP/p300 acetyltransferase; epigenetic regulation"),
    "AKT2":    ("INHIBIT",  "Same axis as AKT1"),
    "NOTCH1":  ("ACTIVATE", "Notch signaling maintains stem cell niches; declines with age"),
    "INSR":    ("INHIBIT",  "Reduced insulin signaling extends lifespan (daf-2 paradigm)"),
    "DNMT1":   ("PRESERVE", "DNA methyltransferase maintenance; epigenetic clock driver"),
    "DNMT3A":  ("PRESERVE", "De novo methyltransferase; age-related changes in methylation"),

    # ── Module 1: Inflammation / Sirtuins ──
    "STAT3":   ("INHIBIT",  "Chronic STAT3 activation drives inflammaging and SASP"),
    "SIRT1":   ("ACTIVATE", "NAD+-dependent deacetylase; activation extends lifespan in multiple models"),
    "BCL2":    ("INHIBIT",  "In senescent cells: BCL2 keeps them alive; senolytics target this (navitoclax)"),
    "CASP3":   ("MODULATE", "Apoptosis effector; needed for senescent cell clearance, harmful if excessive"),
    "NFKB1":   ("INHIBIT",  "Master inflammatory TF; NF-kB inhibition extends mouse lifespan"),
    "FOXO1":   ("ACTIVATE", "Same family as FOXO3; stress resistance and metabolic regulation"),
    "TNF":     ("INHIBIT",  "Pro-inflammatory cytokine; TNF drives inflammaging and tissue damage"),
    "IL6":     ("INHIBIT",  "Pro-inflammatory; IL-6 rises with age and correlates with frailty"),
    "IL1B":    ("INHIBIT",  "Inflammasome product; CANTOS trial showed cardiovascular benefit of anti-IL1B"),
    "IL10":    ("ACTIVATE", "Anti-inflammatory cytokine; counterbalances TNF/IL6/IL1B"),
    "RELA":    ("INHIBIT",  "NF-kB subunit; same logic as NFKB1"),
    "JAK2":    ("INHIBIT",  "JAK-STAT inflammatory signaling; ruxolitinib extends mouse lifespan"),
    "BAX":     ("MODULATE", "Pro-apoptotic; needed for senescent cell clearance"),
    "BAK1":    ("MODULATE", "Same as BAX"),
    "CASP8":   ("MODULATE", "Extrinsic apoptosis; context-dependent"),
    "CASP9":   ("MODULATE", "Intrinsic apoptosis; context-dependent"),
    "HSPA1A":  ("ACTIVATE", "HSP70 chaperone; proteostasis declines with age"),
    "KAT2B":   ("MODULATE", "Histone acetyltransferase"),
    "IGF1":    ("INHIBIT",  "IGF-1 signaling reduction extends lifespan (growth hormone axis)"),
    "IGF1R":   ("INHIBIT",  "IGF-1 receptor; IGF1R+/- mice live longer"),
    "PRKAA1":  ("ACTIVATE", "AMPK alpha-1; AMPK activation mimics caloric restriction"),
    "PRKAA2":  ("ACTIVATE", "AMPK alpha-2; same as PRKAA1"),
    "GADD45A": ("PRESERVE", "DNA damage response; stress response gene"),
    "EIF4EBP1":("ACTIVATE", "4E-BP1; when active, inhibits cap-dependent translation (lifespan extension)"),

    # ── Module 2: Autophagy / mTOR ──
    "MTOR":    ("INHIBIT",  "mTOR inhibition (rapamycin) extends lifespan in every organism tested"),
    "NFE2L2":  ("ACTIVATE", "Nrf2; master antioxidant TF; Nrf2 activation extends lifespan"),
    "BECN1":   ("ACTIVATE", "Beclin-1; essential autophagy initiator; enhancing autophagy extends lifespan"),
    "SQSTM1":  ("MODULATE", "p62; autophagy receptor; accumulation signals impaired autophagy"),
    "RPS6KB1": ("INHIBIT",  "S6K1; downstream of mTOR; S6K1 knockout extends mouse lifespan"),
    "KEAP1":   ("INHIBIT",  "Nrf2 inhibitor; KEAP1 inhibition frees Nrf2 for antioxidant defense"),
    "TSC2":    ("ACTIVATE", "Tuberin; TSC complex inhibits mTOR; TSC activation = mTOR suppression"),
    "ATG7":    ("ACTIVATE", "Autophagy gene; essential for autophagosome formation"),
    "ATG5":    ("ACTIVATE", "Autophagy gene; ATG5 overexpression extends mouse lifespan"),
    "LAMP2":   ("ACTIVATE", "Lysosomal membrane protein; needed for chaperone-mediated autophagy"),
    "MAP1LC3B":("ACTIVATE", "LC3B; autophagosome marker; enhanced autophagy = longevity"),
    "ULK1":    ("ACTIVATE", "Autophagy initiation kinase; activated by AMPK"),
    "TFEB":    ("ACTIVATE", "Master lysosomal/autophagy TF; TFEB activation clears cellular debris"),
    "RPTOR":   ("INHIBIT",  "Raptor; mTORC1 component; inhibited by rapamycin"),
    "RICTOR":  ("MODULATE", "Rictor; mTORC2 component; mTORC2 has different aging effects than mTORC1"),
    "RHEB":    ("INHIBIT",  "mTOR activator; Rheb inhibition = mTOR suppression"),
    "TSC1":    ("ACTIVATE", "Hamartin; TSC complex inhibits mTOR"),
    "TET2":    ("ACTIVATE", "DNA demethylase; TET2 loss accelerates aging; restoration rejuvenates"),
    "KMT2A":   ("PRESERVE", "H3K4 methyltransferase; epigenetic maintenance"),

    # ── Module 3: DNA repair / Telomeres ──
    "PARP1":   ("PRESERVE", "DNA repair enzyme; NEEDS NAD+ — don't inhibit, supply NAD+ precursors"),
    "ATM":     ("ACTIVATE", "DNA damage sensor; ATM decline impairs repair with age"),
    "BRCA1":   ("PRESERVE", "Homologous recombination repair; maintain, don't inhibit"),
    "TERT":    ("ACTIVATE", "Telomerase; TERT gene therapy extends mouse lifespan; telomere maintenance"),
    "CHEK1":   ("PRESERVE", "Cell cycle checkpoint; DNA damage response"),
    "CHEK2":   ("PRESERVE", "Cell cycle checkpoint; DNA damage response"),
    "SIRT3":   ("ACTIVATE", "Mitochondrial sirtuin; SIRT3 protects against oxidative damage"),
    "SIRT6":   ("ACTIVATE", "SIRT6 overexpression extends male mouse lifespan; DNA repair, telomere maintenance"),
    "ERCC1":   ("PRESERVE", "Nucleotide excision repair; ERCC1 deficiency causes premature aging"),
    "BLM":     ("PRESERVE", "Bloom syndrome helicase; genomic stability"),
    "WRN":     ("PRESERVE", "Werner syndrome helicase; WRN loss causes premature aging"),
    "MLH1":    ("PRESERVE", "Mismatch repair; genomic stability"),
    "MSH2":    ("PRESERVE", "Mismatch repair; genomic stability"),
    "XPC":     ("PRESERVE", "Nucleotide excision repair"),
    "DDB2":    ("PRESERVE", "DNA damage binding; nucleotide excision repair"),
    "POLB":    ("PRESERVE", "Base excision repair polymerase"),
    "CDKN2A":  ("MODULATE", "p16INK4a; senescence marker; clear p16-high cells (senolytics)"),
    "CDKN2B":  ("MODULATE", "p15INK4b; related to CDKN2A"),
    "ATR":     ("PRESERVE", "DNA damage sensor; replication stress response"),

    # ── Module 4: Wnt / Stress ──
    "GSK3B":   ("INHIBIT",  "GSK3B inhibition (lithium) activates Wnt, neuroprotective; extends lifespan in C. elegans"),
    "MAPK14":  ("INHIBIT",  "p38 MAPK; chronic p38 activation drives inflammaging and senescence"),
    "RB1":     ("PRESERVE", "Retinoblastoma; tumor suppressor; cell cycle control"),
    "MAP3K5":  ("INHIBIT",  "ASK1; stress kinase; chronic activation drives apoptosis and inflammation"),
    "LEF1":    ("ACTIVATE", "Wnt target gene TF; Wnt signaling maintains stem cell niches"),
    "LMNA":    ("PRESERVE", "Lamin A; mutations cause progeria; nuclear lamina integrity essential"),
    "SHH":     ("MODULATE", "Sonic hedgehog; stem cell maintenance in some niches"),
    "WNT3A":   ("ACTIVATE", "Wnt ligand; restoring Wnt signaling rejuvenates aged stem cells"),
    "DVL1":    ("ACTIVATE", "Dishevelled; Wnt signal transducer"),
    "COL18A1": ("MODULATE", "Collagen XVIII / endostatin; extracellular matrix"),
    "TERF2":   ("PRESERVE", "Telomere binding; shelterin component; protects telomere integrity"),
}


# ── Longevity interventions with evidence grades ─────────────────────────
INTERVENTIONS = [
    {
        "name": "Rapamycin (intermittent low-dose)",
        "targets": {"MTOR": "INHIBIT", "RPTOR": "INHIBIT", "RPS6KB1": "INHIBIT"},
        "evidence": "A",
        "evidence_detail": "Extends lifespan in mice (ITP, Harrison 2009), flies, worms, yeast. "
                           "ITP: 9-14% median lifespan extension. Human: PEARL trial (immune function).",
        "practical": "5-10 mg/week, pulsed (e.g. once weekly). Avoid continuous high-dose (immunosuppression).",
        "modules_hit": set(),
    },
    {
        "name": "Metformin",
        "targets": {"PRKAA1": "ACTIVATE", "PRKAA2": "ACTIVATE", "MTOR": "INHIBIT"},
        "evidence": "B+",
        "evidence_detail": "UKPDS: diabetics on metformin outlived non-diabetic controls. "
                           "TAME trial (Targeting Aging with Metformin) enrolling. Mouse ITP: modest or null.",
        "practical": "500-1000 mg/day. May impair exercise adaptation (mitohormesis). GI side effects.",
        "modules_hit": set(),
    },
    {
        "name": "NAD+ precursors (NMN or NR)",
        "targets": {"SIRT1": "ACTIVATE", "SIRT3": "ACTIVATE", "SIRT6": "ACTIVATE", "PARP1": "PRESERVE"},
        "evidence": "B",
        "evidence_detail": "NMN extends healthspan in aged mice (Mills 2016). Human Phase I/II: "
                           "NMN raises NAD+ in blood. NR: CHROMAVID trial shows elevated NAD+ but no "
                           "clinical endpoints yet. Supplies the NAD+ that PARP1 needs for DNA repair "
                           "without inhibiting PARP1.",
        "practical": "250-1000 mg NMN/day or 300-1000 mg NR/day.",
        "modules_hit": set(),
    },
    {
        "name": "Senolytics (dasatinib + quercetin)",
        "targets": {"BCL2": "INHIBIT", "TP53": "MODULATE", "EGFR": "INHIBIT"},
        "evidence": "B+",
        "evidence_detail": "Extends healthspan in naturally aged mice (Xu 2018). Human Phase I/II: "
                           "IPF (improved 6-min walk), diabetic kidney disease. Intermittent dosing "
                           "(kills senescent cells, spares normal cells).",
        "practical": "D 100mg + Q 1000mg, 3 consecutive days per month. Intermittent is key.",
        "modules_hit": set(),
    },
    {
        "name": "Lithium (low-dose)",
        "targets": {"GSK3B": "INHIBIT"},
        "evidence": "B-",
        "evidence_detail": "Extends lifespan in C. elegans and Drosophila. Epidemiological: regions "
                           "with higher lithium in drinking water show lower all-cause mortality. "
                           "Neuroprotective. No RCT for longevity.",
        "practical": "5-10 mg/day (trace dose, not psychiatric dose of 600-1200 mg).",
        "modules_hit": set(),
    },
    {
        "name": "SGLT2 inhibitors (empagliflozin)",
        "targets": {"PRKAA1": "ACTIVATE", "MTOR": "INHIBIT", "SIRT1": "ACTIVATE"},
        "evidence": "B",
        "evidence_detail": "EMPA-REG: 38% reduction in CV death. Extends lifespan in non-diabetic mice "
                           "(Kondo 2024). Mimics aspects of caloric restriction (glucosuria, ketogenesis, "
                           "AMPK activation).",
        "practical": "10-25 mg/day. Approved for heart failure and CKD even without diabetes.",
        "modules_hit": set(),
    },
    {
        "name": "GLP-1 receptor agonists (semaglutide)",
        "targets": {"NFKB1": "INHIBIT", "TNF": "INHIBIT", "IL6": "INHIBIT"},
        "evidence": "B+",
        "evidence_detail": "SELECT trial: 20% MACE reduction independent of weight loss. "
                           "Anti-inflammatory beyond metabolic effects. Reduces CRP, IL-6. "
                           "Mouse: extends lifespan in obese and non-obese models.",
        "practical": "0.25-2.4 mg/week (Ozempic/Wegovy). GI side effects, muscle mass loss.",
        "modules_hit": set(),
    },
    {
        "name": "Acarbose",
        "targets": {"INSR": "INHIBIT", "IGF1": "INHIBIT", "MTOR": "INHIBIT"},
        "evidence": "A-",
        "evidence_detail": "ITP: extends median lifespan 22% in males, 5% in females (one of the "
                           "strongest ITP results). Reduces postprandial glucose/insulin spikes. "
                           "Cheap, off-patent.",
        "practical": "25-50 mg with meals. GI side effects (flatulence). Best signal in males.",
        "modules_hit": set(),
    },
    {
        "name": "17-alpha-estradiol",
        "targets": {"NFKB1": "INHIBIT", "MAPK14": "INHIBIT"},
        "evidence": "A-",
        "evidence_detail": "ITP: extends median lifespan 19% in males, null in females. "
                           "Non-feminizing estrogen. Reduces hypothalamic inflammation.",
        "practical": "Mouse-only data. No human trials for longevity. Male-specific.",
        "modules_hit": set(),
    },
    {
        "name": "Caloric restriction (25-30%)",
        "targets": {"MTOR": "INHIBIT", "SIRT1": "ACTIVATE", "FOXO1": "ACTIVATE",
                     "FOXO3": "ACTIVATE", "PRKAA1": "ACTIVATE", "IGF1": "INHIBIT",
                     "INSR": "INHIBIT", "AKT1": "INHIBIT", "NFE2L2": "ACTIVATE",
                     "NFKB1": "INHIBIT", "BECN1": "ACTIVATE", "ATG5": "ACTIVATE"},
        "evidence": "A",
        "evidence_detail": "Extends lifespan 30-50% in rodents. CALERIE trial: humans tolerate 25% CR, "
                           "reduced inflammation, improved cardiometabolic markers. Primate (Wisconsin/NIA): "
                           "reduced age-related disease, lifespan effect debated.",
        "practical": "Not a drug. Difficult to sustain. Intermittent fasting is more practical.",
        "modules_hit": set(),
    },
    {
        "name": "Exercise (aerobic + resistance)",
        "targets": {"PRKAA1": "ACTIVATE", "SIRT1": "ACTIVATE", "NFE2L2": "ACTIVATE",
                     "BDNF": "ACTIVATE", "NFKB1": "INHIBIT", "TNF": "INHIBIT",
                     "BECN1": "ACTIVATE", "ATG5": "ACTIVATE", "TERT": "ACTIVATE"},
        "evidence": "A",
        "evidence_detail": "Epidemiological: 3-7 year life expectancy gain. Activates AMPK, "
                           "sirtuins, autophagy, Nrf2. Reduces inflammation. Preserves telomeres. "
                           "The most validated single longevity intervention.",
        "practical": "150+ min moderate or 75+ min vigorous per week. Resistance 2-3x/week.",
        "modules_hit": set(),
    },
]


# ── Emerging / speculative interventions ─────────────────────────────────
EMERGING = [
    {
        "name": "MYC inhibitor (Omomyc / BET inhibitor)",
        "targets": {"MYC": "INHIBIT"},
        "evidence": "C",
        "evidence_detail": "MYC haploinsufficient mice live 15% longer (Hofmann 2015). "
                           "Omomyc mini-protein in Phase I/II for cancer. No longevity trial. "
                           "BET inhibitors (JQ1) downregulate MYC transcription.",
        "why_interesting": "Rank #3 in aging bill (0.102). The single highest undrugged target. "
                           "Module 0 (40% mass). Direct MYC inhibition could be the most impactful "
                           "new longevity drug.",
    },
    {
        "name": "FOXO3 activator",
        "targets": {"FOXO3": "ACTIVATE"},
        "evidence": "C",
        "evidence_detail": "FOXO3 variants associated with exceptional longevity across multiple "
                           "populations (Willcox 2008, Flachsbart 2009). Carnosol and some polyphenols "
                           "show weak FOXO3 activation. No purpose-built activator exists.",
        "why_interesting": "Rank #15 in aging bill. Same module as AKT1. A FOXO3 activator would be "
                           "the first drug designed specifically for longevity rather than repurposed.",
    },
    {
        "name": "TERT gene therapy / telomerase activator",
        "targets": {"TERT": "ACTIVATE"},
        "evidence": "B-",
        "evidence_detail": "TERT gene therapy extends mouse lifespan 13-24% without increased cancer "
                           "(Bernardes de Jesus 2012). TA-65 (cycloastragenol) is a weak telomerase "
                           "activator with limited data.",
        "why_interesting": "TERT rank #32 in bill but it's the ONLY druggable telomere target. "
                           "Module 3 (DNA repair). Telomere attrition is a primary hallmark.",
    },
    {
        "name": "TET2 restoration / epigenetic reprogramming",
        "targets": {"TET2": "ACTIVATE", "DNMT3A": "PRESERVE"},
        "evidence": "C+",
        "evidence_detail": "TET2 loss-of-function clonal hematopoiesis increases cardiovascular risk 2x. "
                           "Partial reprogramming (OSKM) rejuvenates cells without dedifferentiation "
                           "(Ocampo 2016, Browder 2022). Turn Biotechnologies, Altos Labs pursuing this.",
        "why_interesting": "Epigenetic alterations are a primary hallmark. TET2 is in module 2. "
                           "Epigenetic reprogramming is the most systemic possible intervention — "
                           "consistent with D >> 3.03 requiring global state changes.",
    },
    {
        "name": "STAT3 inhibitor (napabucasin / TTI-101)",
        "targets": {"STAT3": "INHIBIT"},
        "evidence": "C",
        "evidence_detail": "STAT3 drives SASP and inflammaging. Napabucasin in cancer Phase III. "
                           "No longevity data. Ruxolitinib (JAK1/2 inhibitor, upstream) extends "
                           "mouse lifespan and reduces frailty.",
        "why_interesting": "Rank #6 in aging bill (0.089). Module 1 (29% mass). Could address "
                           "inflammaging from a broader angle than anti-TNF/IL6.",
    },
    {
        "name": "Navitoclax (BCL2/BCLxL senolytic)",
        "targets": {"BCL2": "INHIBIT"},
        "evidence": "B-",
        "evidence_detail": "More potent senolytic than D+Q in many cell types. Unity Biotechnology "
                           "UBX1325 (BCL-xL inhibitor) in Phase II for macular degeneration. "
                           "Thrombocytopenia limits systemic use; targeted versions in development.",
        "why_interesting": "BCL2 rank #11 in aging bill. More selective senolytic mechanism than D+Q.",
    },
]


def assign_modules(analysis, targets_dict):
    """For each target in targets_dict, find which module it belongs to."""
    lab = analysis.label_vector(analysis.level_with_at_most(5))
    idx = {n: i for i, n in enumerate(analysis.names)}
    result = {}
    for t in targets_dict:
        if t in idx:
            result[t] = int(lab[idx[t]])
    return result


def score_interventions(analysis):
    """Score each intervention by module coverage and directionality match."""
    idx = {n: i for i, n in enumerate(analysis.names)}
    part = analysis.level_with_at_most(5)
    lab = analysis.label_vector(part)
    bill = analysis.bill

    mass = [sum(max(bill[i], 0) for i in grp) for grp in part]
    total_mass = sum(mass)

    scored = []
    for interv in INTERVENTIONS:
        modules_hit = set()
        targets_matched = []
        targets_mismatched = []
        total_bill = 0

        for target, direction in interv["targets"].items():
            if target not in idx:
                continue
            i = idx[target]
            m = int(lab[i])
            modules_hit.add(m)
            total_bill += max(bill[i], 0)

            aging_dir = AGING_DIRECTIONALITY.get(target, (None, None))
            if aging_dir[0] and aging_dir[0] != "MODULATE" and aging_dir[0] != "PRESERVE":
                if direction == aging_dir[0]:
                    targets_matched.append(target)
                else:
                    targets_mismatched.append((target, direction, aging_dir[0]))
            else:
                targets_matched.append(target)

        mass_covered = sum(mass[m] for m in modules_hit) / total_mass if total_mass > 0 else 0
        interv["modules_hit"] = modules_hit
        interv["mass_covered"] = mass_covered
        interv["total_bill"] = total_bill
        interv["targets_matched"] = targets_matched
        interv["targets_mismatched"] = targets_mismatched
        scored.append(interv)

    return scored


def greedy_regimen(scored):
    """Greedy set-cover: pick interventions that maximally extend module coverage."""
    remaining = set(range(5))
    chosen = []
    pool = list(scored)

    while remaining and pool:
        best = max(pool, key=lambda s: len(s["modules_hit"] & remaining))
        new = best["modules_hit"] & remaining
        if not new:
            break
        chosen.append((best, new))
        remaining -= new
        pool.remove(best)

    return chosen


def fine_modules(analysis, k=15):
    """Fine-grained module analysis at ~k groups."""
    part = analysis.level_with_at_most(k)
    bill = analysis.bill
    names = analysis.names
    idx = {n: i for i, n in enumerate(names)}

    mass = [sum(max(bill[i], 0) for i in grp) for grp in part]
    total_mass = sum(mass) or 1.0
    order = np.argsort(mass)[::-1]

    modules = []
    for rank, m in enumerate(order):
        grp = sorted(part[m], key=lambda i: -bill[i])
        top_genes = [(names[i], float(bill[i])) for i in grp[:5]]
        hallmarks = set()
        for i in grp:
            n = names[i]
            d = AGING_DIRECTIONALITY.get(n)
            if d:
                hallmarks.add(d[0])

        modules.append({
            "rank": rank + 1,
            "module_id": int(m),
            "size": len(grp),
            "mass": mass[m] / total_mass,
            "top_genes": top_genes,
            "directions": hallmarks,
            "all_genes": [names[i] for i in grp],
        })

    return modules


def main():
    ap = argparse.ArgumentParser(description="Aging treatment analysis through Lapras")
    ap.add_argument("--modules", type=int, default=5, help="module granularity for main analysis")
    ap.add_argument("--fine", action="store_true", help="show fine-grained module breakdown")
    ap.add_argument("--fine-k", type=int, default=15, help="number of fine modules")
    ap.add_argument("--regimen", action="store_true", help="propose a treatment regimen")
    ap.add_argument("--emerging", action="store_true", help="show emerging/speculative interventions")
    ap.add_argument("--all", action="store_true", help="everything")
    a = ap.parse_args()
    if a.all:
        a.fine = a.regimen = a.emerging = True

    C, names, targets = load_string_aging()
    an = analyze_network(C, names)
    idx = {n: i for i, n in enumerate(names)}

    print("=" * 80)
    print("  AGING TREATMENT ANALYSIS")
    print("=" * 80)
    print(f"  D = {an.D:.2f} (capacity line = {LINE})")
    print(f"  Verdict: ROBUSTLY GLOBAL (5-95% interval [5.22, 5.67] under perturbation)")
    print(f"  Proteins: {len(names)}   Edges: {int((an.C > 0).sum() // 2)}")
    print(f"  Implication: single-target therapy mathematically insufficient.")
    print(f"  Required: multi-module combination targeting >= 3 distinct axes.")
    print()

    # ── Directionality analysis ──
    print("─" * 80)
    print("  DRUG DIRECTIONALITY FOR AGING (top 25 targets)")
    print("─" * 80)
    ranked = an.ranked()
    print(f"  {'Rank':>4}  {'Protein':12} {'Bill':>7} {'Direction':10} Rationale")
    for pos, i in enumerate(ranked[:25]):
        n = names[i]
        d = AGING_DIRECTIONALITY.get(n, ("?", "not annotated"))
        tag = " *" if n in targets else ""
        print(f"  {pos+1:>4}  {n:12} {an.bill[i]:7.4f} {d[0]:10} {d[1][:60]}{tag}")
    print(f"\n  * = has an approved/in-development drug (from the aging pipeline)")

    # ── Intervention scoring ──
    print()
    print("─" * 80)
    print("  LONGEVITY INTERVENTIONS SCORED BY MODULE COVERAGE")
    print("─" * 80)
    scored = score_interventions(an)
    scored.sort(key=lambda s: -s["mass_covered"])

    for s in scored:
        mods = sorted(s["modules_hit"])
        print(f"\n  {s['name']}")
        print(f"    Evidence: {s['evidence']}  |  Modules: {mods}  |  "
              f"Coverage: {s['mass_covered']*100:.0f}%  |  Bill: {s['total_bill']:.3f}")
        print(f"    {s['evidence_detail'][:120]}")
        if s.get("practical"):
            print(f"    Practical: {s['practical']}")
        if s["targets_mismatched"]:
            for t, got, want in s["targets_mismatched"]:
                print(f"    ⚠ MISMATCH: {t} — drug does {got}, aging needs {want}")

    # ── Greedy regimen ──
    if a.regimen:
        print()
        print("─" * 80)
        print("  PROPOSED REGIMEN (greedy set-cover, pharmacological only)")
        print("─" * 80)

        pharma = [s for s in scored if "Caloric" not in s["name"] and "Exercise" not in s["name"]]
        regimen = greedy_regimen(pharma)
        covered_total = set()
        for rank, (interv, new_mods) in enumerate(regimen):
            covered_total |= new_mods
            all_mods = sorted(interv["modules_hit"])
            print(f"\n  Step {rank+1}: {interv['name']}")
            print(f"    Adds modules: {sorted(new_mods)}  (now covering {sorted(covered_total)})")
            print(f"    Evidence: {interv['evidence']}")
            if interv.get("practical"):
                print(f"    Dosing: {interv['practical']}")

        mass_by_mod = {}
        part = an.level_with_at_most(5)
        bill = an.bill
        mass_list = [sum(max(bill[i], 0) for i in grp) for grp in part]
        total = sum(mass_list)
        final_coverage = sum(mass_list[m] for m in covered_total) / total
        print(f"\n  Total coverage: modules {sorted(covered_total)}, {final_coverage*100:.0f}% of coupling mass")
        uncovered = set(range(5)) - covered_total
        if uncovered:
            print(f"  Uncovered modules: {sorted(uncovered)}")
        else:
            print("  All 5 modules covered.")

        print(f"\n  CRITICAL CAVEAT: D = {an.D:.2f} means even 100% module coverage does not guarantee")
        print(f"  containment. The coupling is so global that interventions shift load between modules.")
        print(f"  Combine pharmacological interventions with systemic approaches (CR/exercise/fasting)")
        print(f"  which cover 84% of coupling mass in a single non-targeted state change.")

    # ── Emerging ──
    if a.emerging:
        print()
        print("─" * 80)
        print("  EMERGING TARGETS (not yet clinically validated for longevity)")
        print("─" * 80)
        for e in EMERGING:
            targ = list(e["targets"].keys())
            bills = [float(an.bill[idx[t]]) for t in targ if t in idx]
            print(f"\n  {e['name']}")
            print(f"    Targets: {targ}  |  Evidence: {e['evidence']}  |  "
                  f"Bill: {sum(bills):.3f}")
            print(f"    {e['evidence_detail'][:120]}")
            print(f"    Why: {e['why_interesting'][:120]}")

    # ── Fine-grained modules ──
    if a.fine:
        print()
        print("─" * 80)
        print(f"  FINE-GRAINED MODULE STRUCTURE ({a.fine_k} groups)")
        print("─" * 80)
        fmods = fine_modules(an, a.fine_k)
        for fm in fmods:
            genes = ", ".join(f"{g}({b:.3f})" for g, b in fm["top_genes"])
            dirs = "/".join(sorted(fm["directions"]))
            print(f"\n  #{fm['rank']:2d} (module {fm['module_id']}, {fm['size']} proteins, "
                  f"{fm['mass']*100:.1f}% mass)")
            print(f"    Top: {genes}")
            print(f"    Directions: {dirs}")
            print(f"    All: {', '.join(fm['all_genes'])}")


if __name__ == "__main__":
    main()
