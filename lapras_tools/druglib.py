"""Seed drug library: drug -> HGNC targets, action, development status.

HAND-CURATED from general pharmacology knowledge, NOT pulled from DrugBank or
ChEMBL.  Treat it as a seed: verify every entry against a primary source before
any clinical use, and replace with a DrugBank/ChEMBL export via load_csv() for
real work (columns: drug,targets,action,status; targets separated by '|').

status: approved | investigational
action: inhibitor | antagonist | agonist | modulator | antibody | degrader
"""
import csv
import json

# (drug, targets, action, status)
_SEED = [
    # --- PI3K / AKT / mTOR
    ("alpelisib", "PIK3CA", "inhibitor", "approved"),
    ("capivasertib", "AKT1|AKT2|AKT3", "inhibitor", "approved"),
    ("everolimus", "MTOR", "inhibitor", "approved"),
    ("sirolimus", "MTOR", "inhibitor", "approved"),
    ("idelalisib", "PIK3CD", "inhibitor", "approved"),
    ("copanlisib", "PIK3CA|PIK3CD", "inhibitor", "approved"),
    # --- RAS / MAPK
    ("sotorasib", "KRAS", "inhibitor", "approved"),
    ("adagrasib", "KRAS", "inhibitor", "approved"),
    ("dabrafenib", "BRAF", "inhibitor", "approved"),
    ("vemurafenib", "BRAF", "inhibitor", "approved"),
    ("encorafenib", "BRAF", "inhibitor", "approved"),
    ("trametinib", "MAP2K1|MAP2K2", "inhibitor", "approved"),
    ("cobimetinib", "MAP2K1", "inhibitor", "approved"),
    ("binimetinib", "MAP2K1|MAP2K2", "inhibitor", "approved"),
    ("selumetinib", "MAP2K1|MAP2K2", "inhibitor", "approved"),
    # --- RTKs
    ("erlotinib", "EGFR", "inhibitor", "approved"),
    ("gefitinib", "EGFR", "inhibitor", "approved"),
    ("osimertinib", "EGFR", "inhibitor", "approved"),
    ("afatinib", "EGFR|ERBB2", "inhibitor", "approved"),
    ("lapatinib", "EGFR|ERBB2", "inhibitor", "approved"),
    ("cetuximab", "EGFR", "antibody", "approved"),
    ("panitumumab", "EGFR", "antibody", "approved"),
    ("trastuzumab", "ERBB2", "antibody", "approved"),
    ("pertuzumab", "ERBB2", "antibody", "approved"),
    ("tucatinib", "ERBB2", "inhibitor", "approved"),
    ("crizotinib", "ALK|MET|ROS1", "inhibitor", "approved"),
    ("alectinib", "ALK", "inhibitor", "approved"),
    ("lorlatinib", "ALK|ROS1", "inhibitor", "approved"),
    ("capmatinib", "MET", "inhibitor", "approved"),
    ("tepotinib", "MET", "inhibitor", "approved"),
    ("cabozantinib", "MET|KDR|RET|AXL", "inhibitor", "approved"),
    ("selpercatinib", "RET", "inhibitor", "approved"),
    ("erdafitinib", "FGFR1|FGFR2|FGFR3|FGFR4", "inhibitor", "approved"),
    ("pemigatinib", "FGFR1|FGFR2|FGFR3", "inhibitor", "approved"),
    ("imatinib", "ABL1|KIT|PDGFRA|PDGFRB", "inhibitor", "approved"),
    ("dasatinib", "ABL1|SRC|KIT|PDGFRB", "inhibitor", "approved"),
    ("sunitinib", "KDR|KIT|PDGFRA|PDGFRB|FLT3", "inhibitor", "approved"),
    ("sorafenib", "RAF1|BRAF|KDR|PDGFRB", "inhibitor", "approved"),
    ("lenvatinib", "KDR|FGFR1|RET|KIT", "inhibitor", "approved"),
    ("regorafenib", "KDR|RAF1|BRAF|KIT", "inhibitor", "approved"),
    ("bevacizumab", "VEGFA", "antibody", "approved"),
    ("ramucirumab", "KDR", "antibody", "approved"),
    ("axitinib", "KDR", "inhibitor", "approved"),
    ("larotrectinib", "NTRK1|NTRK2|NTRK3", "inhibitor", "approved"),
    ("gilteritinib", "FLT3|AXL", "inhibitor", "approved"),
    ("midostaurin", "FLT3|KIT", "inhibitor", "approved"),
    # --- cell cycle / DNA damage / epigenetic
    ("palbociclib", "CDK4|CDK6", "inhibitor", "approved"),
    ("ribociclib", "CDK4|CDK6", "inhibitor", "approved"),
    ("abemaciclib", "CDK4|CDK6", "inhibitor", "approved"),
    ("olaparib", "PARP1|PARP2", "inhibitor", "approved"),
    ("talazoparib", "PARP1|PARP2", "inhibitor", "approved"),
    ("niraparib", "PARP1|PARP2", "inhibitor", "approved"),
    ("vorasidenib", "IDH1|IDH2", "inhibitor", "approved"),
    ("ivosidenib", "IDH1", "inhibitor", "approved"),
    ("enasidenib", "IDH2", "inhibitor", "approved"),
    ("venetoclax", "BCL2", "inhibitor", "approved"),
    ("vorinostat", "HDAC1|HDAC2|HDAC3|HDAC6", "inhibitor", "approved"),
    ("tazemetostat", "EZH2", "inhibitor", "approved"),
    ("vismodegib", "SMO", "inhibitor", "approved"),
    ("sonidegib", "SMO", "inhibitor", "approved"),
    ("ibrutinib", "BTK", "inhibitor", "approved"),
    ("acalabrutinib", "BTK", "inhibitor", "approved"),
    ("ruxolitinib", "JAK1|JAK2", "inhibitor", "approved"),
    ("fedratinib", "JAK2", "inhibitor", "approved"),
    ("tofacitinib", "JAK1|JAK3|JAK2", "inhibitor", "approved"),
    ("baricitinib", "JAK1|JAK2", "inhibitor", "approved"),
    ("upadacitinib", "JAK1", "inhibitor", "approved"),
    ("fulvestrant", "ESR1", "degrader", "approved"),
    ("tamoxifen", "ESR1", "modulator", "approved"),
    ("letrozole", "CYP19A1", "inhibitor", "approved"),
    ("anastrozole", "CYP19A1", "inhibitor", "approved"),
    ("enzalutamide", "AR", "antagonist", "approved"),
    ("abiraterone", "CYP17A1", "inhibitor", "approved"),
    ("bortezomib", "PSMB5", "inhibitor", "approved"),
    ("bicalutamide", "AR", "antagonist", "approved"),
    # --- STAT3 / NF-kB / MDM2 / TP53 (mostly undrugged: investigational only)
    ("napabucasin", "STAT3", "inhibitor", "investigational"),
    ("TTI-101", "STAT3", "inhibitor", "investigational"),
    ("IMD-0354", "IKBKB", "inhibitor", "investigational"),   # IKK-beta, upstream of NF-kB; no direct NFKB1 drug in library
    ("navtemadlin", "MDM2", "inhibitor", "investigational"),
    ("idasanutlin", "MDM2", "inhibitor", "investigational"),
    ("eprenetapopt", "TP53", "modulator", "investigational"),
    # --- immune / inflammation
    ("adalimumab", "TNF", "antibody", "approved"),
    ("infliximab", "TNF", "antibody", "approved"),
    ("etanercept", "TNF", "antibody", "approved"),
    ("tocilizumab", "IL6R", "antibody", "approved"),
    ("sarilumab", "IL6R", "antibody", "approved"),
    ("siltuximab", "IL6", "antibody", "approved"),
    ("anakinra", "IL1B|IL1R1", "antagonist", "approved"),
    ("canakinumab", "IL1B", "antibody", "approved"),
    ("secukinumab", "IL17A", "antibody", "approved"),
    ("ustekinumab", "IL12B|IL23A", "antibody", "approved"),
    ("dupilumab", "IL4R", "antibody", "approved"),
    ("omalizumab", "IGHE", "antibody", "approved"),
    ("vedolizumab", "ITGA4|ITGB7", "antibody", "approved"),
    ("abatacept", "CD80|CD86", "antibody", "approved"),
    ("rituximab", "MS4A1", "antibody", "approved"),
    ("pembrolizumab", "PDCD1", "antibody", "approved"),
    ("nivolumab", "PDCD1", "antibody", "approved"),
    ("atezolizumab", "CD274", "antibody", "approved"),
    ("ipilimumab", "CTLA4", "antibody", "approved"),
    ("belimumab", "TNFSF13B", "antibody", "approved"),
    ("methotrexate", "DHFR", "inhibitor", "approved"),
    ("mycophenolate", "IMPDH1|IMPDH2", "inhibitor", "approved"),
    ("tacrolimus", "PPP3CA|FKBP1A", "inhibitor", "approved"),
    ("hydroxychloroquine", "TLR9", "modulator", "approved"),
    ("dexamethasone", "NR3C1", "agonist", "approved"),
    ("prednisone", "NR3C1", "agonist", "approved"),
    ("montelukast", "CYSLTR1", "antagonist", "approved"),
    # --- cardiometabolic
    ("atorvastatin", "HMGCR", "inhibitor", "approved"),
    ("rosuvastatin", "HMGCR", "inhibitor", "approved"),
    ("evolocumab", "PCSK9", "antibody", "approved"),
    ("ezetimibe", "NPC1L1", "inhibitor", "approved"),
    ("lisinopril", "ACE", "inhibitor", "approved"),
    ("enalapril", "ACE", "inhibitor", "approved"),
    ("losartan", "AGTR1", "antagonist", "approved"),
    ("valsartan", "AGTR1", "antagonist", "approved"),
    ("sacubitril", "MME", "inhibitor", "approved"),
    ("metoprolol", "ADRB1", "antagonist", "approved"),
    ("carvedilol", "ADRB1|ADRB2", "antagonist", "approved"),
    ("spironolactone", "NR3C2", "antagonist", "approved"),
    ("empagliflozin", "SLC5A2", "inhibitor", "approved"),
    ("dapagliflozin", "SLC5A2", "inhibitor", "approved"),
    ("semaglutide", "GLP1R", "agonist", "approved"),
    ("liraglutide", "GLP1R", "agonist", "approved"),
    ("tirzepatide", "GLP1R|GIPR", "agonist", "approved"),
    ("metformin", "PRKAA1", "modulator", "approved"),
    ("sitagliptin", "DPP4", "inhibitor", "approved"),
    ("pioglitazone", "PPARG", "agonist", "approved"),
    ("resmetirom", "THRB", "agonist", "approved"),
    ("insulin", "INSR", "agonist", "approved"),
    ("warfarin", "VKORC1", "inhibitor", "approved"),
    ("rivaroxaban", "F10", "inhibitor", "approved"),
    ("apixaban", "F10", "inhibitor", "approved"),
    ("clopidogrel", "P2RY12", "antagonist", "approved"),
    ("aspirin", "PTGS1|PTGS2", "inhibitor", "approved"),
    ("ibuprofen", "PTGS1|PTGS2", "inhibitor", "approved"),
    ("celecoxib", "PTGS2", "inhibitor", "approved"),
    ("alendronate", "FDPS", "inhibitor", "approved"),
    ("denosumab", "TNFSF11", "antibody", "approved"),
    ("romosozumab", "SOST", "antibody", "approved"),
    ("pirfenidone", "TGFB1", "modulator", "approved"),
    ("nintedanib", "FGFR1|PDGFRA|KDR", "inhibitor", "approved"),
    # --- CNS
    ("levodopa", "DRD2", "agonist", "approved"),   # dopamine precursor; acts via dopamine receptors
    ("pramipexole", "DRD2|DRD3", "agonist", "approved"),
    ("haloperidol", "DRD2", "antagonist", "approved"),
    ("risperidone", "DRD2|HTR2A", "antagonist", "approved"),
    ("olanzapine", "DRD2|HTR2A", "antagonist", "approved"),
    ("fluoxetine", "SLC6A4", "inhibitor", "approved"),
    ("sertraline", "SLC6A4", "inhibitor", "approved"),
    ("venlafaxine", "SLC6A4|SLC6A2", "inhibitor", "approved"),
    ("ketamine", "GRIN1|GRIN2B", "antagonist", "approved"),
    ("memantine", "GRIN1|GRIN2B", "antagonist", "approved"),
    ("donepezil", "ACHE", "inhibitor", "approved"),
    ("lecanemab", "APP", "antibody", "approved"),
    ("donanemab", "APP", "antibody", "approved"),
    ("riluzole", "SCN1A", "modulator", "approved"),
    ("carbamazepine", "SCN1A|SCN2A", "inhibitor", "approved"),
    ("lamotrigine", "SCN1A|SCN2A", "inhibitor", "approved"),
    ("valproate", "ABAT|SCN2A", "inhibitor", "approved"),   # antiepileptic mechanism (GABA transaminase, Na channel)
    ("levetiracetam", "SV2A", "modulator", "approved"),
    ("natalizumab", "ITGA4", "antibody", "approved"),
    ("fingolimod", "S1PR1", "modulator", "approved"),
    ("ocrelizumab", "MS4A1", "antibody", "approved"),
    # --- infection
    ("dolutegravir", "HIV1_INTEGRASE", "inhibitor", "approved"),
    ("tenofovir", "HIV1_RT", "inhibitor", "approved"),
    ("maraviroc", "CCR5", "antagonist", "approved"),
    ("isoniazid", "INHA", "inhibitor", "approved"),
    ("rifampin", "RPOB", "inhibitor", "approved"),
]


# Genes a drug acts THROUGH rather than ON (synthetic lethality, pathway
# dependence).  Counted as a weaker match than a direct target.
_INDIRECT = {
    "olaparib": "BRCA1|BRCA2|PALB2|ATM",
    "talazoparib": "BRCA1|BRCA2|PALB2",
    "niraparib": "BRCA1|BRCA2",
    "letrozole": "ESR1",
    "anastrozole": "ESR1",
    "abiraterone": "AR",
    "pembrolizumab": "CD274",
}


def heading_matches(heading, keywords):
    h = heading.lower()
    pos = [k for k in keywords if not k.startswith("-")]
    neg = [k[1:] for k in keywords if k.startswith("-")]
    return any(k in h for k in pos) and not any(k in h for k in neg)


class Drug:
    __slots__ = ("name", "targets", "action", "status", "indirect", "indications", "verified", "seed_only", "marketed")

    def __init__(self, name, targets, action, status, indirect="", indications="", verified=False, seed_only=""):
        self.name, self.action, self.status = name, action, status
        self.marketed = None          # True/False from Drugs@FDA (fda_status.py); None = unknown
        self.seed_only = tuple(t for t in (seed_only or "").split("|") if t)   # targets ChEMBL does not list
        self.targets = tuple(t for t in targets.split("|") if t)
        self.indirect = tuple(t for t in (indirect or _INDIRECT.get(name, "")).split("|") if t)
        # indications: {MeSH heading: max clinical phase}, from "heading@phase|..." strings
        self.indications = {}
        for tok in (indications or "").split("|"):
            if "@" in tok:
                h, p = tok.rsplit("@", 1)
                self.indications[h] = int(p)
        self.verified = verified                  # True when the row came from ChEMBL

    def best_indication(self, keywords, min_phase=3):
        """(heading, phase) of the highest-phase ChEMBL indication matching `keywords`
        at >= min_phase, else None.  Phase 4 = approved; 3 = phase-3 trials only.
        Keywords are lower-case substrings of the MeSH heading; a keyword starting with '-' is an
        exclusion ('-paraganglioma' stops 'glioma' matching 'Paraganglioma')."""
        hits = [(h, p) for h, p in self.indications.items()
                if p >= min_phase and heading_matches(h, keywords)]
        return max(hits, key=lambda x: x[1]) if hits else None

    def evidence_count(self, keywords, min_phase=4):
        """How many distinct ChEMBL indication headings for this disease the drug holds at >= min_phase
        (breadth of approved use; used to order equally eligible drugs on one target)."""
        return sum(1 for h, p in self.indications.items()
                   if p >= min_phase and heading_matches(h, keywords))

    def indicated_for(self, keywords, min_phase=3):
        return self.best_indication(keywords, min_phase) is not None

    @property
    def approved(self):
        return self.status == "approved"

    def __repr__(self):
        return f"{self.name} ({'/'.join(self.targets)} {self.action}, {self.status})"


def seed_library():
    return {d[0].lower(): Drug(*d) for d in _SEED}


def load_csv(path):
    lib = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            lib[row["drug"].lower()] = Drug(row["drug"], row["targets"], row["action"], row["status"],
                                            indications=row.get("indications", ""),
                                            verified=bool(row.get("chembl_id")),
                                            seed_only=row.get("seed_only", ""))
    return lib


def default_library():
    """ChEMBL-derived library (data/druglib_chembl.csv, built by build_druglib) when present,
    with seed indirect (synthetic-lethal) targets layered on; otherwise the hand-written seed."""
    import os
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    # MERGE, never replace: bulk ChEMBL (every drug with a mechanism, phase >= 3, touching a network
    # protein) first, then the 167 individually re-derived drugs for any the bulk file lacks, then the
    # hand-written seed for those ChEMBL does not have at all (e.g. TTI-101).  Earlier sources win.
    lib = {}
    for name in ("druglib_chembl_full.csv", "druglib_chembl.csv"):
        p = os.path.join(here, name)
        if os.path.exists(p):
            for k, d in load_csv(p).items():
                lib.setdefault(k, d)
    for k, d in seed_library().items():
        lib.setdefault(k, d)               # verified=False: no indications, flagged as unverified
    for d in lib.values():
        if d.name in _INDIRECT:
            d.indirect = tuple(_INDIRECT[d.name].split("|"))
    mpath = os.path.join(here, "fda_marketed.json")           # built by `python3 -m lapras_tools.fda_status`
    if os.path.exists(mpath):
        ms = json.load(open(mpath))
        for k, d in lib.items():
            d.marketed = {"marketed": True, "not marketed": False}.get(ms.get(k))
    return lib


# MeSH-heading keywords per Lapras disease key (lower-case substrings)
DISEASE_KEYWORDS = {
    # Headings audited against ChEMBL's actual MeSH terms (validate with print_keyword_headings()):
    # loose substrings had matched COPD drugs ('Lung Diseases, Obstructive') to lung cancer, kidney
    # stones/cancer to CKD, cystic fibrosis/myelofibrosis to systemic fibrosis, fertility drugs to
    # ovarian cancer, 'Paraganglioma' to glioma and Wolff-Parkinson-White to Parkinson's.
    "breast_cancer": ["breast neoplasm", "breast carcinoma"],
    "lung_cancer": ["lung neoplasm", "carcinoma, non-small-cell lung", "small cell lung", "adenocarcinoma of lung"],
    "colorectal_cancer": ["colorectal neoplasm", "colonic neoplasm", "rectal neoplasm"],
    "melanoma": ["melanoma"], "prostate_cancer": ["prostatic neoplasm"], "liver_cancer": ["hepatocellular", "liver neoplasm"],
    "glioblastoma": ["glioblastoma", "glioma", "astrocytoma", "brain neoplasm", "-paraganglioma"],
    "ovarian_cancer": ["ovarian neoplasm", "carcinoma, ovarian"],
    "parkinsons": ["parkinson disease", "parkinsonian", "-wolff"], "huntingtons": ["huntington"], "multiple_sclerosis": ["multiple sclerosis"],
    "rheumatoid_arthritis": ["arthritis, rheumatoid"], "lupus": ["lupus"], "type1_diabetes": ["diabetes mellitus, type 1"],
    "type2_diabetes": ["diabetes mellitus, type 2"], "nafld": ["fatty liver", "steatohepatitis"], "obesity": ["obesity", "overweight"],
    "atherosclerosis": ["atherosclerosis", "hypercholesterol", "hyperlipid", "coronary artery disease", "coronary disease"],
    "heart_failure": ["heart failure"],
    "crohns": ["crohn"], "ipf": ["idiopathic pulmonary fibrosis"], "depression": ["depressive disorder", "depression, postpartum", "-respiratory"],
    "schizophrenia": ["schizophren"],
    "hiv": ["hiv", "acquired immunodeficiency"], "tuberculosis": ["tuberculosis"], "osteoporosis": ["osteoporosis"],
    "chronic_kidney": ["kidney failure, chronic", "renal insufficiency, chronic", "kidney diseases", "diabetic nephropathies", "-polycystic"],
    "sickle_cell": ["sickle"], "asthma": ["asthma"], "epilepsy": ["epilep", "seizure"],
    "fibrosis_systemic": ["scleroderma", "systemic sclerosis"],
}


def print_keyword_headings(lib, min_phase=3, top=30):
    """Audit aid: which ChEMBL indication headings does each disease's keyword list match?
    Read the output for false matches whenever the keywords or the library change."""
    for k, kw in DISEASE_KEYWORDS.items():
        heads = {}
        for d in lib.values():
            for h, p in d.indications.items():
                if p >= min_phase and heading_matches(h, kw):
                    heads.setdefault(h, set()).add(d.name)
        print(f"{k}: " + "; ".join(f"{h} ({len(v)})" for h, v in sorted(heads.items(), key=lambda x: -len(x[1]))[:top]))


def by_target(lib, approved_only=False):
    """target gene -> list of Drug, approved first."""
    out = {}
    for d in lib.values():
        if approved_only and not d.approved:
            continue
        for t in d.targets + d.indirect:
            out.setdefault(t, []).append(d)
    for t in out:
        # direct targets first, then approved, then currently marketed (unknown next, withdrawn last),
        # then ChEMBL-verified over hand-written, then selective, then name
        out[t].sort(key=lambda d: (t not in d.targets, not d.approved,
                                   {True: 0, None: 1, False: 2}[d.marketed], not d.verified, len(d.targets), d.name))
    return out
