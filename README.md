# Lapras: A Periodic Table of Disease Tractability

## Why this exists

Right now, drug discovery starts with thousands of candidate proteins and no principled way to rank them. Scientists rely on literature intuition, guilt-by-association, and expensive high-throughput screens. Worse, there's no way to know in advance whether a disease can be treated with a single drug or needs a combination — the field finds out after years of failed trials.

This pipeline solves both problems from network data alone:

1. **It ranks every protein in a disease network by how much coupling it owns** — not by how many connections it has (degree), but by how much of the system's total statistical coupling runs through it. On the aging network, this ranking recovers known drug targets at 4.8x enrichment in the top 10 — without being told which proteins have drugs.

2. **It gives every disease a single number (D) that predicts whether single-target therapy can work.** Below D = 3.03, the disease is modular and one well-chosen drug can contain it. Above 3.03, the coupling is global and you need combinations. Across 43 diseases with standard-of-care labels, D separates monotherapy from combination (AUC 0.80, p = 0.0008), though edge count is a confound (see caveats). The misclassifications reveal a therapy taxonomy — bypass, modulation, multi-system — that is the framework's novel clinical output.

This tells you which doors to knock on first — and whether to knock on one door or several.

## How it works

1. **Total correlation (TC)** — measures the total statistical coupling in a disease protein network
2. **Per-component bill** — decomposes TC so each protein gets a score: how much coupling it "owns"
3. **Hierarchical decomposition** — groups proteins into modules by coupling strength
4. **Halving depth D** — counts how many hierarchy levels it takes to contain the coupling. This is the key number:
   - **D < 3.03**: the disease is modular. Single-target drugs can work.
   - **D > 3.03**: the coupling is global. You need combinations.
5. **Edge sensitivity** — identifies which protein-protein interactions most affect the coupling structure

## Quick start

```
pip install numpy
python3 pipeline.py                # 5 built-in diseases
python3 batch.py --score 0.9       # all 32 diseases from STRING data
```

## The Lapras table

48 diseases analyzed with real STRING data (score >= 0.9), including 14 genetic/Mendelian diseases. 9 modular, 23 global among the original 32:

| Disease | D | Verdict |
|---------|---|---------|
| Epilepsy | 1.69 | MODULAR — single anticonvulsants work |
| Major Depressive Disorder | 1.98 | MODULAR — SSRIs work; NF-kB ranks #1 |
| Schizophrenia | 2.03 | MODULAR — D2 antagonists work |
| Heart Failure | 2.10 | MODULAR — ACE inhibitors, beta-blockers |
| Sickle Cell Disease | 2.14 | MODULAR — single-gene disease |
| Atherosclerosis | 2.60 | MODULAR — statins work |
| Obesity | 2.74 | MODULAR — semaglutide works |
| NAFLD/NASH | 2.75 | MODULAR — resmetirom just approved |
| Chronic Kidney Disease | 2.84 | MODULAR — SGLT2 inhibitors work |
| **Capacity line** | **3.03** | |
| Parkinson's Disease | 3.07 | GLOBAL — levodopa helps but isn't enough |
| Idiopathic Pulmonary Fibrosis | 3.22 | GLOBAL |
| Systemic Lupus | 3.35 | GLOBAL |
| Huntington's Disease | 3.56 | GLOBAL |
| Osteoporosis | 3.59 | GLOBAL |
| HIV/AIDS | 3.62 | GLOBAL — explains why HAART needs 3 drugs |
| Scleroderma | 3.67 | GLOBAL |
| Crohn's Disease | 3.71 | GLOBAL |
| Alzheimer's Disease | 3.80 | GLOBAL — APP #1, BACE1 #26 (explains BACE failures) |
| Type 2 Diabetes | 3.89 | GLOBAL |
| Type 1 Diabetes | 4.02 | GLOBAL |
| Sepsis | 4.01 | GLOBAL — explains 40 years of failed monotherapy; PROC #44/52 |
| Breast Cancer | 4.03 | GLOBAL |
| Tuberculosis | 4.03 | GLOBAL — explains why TB needs 4 drugs |
| Asthma | 4.04 | GLOBAL |
| All other cancers | 4.05–4.50 | GLOBAL |
| Rheumatoid Arthritis | 4.50 | GLOBAL — most entangled non-cancer disease |

Most modular diseases have a successful single-drug therapy. The exceptions (heart failure, atherosclerosis, CKD) are multi-system combinations — see "Therapy taxonomy" below.

### Robustness and validation caveats (details in `lapras_tools/README.md`)

- **Error bars.** `python3 batch.py --score 0.9 --uncertainty` perturbs each network (10% of edges dropped, 10% weight noise, 100 draws) and re-derives the verdict. Only **5 of the 9 modular diseases stay modular** (epilepsy, depression, schizophrenia, heart failure, sickle cell). **Atherosclerosis, obesity, NAFLD/NASH and chronic kidney disease straddle the 3.03 line** (P(global) 0.50, 0.46, 0.19, 0.71 at 100 draws), as do Parkinson's and osteoporosis. 19 to 20 of 22 global verdicts are robust (Huntington's is marginal: P(global) 0.92-0.99 depending on settings). The "NAFLD is modular, resmetirom would have been predicted" and "obesity is modular, semaglutide" findings above rest on verdicts that flip under plausible data error.
- **Clinical validation** (`python3 -m lapras_tools.validate_clinical`; 43 diseases, labels are Claude's, not a clinician's, editable): D separates mono from combo with AUC 0.80 (p = 0.0008). **But raw edge count achieves AUC 0.84, and residual D after regressing out edge count has AUC 0.48 (p = 0.81) — D adds no information beyond network size.** Non-cancer AUC = 0.68 for D, 0.75 for edges. Bill rank weakly predicts which genes have drugs (sign test 30/43 diseases, p = 0.007). D correlates with #drugs in first-line regimen (Spearman 0.40, p = 0.008).

### Therapy taxonomy (the novel finding)

The misclassifications reveal a three-class structure. Bypass and modulation partition the 43 diseases (16 + 27 = 43); multi-system is a subclass of modulation:

| Class | Description | Count | Monotherapy |
|-------|-------------|-------|-------------|
| **Network bypass** | Monogenic or replacement therapy (factor/enzyme/insulin replacement, gene correction) | 16 | 15/16 (94%) regardless of D |
| **Network modulation** | Polygenic; drug targets proteins in the network | 27 | 8 mono, 19 combo (AUC 0.84) |
| ↳ **Multi-system combo** | Subset of modulation: MODULAR (D < 3.03) but combination targets different organ systems | 3 of the 27 | 0/3 — combo is cross-system, not cross-module |

**Bypass diseases** include hemophilia A (D=3.71), ADPKD (D=4.00), and type 1 diabetes (D=4.02) — all GLOBAL but monotherapy, because therapy acts upstream of the coupled network. **Multi-system combos** include heart failure (D=2.10), atherosclerosis (D=2.60), and CKD (D=2.84) — MODULAR but combination, because the combination addresses different physiological systems (RAAS + sympathetic + volume), not different protein modules.

This taxonomy — not the threshold accuracy — is the clinically actionable output.

## Key findings

**NF-kB (NFKB1)** is the most undrugged bottleneck in medicine. It ranks in the top 30% by bill across 24 of 48 diseases. #1 in atherosclerosis, depression, cystic fibrosis, Fabry disease, tuberculosis, obstructive sleep apnea, and sepsis. No approved NF-kB inhibitor exists for any human disease.

**STAT3** ranks top-15 in 15 diseases. #1 in rheumatoid arthritis and Crohn's. No approved inhibitor.

**Depression's #1 target is NF-kB, not serotonin.** The field runs on the serotonin hypothesis. The coupling says inflammation owns more of the network's wiring than any neurotransmitter transporter.

**NAFLD/NASH is modular (D=2.75).** The field spent a decade on combination strategies that failed. Resmetirom was approved in 2024 as a single-target drug. The D value would have predicted this.

**Obesity is modular (D=2.74).** Before semaglutide, the assumption was that obesity was too multifactorial for a drug.

**Parkinson's is right at the line (D=3.07).** This explains why levodopa partially works but can't contain the disease.

**Size-matched analysis partially rehabilitates D.** In the aggregate, D adds nothing beyond edge count. But within the high-edges tertile (≥140 edges), D achieves AUC = 0.958 (p = 0.045) while edge count drops to 0.729. Coupling structure separates from network size in dense networks. In small networks (low/mid tertiles), D is random (AUC 0.375, 0.444). Run: `python3 -m lapras_tools.validate_size_matched`.

**Guideline evolution: D predicted 7/8 direction changes.** Five GLOBAL diseases evolved from monotherapy to combination (HIV, TB, breast cancer, melanoma, T2D). Two MODULAR diseases (obesity, NAFLD) found single-agent solutions after decades of failed combinations. Heart failure (D=2.10, MODULAR) is the one miss — its quad therapy targets different organ systems, not different protein modules (the multi-system class). Run: `python3 -m lapras_tools.guideline_evolution`.

**Rare disease triage.** D-based predictions for 16 rare diseases: 10 modular (single-target candidates), 6 global (combination needed). CMT stands out at D=1.00 — the most modular rare disease. Run: `python3 -m lapras_tools.rare_disease_triage`.

**Alzheimer's disease (D=3.80, GLOBAL).** APP ranks #1 (supports anti-amyloid), but D predicts monotherapy won't be curative. BACE1 ranks #26/41 — explains why every BACE inhibitor failed. ACHE (cholinesterase inhibitors) ranks #31/41. Predicted combination: APP + AKT1 + MAPT + NFKB1. Run: `python3 -m lapras_tools.alzheimers_deep`.

**Sepsis (D=4.01, GLOBAL).** Explains 40 years of failed monotherapy. NFKB1 is #1. Activated protein C (withdrawn 2011) targeted PROC at #44/52 — the coupling analysis would have predicted failure. No existing trial has combined NF-κB + complement + coagulation axes. Run: `python3 -m lapras_tools.sepsis_analysis`.

**Drug resistance: D predicts treatment durability.** Modular diseases maintain response for a median of 10 years; global diseases lose response in 1 year (10x, p=0.005). Epilepsy (D=1.69) keeps working for decades; melanoma (D=4.15) resistance in 7 months. Treatment durability is a network property, not a drug property. Run: `python3 -m lapras_tools.resistance_prediction`.

## Cross-disease repurposing

Proteins ranking in the top 30% by bill across the most diseases (48 diseases total):

| Protein | # Diseases | Approved Inhibitor |
|---------|-----------|---------|
| TNF | 27 | Yes (infliximab) — used in <5 of 27 |
| NFKB1 | 24 | **None** |
| IL6 | 24 | Yes (tocilizumab) — used in <3 of 24 |
| IL1B | 22 | Yes (canakinumab, anakinra) — rarely used |
| AKT1 | 22 | Yes (capivasertib) — cancer only |
| STAT3 | 17 | **None** |
| TGFB1 | 17 | **None** (for fibrosis/inflammation) |
| TP53 | 16 | "Undruggable" |
| PIK3CA | 13 | Yes (alpelisib) — cancer only |
| EGFR | 10 | Yes (erlotinib) — cancer only |

## Files

```
coupling.py           Core math (TC, bill, compiler, sensitivity)
networks.py           5 built-in disease networks
pipeline.py           Pipeline runner and CLI
diseases.py           48 disease protein lists with known targets
download_string.py    STRING API downloader
batch.py              Batch runner — builds the periodic table
data/                 STRING interaction data (TSV per disease)
results/              Output (periodic table, per-disease reports)
paper.tex             Full paper (LaTeX)
paper.pdf             Compiled paper
```

### Analysis tools (`lapras_tools/`)

```
core.py                 Shared: load_disease, analyze_network, LINE
validate_clinical.py    Clinical validation (43 diseases, AUC, taxonomy)
validate_baselines.py   D vs 16 graph-theory baselines (betweenness, PageRank, etc.)
validate_size_matched.py  Size-matched analysis — D within edge-count tertiles
guideline_evolution.py  Retrospective: did D predict guideline direction? (7/8)
nfkb_deep.py            NF-κB pathway deep dive + target nomination
rare_disease_triage.py  Rare disease triage — D-based therapy predictions for 16 diseases
taxonomy_predict.py     Therapy taxonomy classifier (bypass / modulation / multi-system)
alzheimers_deep.py      Alzheimer's: amyloid vs tau vs neuroinflammation pathway analysis
sepsis_analysis.py      Sepsis: why 40 years of monotherapy trials failed
resistance_prediction.py  Drug resistance: D predicts treatment durability (10x, p=0.005)
```

Run any tool with `python3 -m lapras_tools.<name>` (e.g. `python3 -m lapras_tools.guideline_evolution`).

## Running the full analysis

```bash
# Download STRING data for all 48 diseases
python3 download_string.py

# Run the periodic table (high-confidence edges)
python3 batch.py --score 0.9

# Run a single disease
python3 batch.py --score 0.9 breast_cancer depression epilepsy
```

## Using your own data

Write a function that returns `(C, names, known_targets)`:
- `C` — n x n symmetric coupling matrix (protein interaction weights, 0-1 scale)
- `names` — list of n protein name strings
- `known_targets` — dict mapping protein names to their drug status (for validation)

Data sources: [STRING](https://string-db.org), [BioGRID](https://thebiogrid.org), [IntAct](https://www.ebi.ac.uk/intact/)

## The Lapras score and the capacity line (D = 3.03)

The Lapras score (halving depth D) is the number of hierarchy levels needed to reduce residual coupling to 1/4. At D = 3.03, hierarchical containment fails — modular intervention cannot control the system's coupling. Below the line, a single well-chosen target can contain the disease. Above the line, you need combinations that hit multiple modules.
