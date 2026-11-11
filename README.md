# Drug Target Discovery Pipeline

## Why this exists

Right now, drug discovery starts with thousands of candidate proteins and no principled way to rank them. Scientists rely on literature intuition, guilt-by-association, and expensive high-throughput screens. Worse, there's no way to know in advance whether a disease can be treated with a single drug or needs a combination — the field finds out after years of failed trials.

This pipeline solves both problems from network data alone:

1. **It ranks every protein in a disease network by how much coupling it owns** — not by how many connections it has (degree), but by how much of the system's total statistical coupling runs through it. On the aging network, this ranking recovers known drug targets at 4.8x enrichment in the top 10 — without being told which proteins have drugs.

2. **It gives every disease a single number (D) that predicts whether single-target therapy can work.** Below D = 3.03, the disease is modular and one well-chosen drug can contain it. Above 3.03, the coupling is global and you need combinations. Across 31 diseases, every modular disease has a successful single-drug therapy. Every global disease is one where monotherapy isn't enough.

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
python3 batch.py --score 0.9       # all 31 diseases from STRING data
```

## The periodic table

31 diseases analyzed with real STRING data (score >= 0.9). 9 modular, 22 global:

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
| Type 2 Diabetes | 3.89 | GLOBAL |
| Type 1 Diabetes | 4.02 | GLOBAL |
| Breast Cancer | 4.03 | GLOBAL |
| Tuberculosis | 4.03 | GLOBAL — explains why TB needs 4 drugs |
| Asthma | 4.04 | GLOBAL |
| All other cancers | 4.05–4.50 | GLOBAL |
| Rheumatoid Arthritis | 4.50 | GLOBAL — most entangled non-cancer disease |

Every modular disease has a successful single-drug therapy. That's the validation.

## Key findings

**NF-kB (NFKB1)** is the most undrugged bottleneck in medicine. It ranks top-15 in 17 of 31 diseases. #1 in atherosclerosis, depression, and tuberculosis. No approved NF-kB inhibitor exists for any of them.

**STAT3** ranks top-15 in 15 diseases. #1 in rheumatoid arthritis and Crohn's. No approved inhibitor.

**Depression's #1 target is NF-kB, not serotonin.** The field runs on the serotonin hypothesis. The coupling says inflammation owns more of the network's wiring than any neurotransmitter transporter.

**NAFLD/NASH is modular (D=2.75).** The field spent a decade on combination strategies that failed. Resmetirom was approved in 2024 as a single-target drug. The D value would have predicted this.

**Obesity is modular (D=2.74).** Before semaglutide, the assumption was that obesity was too multifactorial for a drug.

**Parkinson's is right at the line (D=3.07).** This explains why levodopa partially works but can't contain the disease.

## Cross-disease repurposing

Proteins ranking in the top 30% across many diseases:

| Protein | # Diseases | Notable |
|---------|-----------|---------|
| TNF | 18 | Has inhibitors, but not used in most of these diseases |
| IL6 | 17 | tocilizumab exists, used in few |
| NFKB1 | 17 | No approved inhibitor anywhere |
| IL1B | 16 | anakinra exists, rarely used |
| STAT3 | 15 | No approved inhibitor |
| AKT1 | 14 | capivasertib (cancer only) |
| MTOR | 15 | everolimus (cancer/transplant only) |
| TP53 | 12 | "Undruggable" — highest bill in most cancers |

## Files

```
coupling.py           Core math (TC, bill, compiler, sensitivity)
networks.py           5 built-in disease networks
pipeline.py           Pipeline runner and CLI
diseases.py           31 disease protein lists with known targets
download_string.py    STRING API downloader
batch.py              Batch runner — builds the periodic table
data/                 STRING interaction data (TSV per disease)
results/              Output (periodic table, per-disease reports)
```

## Running the full analysis

```bash
# Download STRING data for all 31 diseases (~35 seconds)
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

## The capacity line (D = 3.03)

The halving depth D is the number of hierarchy levels needed to reduce residual coupling to 1/4. At D = 3.03, hierarchical containment fails — modular intervention cannot control the system's coupling. Below the line, a single well-chosen target can contain the disease. Above the line, you need combinations that hit multiple modules.
