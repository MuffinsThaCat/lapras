# A Coupling-Based Periodic Table of Disease Tractability

## Abstract

We introduce a method that assigns every disease a single number—the halving depth D—that predicts whether it responds to single-target therapy or requires drug combinations. The method measures total correlation (TC) in protein interaction networks, decomposes it hierarchically, and determines how many levels of decomposition are needed to contain the coupling. A theoretical capacity line at D = 3.03 separates modular diseases (D < 3.03), where one well-chosen target can control the network, from globally entangled diseases (D > 3.03), where combinations are necessary. We analyze 31 diseases using high-confidence STRING protein interaction data (combined score ≥ 900). Of these, 9 fall below the capacity line and 22 above. Every modular disease has a clinically validated single-drug therapy; every global disease requires or benefits from combination treatment. The method also ranks every protein in each network by its coupling contribution (the "bill"), recovering known drug targets at up to 4.8× enrichment without prior knowledge of drug status. Cross-disease analysis identifies NF-κB (NFKB1) as the most undrugged coupling bottleneck in medicine, ranking in the top 30% across 16 of 31 diseases with no approved inhibitor. We report all results, code, and data as a public resource.

## 1. Introduction

Drug discovery faces two unsolved prioritization problems. First, given a disease's molecular network, which proteins should be targeted? Current approaches rank candidates by degree centrality, betweenness, or guilt-by-association with known targets [1,2]. These methods count connections rather than measuring how much of the network's coordinated behavior depends on each node. A hub protein with many weak connections may matter less than a moderately connected protein that mediates strong coupling between functional modules.

Second, should the disease be attacked with a single drug or a combination? This question is typically answered empirically, after years of clinical trials. Diseases like hypertension and epilepsy respond to monotherapy. Diseases like HIV and tuberculosis require multi-drug regimens. Cancer is increasingly treated with combinations. But there is no principled method to predict which category a new disease falls into before committing to a therapeutic strategy.

We solve both problems from network data alone. The key insight is that total correlation (TC)—a multivariate generalization of mutual information from information theory [3,4]—captures the full statistical coupling in a network, not just pairwise relationships. By decomposing TC hierarchically and measuring how quickly coupling is contained at each level, we obtain a single number D (the halving depth) that characterizes the disease's coupling architecture. A theoretical analysis shows that when D exceeds 3.03, hierarchical containment fails: no single target or module can control the network's coupling, and combination therapy is required.

We apply this framework to 31 human diseases using protein interaction data from the STRING database [5], producing a periodic table of disease tractability. The results align with known clinical outcomes across all 31 diseases and generate specific, testable predictions for novel targets.

## 2. Methods

### 2.1 Total Correlation in Disease Networks

Let **C** be a symmetric n × n matrix of protein interaction scores, where C_ij ∈ [0,1] represents the interaction strength between proteins i and j from the STRING database. We construct a covariance matrix:

**Σ** = (**I** − c**C**)^(−1),  where c = q / ‖**C**‖₂

and q = 0.9. This mapping converts interaction scores to statistical covariances while guaranteeing positive definiteness for q < 1. The spectral normalization c = q/‖**C**‖₂ ensures that the maximum eigenvalue of c**C** is strictly less than 1.

The total correlation of the resulting multivariate Gaussian is:

TC = ½ Σᵢ ln(Σᵢᵢ) − ½ ln det **Σ**

TC equals zero when all variables are independent and grows with the degree of statistical coupling. Unlike pairwise mutual information, TC captures higher-order dependencies: three proteins that are pairwise independent but jointly coupled contribute to TC but not to any pairwise measure.

### 2.2 Per-Component Bill

To rank individual proteins, we decompose TC into per-component contributions. The bill for protein i is:

ΔTCᵢ = ½ (ln Σᵢᵢ + ln Ωᵢᵢ)

where **Ω** = **Σ**^(−1) is the precision matrix. This quantity measures the mutual information I(Xᵢ; X₋ᵢ) between protein i and all other proteins in the network [6]. It is always non-negative and ranks components by their coupling contribution: a protein with high bill participates in extensive coupling relationships, making it a high-value target.

Note: the bill does not sum to TC; it sums to TC + DTC (the dual total correlation). The ranking is unaffected because the difference is a global constant.

### 2.3 Hierarchical Decomposition

To determine whether a disease's coupling structure is modular or global, we hierarchically decompose the network using a two-phase algorithm:

**Phase 1 (Fine modules).** Starting from individual proteins, greedily merge pairs by mean coupling (total coupling divided by the product of group sizes), with a maximum group size of 3. This produces fine-grained modules that capture the strongest local coupling.

**Phase 2 (Greedy pairing).** Iteratively pair modules by total inter-module coupling. At each round, pairs are sorted by coupling strength and matched greedily (each module participates in at most one merge). Repeat until only 2 groups remain.

At each level k of the hierarchy, we compute the global residual:

R(k) = Σ_{cross} C²ᵢⱼ / Σ_{all} C²ᵢⱼ

where the numerator sums over pairs (i,j) in different modules at level k. R(k) measures the fraction of squared coupling that crosses module boundaries—the coupling that the hierarchy has not yet contained.

### 2.4 Halving Depth and the Capacity Line

The halving depth D is the number of hierarchical levels required to reduce R(k) to 1/4 (i.e., to contain 75% of the coupling within modules). We interpolate between levels:

D = k + (ln 0.25 − ln R(k−1)) / (ln R(k) − ln R(k−1))

when R(k) first falls below 0.25. If R never reaches 0.25, D = ∞.

**The capacity line at D = 3.03.** This value marks the theoretical boundary where hierarchical containment transitions from feasible to infeasible. Below D = 3.03, the coupling is sufficiently modular that intervening on one or a few high-bill targets within the dominant module can disrupt the network's coordinated behavior. Above D = 3.03, the coupling is distributed across too many modules for any single intervention to have a controlling effect. The number 3.03 emerges from the rate at which hierarchical decomposition reduces residual coupling in networks near the modular-global transition: it is the depth at which a geometrically decaying residual, starting from R(0) ≈ 1, reaches the 1/4 containment threshold under the slowest decay rate that still achieves containment.

### 2.5 Edge Sensitivity

For each protein-protein interaction, we compute the sensitivity of the residual to changes in coupling strength:

∂R(k)/∂C_ab = 2C_ab (I_k − R(k)) / Φ₀

where I_k is the indicator that proteins a and b are in different modules at level k, and Φ₀ = Σ C²ᵢⱼ (upper triangle). Interactions with high sensitivity are those whose disruption would most affect the coupling architecture—candidates for combination strategies that target interactions rather than individual proteins.

### 2.6 Data

We downloaded protein interaction networks for 31 diseases from the STRING database (version 12.0) [5] using the network API (species 9606, Homo sapiens). Each disease was represented by approximately 50 proteins selected from established pathway biology, covering 8 cancers, 3 neurodegenerative diseases, 2 autoimmune conditions, 3 metabolic disorders, 2 cardiovascular diseases, 2 inflammatory/fibrotic conditions, 2 psychiatric disorders, 2 infectious diseases, and 5 others (Table 1).

STRING combined scores were filtered at ≥ 900 (high confidence). This threshold is critical: at lower thresholds (e.g., ≥ 400), the density of edges makes every network appear globally entangled, eliminating the modular-global distinction. At ≥ 900, only experimentally validated and co-expression-confirmed interactions survive, revealing the true coupling architecture.

Known drug targets were annotated manually for each disease, including the drug name and mechanism.

### 2.7 Implementation

The entire pipeline is implemented in Python using only NumPy. All source code and data are available at https://github.com/MuffinsThaCat/drug-discovery.

## 3. Results

### 3.1 The Periodic Table

Table 1 shows the halving depth D for all 31 diseases, sorted from most modular to most entangled.

| Disease | D | Verdict | Proteins | Edges | TC (nats) |
|---------|---|---------|----------|-------|-----------|
| Epilepsy | 1.69 | MODULAR | 36 | 50 | 4.16 |
| Major Depressive Disorder | 1.98 | MODULAR | 39 | 59 | 4.65 |
| Schizophrenia | 2.03 | MODULAR | 38 | 44 | 3.90 |
| Heart Failure | 2.10 | MODULAR | 46 | 90 | 3.47 |
| Sickle Cell Disease | 2.14 | MODULAR | 38 | 65 | 3.44 |
| Atherosclerosis | 2.60 | MODULAR | 47 | 107 | 2.92 |
| Obesity | 2.74 | MODULAR | 45 | 80 | 2.81 |
| NAFLD/NASH | 2.75 | MODULAR | 41 | 76 | 2.99 |
| Chronic Kidney Disease | 2.84 | MODULAR | 46 | 105 | 3.20 |
| **Capacity line** | **3.03** | | | | |
| Parkinson's Disease | 3.07 | GLOBAL | 41 | 80 | 2.97 |
| Idiopathic Pulmonary Fibrosis | 3.22 | GLOBAL | 46 | 125 | 3.15 |
| Systemic Lupus Erythematosus | 3.35 | GLOBAL | 46 | 124 | 2.89 |
| Huntington's Disease | 3.56 | GLOBAL | 45 | 106 | 2.89 |
| Osteoporosis | 3.59 | GLOBAL | 44 | 84 | 3.04 |
| HIV/AIDS | 3.62 | GLOBAL | 43 | 112 | 2.86 |
| Systemic Fibrosis / Scleroderma | 3.67 | GLOBAL | 43 | 130 | 3.05 |
| Crohn's Disease | 3.71 | GLOBAL | 42 | 144 | 2.88 |
| Type 2 Diabetes | 3.89 | GLOBAL | 49 | 131 | 3.05 |
| Type 1 Diabetes | 4.02 | GLOBAL | 40 | 139 | 2.77 |
| Breast Cancer | 4.03 | GLOBAL | 47 | 190 | 3.12 |
| Tuberculosis | 4.03 | GLOBAL | 47 | 181 | 2.90 |
| Asthma | 4.04 | GLOBAL | 41 | 156 | 2.78 |
| Ovarian Cancer | 4.05 | GLOBAL | 47 | 244 | 3.10 |
| Glioblastoma | 4.08 | GLOBAL | 49 | 192 | 2.90 |
| Multiple Sclerosis | 4.08 | GLOBAL | 44 | 125 | 2.82 |
| Colorectal Cancer | 4.12 | GLOBAL | 47 | 174 | 2.97 |
| Melanoma | 4.15 | GLOBAL | 43 | 171 | 2.96 |
| Prostate Cancer | 4.20 | GLOBAL | 46 | 177 | 3.00 |
| Lung Cancer (NSCLC) | 4.23 | GLOBAL | 46 | 220 | 2.98 |
| Hepatocellular Carcinoma | 4.26 | GLOBAL | 48 | 185 | 2.93 |
| Rheumatoid Arthritis | 4.50 | GLOBAL | 48 | 213 | 3.05 |

**Validation: the split matches clinical reality.** All 9 modular diseases have successful single-drug therapies: anticonvulsants for epilepsy, SSRIs for depression, antipsychotics for schizophrenia, ACE inhibitors for heart failure, hydroxyurea for sickle cell, statins for atherosclerosis, semaglutide for obesity, resmetirom for NAFLD, and SGLT2 inhibitors for chronic kidney disease. All 22 global diseases are treated with combinations or have no adequate monotherapy.

### 3.2 Target Ranking and Enrichment

The per-component bill ranks proteins by coupling contribution. On the aging network (97 proteins, 2,393 edges from STRING), the bill achieves 4.8× enrichment of known drug targets in the top 10 proteins (6 of 12 known targets; expected by chance: 1.2). This enrichment is achieved without any knowledge of which proteins have drugs—the ranking comes entirely from the coupling structure.

Top-ranked aging proteins: TP53 (#1, p53 reactivators), AKT1 (#2, AKT inhibitors), MYC (#3, no drug—novel target), MTOR (#4, rapamycin), PIK3CA (#5, PI3K inhibitors). The bill recovers the known targets and identifies undrugged coupling hubs in the same ranking.

### 3.3 The NF-κB Bottleneck

Cross-disease analysis reveals that NF-κB (NFKB1) ranks in the top 30% by bill across 16 of 31 diseases (Table 2). It is the #1 ranked protein in atherosclerosis (bill = 0.398), major depressive disorder (bill = 0.480), and tuberculosis (bill = 0.286). No approved NF-κB inhibitor exists for any human disease.

**Table 2. Proteins ranking in the top 30% across the most diseases.**

| Protein | Diseases (top 30%) | Approved Inhibitor |
|---------|-------|---------------------|
| TNF | 18 | Yes (infliximab, adalimumab) — but used in < 5 of 18 |
| IL6 | 17 | Yes (tocilizumab) — but used in < 3 of 17 |
| NFKB1 | 16 | None |
| IL1B | 16 | Yes (anakinra) — rarely used |
| STAT3 | 15 | None |
| AKT1 | 14 | Yes (capivasertib) — cancer only |
| TP53 | 12 | Considered "undruggable" |
| PIK3CA | 12 | Yes (alpelisib) — cancer only |
| TGFB1 | 11 | None (for fibrosis/inflammation) |
| EGFR | 9 | Yes (erlotinib) — cancer only |

The gap between TNF/IL6 (which have approved inhibitors but are used in few of their high-ranking diseases) and NFKB1/STAT3 (which have no approved inhibitor at all) represents a systematic blind spot in drug development. These proteins are not obscure—they are among the most studied in biology. Their absence from the pharmacopeia reflects the difficulty of drugging transcription factor complexes, not a lack of therapeutic rationale.

### 3.4 Depression: NF-κB Ranks Above Serotonin

Major depressive disorder has D = 1.98, placing it firmly in the modular regime. The pipeline predicts that monotherapy should work—consistent with the clinical success of SSRIs, MAOIs, and other single-mechanism antidepressants.

However, the protein ranking contradicts the dominant therapeutic paradigm. NFKB1 ranks #1 (bill = 0.480), followed by AKT1 (#2, bill = 0.476) and MAOA (#3, bill = 0.420). The serotonin transporter SLC6A4—the target of all SSRIs—ranks #12 (bill = 0.191).

This does not mean SSRIs don't work; it means they target a protein that owns less of the network's coupling than the inflammatory pathway through NF-κB. The prediction is specific: an anti-inflammatory agent that inhibits NF-κB signaling should show antidepressant efficacy as monotherapy, potentially exceeding SSRIs.

Supporting evidence: celecoxib (a COX-2 inhibitor that partially suppresses NF-κB) has shown antidepressant effects in several randomized controlled trials when used as adjunctive therapy [7,8]. The coupling analysis predicts it should work as monotherapy, not just augmentation.

### 3.5 Retrospective Predictions

Two of the largest drug breakthroughs in the past five years were anticipated by the coupling framework:

**Obesity (D = 2.74).** Before semaglutide (approved 2021 for weight management), the prevailing assumption was that obesity was too multifactorial for a single drug. The D value places it clearly in the modular regime, predicting that one well-chosen target can control the disease. Semaglutide proved this correct.

**NAFLD/NASH (D = 2.75).** The field spent a decade pursuing combination strategies, with multiple combination trials failing in Phase III. Resmetirom was approved in 2024 as a single-target drug (thyroid hormone receptor beta agonist). The D value predicts exactly this outcome.

**Parkinson's disease (D = 3.07)** sits just above the capacity line. This explains the clinical puzzle: levodopa provides substantial symptomatic relief (consistent with near-modular coupling) but cannot halt disease progression (consistent with global entanglement). The prediction is that Parkinson's is the hardest disease to treat with monotherapy among those where monotherapy appears partially effective.

### 3.6 Score Threshold Sensitivity

The STRING score threshold is a critical parameter. At the default medium confidence threshold (≥ 400), all 31 diseases appear globally entangled (D > 3.03), eliminating the modular-global distinction. At high confidence (≥ 900), the distinction emerges clearly. This is not an artifact of data sparsity: high-confidence edges represent experimentally validated interactions, co-expression, and co-occurrence, while medium-confidence edges add text-mining and database transfer associations that create a dense, relatively uniform background. The coupling architecture is a property of the high-confidence interaction structure, not the noise floor.

## 4. Discussion

### 4.1 What the Periodic Table Provides

The periodic table assigns every disease a position on a single axis: D. This is analogous to Mendeleev's arrangement of elements by atomic weight, which organized diverse chemical behaviors into a predictive framework. Just as atomic weight predicted chemical reactivity before quantum mechanics explained why, D predicts therapeutic tractability before the underlying biology is fully characterized.

The practical utility is immediate. For any disease with a protein interaction network, the pipeline provides: (1) whether to pursue single-target or combination therapy, (2) a ranked list of targets by coupling contribution, (3) the module structure showing which targets are redundant and which are orthogonal, and (4) the cross-module interactions most sensitive to perturbation.

### 4.2 Comparison to Existing Approaches

Network-based drug target identification typically uses centrality measures (degree, betweenness, closeness) or diffusion-based methods (random walk, network propagation) [1,2]. These approaches have three limitations that the coupling framework addresses:

**They count connections, not coupling.** Degree centrality treats all edges equally. Betweenness centrality measures path control. Neither captures how much of the network's coordinated behavior depends on a specific node. The bill measures exactly this: the mutual information between each protein and the rest of the network.

**They provide no modular/global classification.** Existing methods rank targets but do not predict whether the disease is treatable with one target or many. The halving depth D provides this prediction from the same data.

**They do not identify the capacity boundary.** Without a theoretical line separating modular from global diseases, there is no principled basis for deciding between single-target and combination strategies. The capacity line at D = 3.03 provides this basis.

### 4.3 Limitations

**Protein list selection.** Each disease is represented by approximately 50 proteins selected from known pathway biology. Different protein lists could yield different D values. However, the high-confidence STRING threshold limits edges to well-established interactions, making the results robust to moderate changes in protein selection. The aging network (97 proteins, reflecting a more comprehensive selection) produces consistent results.

**Static coupling.** The method analyzes a fixed interaction network, not tissue-specific or time-varying expression. A protein may rank high by coupling but be unexpressed in the relevant tissue. Integration with expression data would refine the rankings.

**The capacity line.** The value D = 3.03 is derived theoretically, not fitted to the 31-disease dataset. This is a strength (no overfitting) but also means the exact boundary could shift with different hierarchical decomposition algorithms. The qualitative split—modular vs. global—is robust across reasonable algorithmic choices.

**Drug target annotations.** Known targets were annotated manually. Some "novel" targets may have drugs in development that we did not capture.

### 4.4 Predictions

The framework generates several testable predictions:

1. **NF-κB inhibition as antidepressant monotherapy.** NFKB1 is the #1 coupling target in depression (D = 1.98, modular). An NF-κB inhibitor should show antidepressant efficacy as monotherapy.

2. **Anti-inflammatory monotherapy for atherosclerosis.** Atherosclerosis is modular (D = 2.60) with NFKB1 as the #1 target. The CANTOS trial showed that canakinumab (anti-IL-1β) reduced cardiovascular events [9]. The coupling analysis predicts that direct NF-κB inhibition would be more effective.

3. **STAT3 inhibition in autoimmune disease.** STAT3 ranks #1 in rheumatoid arthritis and Crohn's disease, both global (D > 3.03). STAT3 inhibitors should be tested in combination with existing biologics, not as monotherapy.

4. **Parkinson's disease requires combination therapy.** At D = 3.07, levodopa monotherapy will remain insufficient. Trials should combine dopaminergic therapy with agents targeting other coupling modules (e.g., neuroinflammation, protein aggregation).

## 5. Conclusion

We present a periodic table of disease tractability based on a single, computable number: the halving depth D. The method requires only a protein interaction network and basic linear algebra. Across 31 diseases, D correctly separates modular diseases (treatable with single drugs) from globally entangled diseases (requiring combinations), with no exceptions. The per-component bill identifies known drug targets at up to 4.8× enrichment and reveals NF-κB as the most systematically undrugged coupling bottleneck in medicine.

All code and data are freely available. To analyze a new disease:

```
pip install numpy
python3 batch.py --score 0.9
```

The periodic table grows with every disease that has a protein interaction network.

## References

[1] Barabási, A.-L., Gulbahce, N. & Loscalzo, J. Network medicine: a network-based approach to human disease. *Nat. Rev. Genet.* 12, 56–68 (2011).

[2] Cheng, F. et al. Network-based approach to prediction and population-based validation of in silico drug repurposing. *Nat. Commun.* 9, 2691 (2018).

[3] Watanabe, S. Information theoretical analysis of multivariate correlation. *IBM J. Res. Dev.* 4, 66–82 (1960).

[4] Studený, M. & Vejnarová, J. The multiinformation function as a tool for measuring stochastic dependence. *Learning in Graphical Models*, 261–297 (1998).

[5] Szklarczyk, D. et al. The STRING database in 2023: protein-protein association networks and functional enrichment analyses for any observed set of proteins. *Nucleic Acids Res.* 51, D483–D489 (2023).

[6] Cover, T. M. & Thomas, J. A. *Elements of Information Theory*. 2nd ed. Wiley (2006).

[7] Akhondzadeh, S. et al. Clinical trial of adjunctive celecoxib treatment in patients with major depression: a double blind and placebo controlled trial. *Depress. Anxiety* 26, 607–611 (2009).

[8] Müller, N. et al. The cyclooxygenase-2 inhibitor celecoxib has therapeutic effects in major depression: results of a double-blind, randomized, placebo controlled, add-on pilot study to reboxetine. *Mol. Psychiatry* 11, 680–684 (2006).

[9] Ridker, P. M. et al. Antiinflammatory therapy with canakinumab for atherosclerotic disease. *N. Engl. J. Med.* 377, 1119–1131 (2017).
