"""Tool 7 - Build coupling matrices from non-STRING sources.

For diseases without well-studied STRING networks, constructs a coupling
matrix from alternative data: co-expression (RNA-seq), generic interaction
files, pathway co-membership, or a combination.

    python3 -m lapras_tools.network_builder --genes genes.txt --coexpr samples.tsv
    python3 -m lapras_tools.network_builder --genes genes.txt --coexpr-npz data/tcga_brca_tpm.npz
    python3 -m lapras_tools.network_builder --genes genes.txt --edges interactions.tsv
    python3 -m lapras_tools.network_builder --genes genes.txt --pathways gene_pathways.tsv
    python3 -m lapras_tools.network_builder --genes genes.txt --coexpr-npz a.npz --edges b.tsv --merge
    python3 -m lapras_tools.network_builder --genes genes.txt --coexpr-npz data/tcga_brca_tpm.npz --out data/my_disease.tsv
    python3 -m lapras_tools.network_builder --demo breast_cancer

Output: a TSV compatible with load_string_network, plus the Lapras analysis.
Add the TSV to data/ and an entry to diseases.py to bring a new disease into
the full Lapras pipeline (combo_designer, patient, variant_layer, etc.).

Gene list: one gene symbol per line, or a TSV with gene in the first column.

Co-expression: expression matrix (TSV: samples x genes, first column = sample
ID) or .npz with 'genes' and 'tpm' arrays (like the TCGA data already in
data/).  Coupling = |Spearman rho|, thresholded at --threshold percentile.

Edges: 3-column TSV (gene1, gene2, score in [0,1]).  From BioGRID, IntAct,
FunCoup, HumanNet, or any source.

Pathways: 2-column TSV (gene, pathway_name).  Two genes in the same pathway
get coupling proportional to 1/sqrt(pathway_size); multiple shared pathways
are combined with the STRING formula.

Merge: combines multiple sources with naive Bayes-like score integration:
  combined = 1 - prod(1 - s_i)  for each source s_i in [0, 1]
(the same formula STRING uses internally for its evidence channels).
"""
import argparse, csv, os
import numpy as np
from .core import analyze_network, LINE, DATA_DIR


def read_gene_list(path):
    """Read gene symbols from a file (one per line, or first column of TSV)."""
    genes = []
    with open(path) as f:
        for line in f:
            g = line.strip().split("\t")[0].split(",")[0].strip().upper()
            if g and not g.startswith("#"):
                genes.append(g)
    return list(dict.fromkeys(genes))  # deduplicate, preserve order


# ---------------------------------------------------------------- ranking

def _rank_column(col):
    """Average ranks for one array (ties share the mean rank)."""
    order = np.argsort(col, kind="mergesort")
    r = np.empty(len(col), dtype=float)
    r[order] = np.arange(len(col), dtype=float)
    for v in np.unique(col):
        m = col == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def _rank_columns(X):
    """Rank each column of X."""
    return np.column_stack([_rank_column(X[:, j])
                            for j in range(X.shape[1])])


def _corr_matrix(X):
    """Pearson correlation matrix of columns of X."""
    X = X - X.mean(axis=0)
    norms = np.sqrt((X * X).sum(axis=0))
    norms[norms == 0] = 1.0
    X = X / norms
    return X.T @ X


# ---------------------------------------------------------------- builders

def from_coexpression(expr_path, genes, method="spearman", threshold=0.7):
    """Build C from expression matrix (TSV: samples x genes, header = gene names).
    Returns (C, names)."""
    with open(expr_path) as f:
        sniff = f.read(2048); f.seek(0)
        delim = "\t" if sniff.count("\t") >= sniff.count(",") else ","
        reader = csv.reader(f, delimiter=delim)
        header = next(reader)
        col_genes = [h.strip().upper() for h in header[1:]]
        rows = []
        n_dropped = 0
        for row in reader:
            if len(row) < 2:
                continue
            try:
                rows.append([float(x) for x in row[1:]])
            except ValueError:
                n_dropped += 1
                continue
    if n_dropped:
        print(f"  WARNING: {n_dropped} sample row(s) dropped "
              f"(non-numeric values)")
    expr = np.array(rows)
    return _coexpr_from_matrix(expr, col_genes, genes, method, threshold)


def from_coexpression_npz(npz_path, genes, method="spearman", threshold=0.7):
    """Build C from .npz with 'genes' and 'tpm' arrays (TCGA format)."""
    z = np.load(npz_path)
    all_genes = [g.upper() for g in z["genes"]]
    tpm = z["tpm"].astype(float)
    return _coexpr_from_matrix(tpm, all_genes, genes, method, threshold)


def _coexpr_from_matrix(expr, col_genes, target_genes, method, threshold):
    """Shared implementation: expression matrix + column gene names -> (C, names)."""
    gene_set = set(g.upper() for g in target_genes)
    col_idx = {g: i for i, g in enumerate(col_genes)}
    keep = [(col_idx[g], g) for g in gene_set if g in col_idx]
    if len(keep) < 3:
        raise ValueError(f"only {len(keep)} of {len(target_genes)} genes "
                         "found in expression data (need >= 3)")

    indices, names = zip(*sorted(keep, key=lambda x: x[1]))
    names = list(names)
    sub = expr[:, list(indices)]

    if method == "spearman":
        R = _corr_matrix(_rank_columns(sub))
    else:
        R = _corr_matrix(np.log2(sub + 1))

    C = np.abs(R)
    np.fill_diagonal(C, 0.0)

    n = len(names)
    vals = C[np.triu_indices(n, 1)]
    cutoff = np.percentile(vals, threshold * 100)
    C[C < cutoff] = 0.0

    print(f"  co-expression: {len(names)}/{len(target_genes)} genes found, "
          f"{expr.shape[0]} samples, threshold={cutoff:.3f} "
          f"(top {(1 - threshold) * 100:.0f}% of pairs)")
    return C, names


def from_edges(path, genes=None, min_score=0.0):
    """Load 3-column TSV (gene1, gene2, score) as coupling matrix."""
    edges = {}
    all_genes = set()
    gene_set = set(g.upper() for g in genes) if genes else None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split(",")
            if len(parts) < 3:
                continue
            a, b = parts[0].strip().upper(), parts[1].strip().upper()
            try:
                score = float(parts[2])
            except ValueError:
                continue
            if a == b or score < min_score:
                continue
            if gene_set and (a not in gene_set or b not in gene_set):
                continue
            all_genes.update((a, b))
            key = tuple(sorted((a, b)))
            edges[key] = max(edges.get(key, 0), score)

    names = sorted(all_genes)
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    C = np.zeros((n, n))
    for (a, b), s in edges.items():
        C[idx[a], idx[b]] = C[idx[b], idx[a]] = s

    print(f"  edges: {n} genes, {len(edges)} edges from "
          f"{os.path.basename(path)}")
    return C, names


def from_pathways(path, genes):
    """Two genes in the same pathway -> coupling ~ 1/sqrt(size).
    Multiple shared pathways combined with STRING's naive Bayes formula."""
    gene_pathways = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split(",")
            if len(parts) < 2:
                continue
            gene = parts[0].strip().upper()
            pathway = parts[1].strip()
            gene_pathways.setdefault(gene, set()).add(pathway)

    gene_set = set(g.upper() for g in genes)
    pathway_genes = {}
    for g, ps in gene_pathways.items():
        if g not in gene_set:
            continue
        for p in ps:
            pathway_genes.setdefault(p, set()).add(g)
    pathway_genes = {p: gs for p, gs in pathway_genes.items() if len(gs) >= 2}

    found = set()
    for gs in pathway_genes.values():
        found.update(gs)
    names = sorted(found)
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)

    pair_scores = {}
    for pw, gs in pathway_genes.items():
        gs = sorted(gs)
        s = 1.0 / np.sqrt(len(gs))
        for a in range(len(gs)):
            for b in range(a + 1, len(gs)):
                key = (idx[gs[a]], idx[gs[b]])
                pair_scores.setdefault(key, []).append(s)

    C = np.zeros((n, n))
    for (i, j), scores in pair_scores.items():
        combined = 1.0 - np.prod([1.0 - s for s in scores])
        C[i, j] = C[j, i] = min(combined, 1.0)

    print(f"  pathways: {n} genes in {len(pathway_genes)} pathways, "
          f"{int((C > 0).sum() // 2)} edges")
    return C, names


# ---------------------------------------------------------------- merge

def merge_networks(networks):
    """Combine multiple (C, names) with STRING-style naive Bayes:
    combined = 1 - prod(1 - s_k)."""
    all_names = set()
    for _, names in networks:
        all_names.update(names)
    merged_names = sorted(all_names)
    idx = {n: i for i, n in enumerate(merged_names)}
    n = len(merged_names)

    complement = np.ones((n, n))
    for C, names in networks:
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                s = C[a, b]
                if s > 0:
                    i, j = idx[names[a]], idx[names[b]]
                    complement[i, j] *= (1.0 - s)
                    complement[j, i] *= (1.0 - s)

    C_merged = 1.0 - complement
    np.fill_diagonal(C_merged, 0.0)

    n_edges = int((C_merged > 0).sum() // 2)
    print(f"  merged: {n} genes, {n_edges} edges "
          f"(from {len(networks)} sources)")
    return C_merged, merged_names


# ---------------------------------------------------------------- output

def write_string_tsv(C, names, outpath):
    """Write in download_string.py format for load_string_network."""
    n_edges = 0
    with open(outpath, "w") as f:
        f.write("stringId_A\tstringId_B\tpreferredName_A\t"
                "preferredName_B\tncbitaxid\tscore\n")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if C[i, j] > 0:
                    f.write(f".\t.\t{names[i]}\t{names[j]}\t9606\t"
                            f"{C[i, j]:.4f}\n")
                    n_edges += 1
    print(f"  written: {outpath} ({n_edges} edges)")
    return outpath


def report_network(C, names, label="custom network"):
    """Analyze and print summary."""
    n = len(names)
    n_edges = int((C > 0).sum() // 2)
    density = n_edges / (n * (n - 1) / 2) if n > 1 else 0
    print(f"\n  {label}: {n} proteins, {n_edges} edges "
          f"(density {density:.3f})")

    if n < 3 or not (C > 0).any():
        print("    network too sparse for Lapras analysis "
              "(need >= 3 connected nodes)")
        return None

    try:
        an = analyze_network(C, names)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    analysis failed: {e}")
        return None

    print(f"    D = {an.D:.2f} ({an.verdict}), TC = {an.tc:.2f}")
    print(f"    top-10 by bill: "
          f"{', '.join(an.names[i] for i in an.ranked()[:10])}")
    print(f"    hierarchy: {len(an.levels)} levels, "
          f"coarsest = {len(an.levels[-1]) if an.levels else 0} modules")
    return an


# ---------------------------------------------------------------- demo

def run_demo(disease, score=0.9):
    """Build a co-expression network for a disease that already has TCGA data
    and compare it to the STRING network."""
    from .core import load_disease
    C_string, names_string, _, label = load_disease(disease, score)
    an_string = analyze_network(C_string, names_string)

    npz_map = {
        "breast_cancer": "tcga_brca_tpm.npz",
        "lung_cancer": "tcga_luad_tpm.npz",
    }
    npz_name = npz_map.get(disease)
    if not npz_name:
        print(f"no TCGA data bundled for {disease} "
              f"(available: {', '.join(npz_map)})")
        return

    npz_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", npz_name)
    if not os.path.exists(npz_path):
        print(f"TCGA file not found: {npz_path}")
        return

    print(f"\n{label}: STRING network (score >= {score})")
    print(f"  D = {an_string.D:.2f} ({an_string.verdict}), "
          f"{len(names_string)} proteins, "
          f"{int((C_string > 0).sum() // 2)} edges")
    print(f"  top-10: "
          f"{', '.join(an_string.names[i] for i in an_string.ranked()[:10])}")

    from diseases import DISEASES
    genes = DISEASES[disease]["proteins"]

    for threshold in (0.6, 0.7, 0.8):
        print(f"\nco-expression network (Spearman, threshold={threshold}):")
        C_co, names_co = from_coexpression_npz(
            npz_path, genes, "spearman", threshold)
        an_co = report_network(C_co, names_co,
                               f"co-expression (threshold={threshold})")
        if an_co is not None:
            overlap = set(names_string) & set(names_co)
            if len(overlap) >= 3:
                top_s = set(an_string.names[i]
                            for i in an_string.ranked()[:10])
                top_c = set(an_co.names[i]
                            for i in an_co.ranked()[:10])
                print(f"    top-10 overlap with STRING: "
                      f"{len(top_s & top_c)}/10")

    print(f"\nmerged (STRING + co-expression at 0.7):")
    C_co, names_co = from_coexpression_npz(npz_path, genes, "spearman", 0.7)
    C_merged, names_merged = merge_networks([
        (C_string, names_string), (C_co, names_co)])
    report_network(C_merged, names_merged, "STRING + co-expression merged")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", metavar="PATH",
                    help="gene list file (one per line)")
    ap.add_argument("--coexpr", metavar="PATH",
                    help="expression TSV (samples x genes)")
    ap.add_argument("--coexpr-npz", metavar="PATH",
                    help="expression .npz with genes + tpm")
    ap.add_argument("--edges", metavar="PATH",
                    help="interaction file (3-col TSV)")
    ap.add_argument("--pathways", metavar="PATH",
                    help="gene-pathway mapping (2-col TSV)")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="co-expression percentile (default 0.7 = top 30%%)")
    ap.add_argument("--method", choices=["spearman", "pearson"],
                    default="spearman")
    ap.add_argument("--merge", action="store_true",
                    help="combine all sources with naive Bayes")
    ap.add_argument("--out", metavar="PATH", help="output TSV path")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--demo", metavar="DISEASE",
                    help="compare STRING vs co-expression for a disease "
                         "(breast_cancer, lung_cancer)")
    a = ap.parse_args()

    if a.demo:
        run_demo(a.demo)
        return

    if not a.genes:
        ap.error("give --genes or --demo")

    genes = read_gene_list(a.genes)
    print(f"gene list: {len(genes)} genes from {a.genes}")

    networks = []
    labels = []

    if a.coexpr:
        C, names = from_coexpression(a.coexpr, genes, a.method, a.threshold)
        networks.append((C, names))
        labels.append("co-expression")
        report_network(C, names,
                       f"co-expression ({a.method}, "
                       f"threshold={a.threshold})")

    if a.coexpr_npz:
        C, names = from_coexpression_npz(
            a.coexpr_npz, genes, a.method, a.threshold)
        networks.append((C, names))
        labels.append("co-expression (npz)")
        report_network(C, names,
                       f"co-expression from npz ({a.method}, "
                       f"threshold={a.threshold})")

    if a.edges:
        C, names = from_edges(a.edges, genes, a.min_score)
        networks.append((C, names))
        labels.append("edges")
        report_network(C, names,
                       f"edges from {os.path.basename(a.edges)}")

    if a.pathways:
        C, names = from_pathways(a.pathways, genes)
        networks.append((C, names))
        labels.append("pathways")
        report_network(C, names, "pathway co-membership")

    if not networks:
        ap.error("give at least one of --coexpr, --coexpr-npz, "
                 "--edges, --pathways")

    if a.merge and len(networks) > 1:
        C_final, names_final = merge_networks(networks)
        report_network(C_final, names_final,
                       f"merged ({' + '.join(labels)})")
    elif len(networks) == 1:
        C_final, names_final = networks[0]
    else:
        print("\nmultiple sources provided but --merge not set; "
              "using the first source")
        C_final, names_final = networks[0]

    if a.out:
        write_string_tsv(C_final, names_final, a.out)
        print(f"\nto use this network with Lapras:")
        print(f"  1. add a diseases.py entry with known_targets")
        print(f"  2. run: python3 -m lapras_tools.combo_designer "
              f"<disease_key>")


if __name__ == "__main__":
    main()
