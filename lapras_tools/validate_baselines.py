"""Compare D against standard graph-theory baselines for predicting mono/combo.

Tests whether D outperforms or matches betweenness centrality, degree centrality,
PageRank concentration, clustering coefficient, modularity, and other network
metrics at predicting monotherapy vs combination standard of care.

    python3 -m lapras_tools.validate_baselines
"""
import numpy as np
import networkx as nx
from .core import load_disease, analyze_network, LINE, disease_keys
from .validate_clinical import LABELS, CANCERS, auc_mann_whitney


def graph_metrics(C, names):
    """Compute graph-theory metrics from coupling matrix."""
    n = len(names)
    G = nx.Graph()
    for i in range(n):
        G.add_node(names[i])
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] > 0:
                G.add_edge(names[i], names[j], weight=C[i, j])

    n_edges = G.number_of_edges()
    if n_edges == 0:
        return None

    deg = dict(G.degree(weight="weight"))
    bc = nx.betweenness_centrality(G, weight="weight")
    try:
        pr = nx.pagerank(G, weight="weight", max_iter=200)
    except nx.PowerIterationFailedConvergence:
        pr = {node: 1.0 / n for node in G.nodes()}
    cc = nx.clustering(G, weight="weight")

    deg_vals = np.array(list(deg.values()))
    bc_vals = np.array(list(bc.values()))
    pr_vals = np.array(list(pr.values()))
    cc_vals = np.array(list(cc.values()))

    try:
        components = list(nx.connected_components(G))
        if len(components) == 1:
            close = nx.closeness_centrality(G)
            close_mean = np.mean(list(close.values()))
        else:
            close_mean = 0.0
    except Exception:
        close_mean = 0.0

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = greedy_modularity_communities(G, weight="weight")
        modularity = nx.community.modularity(G, comms, weight="weight")
    except Exception:
        modularity = 0.0

    def gini(x):
        x = np.sort(x)
        n = len(x)
        if n == 0 or x.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return (2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))

    return {
        "n": n,
        "edges": n_edges,
        "density": 2 * n_edges / (n * (n - 1)) if n > 1 else 0,
        "deg_mean": deg_vals.mean(),
        "deg_max": deg_vals.max(),
        "deg_gini": gini(deg_vals),
        "bc_mean": bc_vals.mean(),
        "bc_max": bc_vals.max(),
        "bc_gini": gini(bc_vals),
        "pr_max": pr_vals.max(),
        "pr_gini": gini(pr_vals),
        "clustering": cc_vals.mean(),
        "closeness_mean": close_mean,
        "modularity": modularity,
    }


def load_all():
    rows = []
    for k in disease_keys():
        if k not in LABELS:
            continue
        C, names, _, label = load_disease(k, 0.9)
        an = analyze_network(C, names)
        gm = graph_metrics(C, names)
        if gm is None:
            continue
        gm["key"] = k
        gm["label"] = label
        gm["D"] = an.D
        gm["tc"] = an.tc
        gm["y"] = LABELS[k]
        rows.append(gm)
    return rows


def main():
    rows = load_all()
    Y = np.array([r["y"] for r in rows])

    predictors = [
        ("D (halving depth)",       [r["D"] for r in rows]),
        ("edges",                   [r["edges"] for r in rows]),
        ("edge density",            [r["density"] for r in rows]),
        ("n (proteins)",            [r["n"] for r in rows]),
        ("TC (total correlation)",  [r["tc"] for r in rows]),
        ("degree mean",             [r["deg_mean"] for r in rows]),
        ("degree max",              [r["deg_max"] for r in rows]),
        ("degree Gini",             [r["deg_gini"] for r in rows]),
        ("betweenness mean",        [r["bc_mean"] for r in rows]),
        ("betweenness max",         [r["bc_max"] for r in rows]),
        ("betweenness Gini",        [r["bc_gini"] for r in rows]),
        ("PageRank max",            [r["pr_max"] for r in rows]),
        ("PageRank Gini",           [r["pr_gini"] for r in rows]),
        ("clustering coeff",        [r["clustering"] for r in rows]),
        ("closeness mean",          [r["closeness_mean"] for r in rows]),
        ("modularity",              [r["modularity"] for r in rows]),
    ]

    nc_mask = np.array([r["key"] not in CANCERS for r in rows])

    print("=" * 90)
    print("BASELINE COMPARISON: D vs graph-theory metrics for predicting mono/combo")
    print(f"n = {len(rows)} diseases ({sum(Y == 0)} mono, {sum(Y == 1)} combo)")
    print("=" * 90)

    print(f"\n  {'Predictor':25s} {'AUC (all)':>10} {'p':>8} {'AUC (non-ca)':>14} {'p':>8}")
    print("  " + "-" * 69)

    results = []
    for name, vals in predictors:
        v = np.array(vals, float)
        a1, p1 = auc_mann_whitney(v[Y == 0], v[Y == 1])
        v_nc = v[nc_mask]; y_nc = Y[nc_mask]
        a2, p2 = auc_mann_whitney(v_nc[y_nc == 0], v_nc[y_nc == 1])
        print(f"  {name:25s} {a1:10.3f} {p1:8.4f} {a2:14.3f} {p2:8.4f}")
        results.append((name, a1, p1, a2, p2))

    # residual analysis: regress out each top baseline from D
    print(f"\n{'=' * 90}")
    print("RESIDUAL ANALYSIS: does D add information beyond each baseline?")
    print("=" * 90)

    D = np.array([r["D"] for r in rows], float)
    for name, vals in predictors:
        if name == "D (halving depth)":
            continue
        v = np.array(vals, float)
        X = np.column_stack([np.ones(len(v)), v])
        beta = np.linalg.lstsq(X, D, rcond=None)[0]
        D_resid = D - X @ beta
        auc_r, p_r = auc_mann_whitney(D_resid[Y == 0], D_resid[Y == 1])
        signal = "YES" if p_r < 0.05 else "no"
        print(f"  D | {name:25s}  residual AUC = {auc_r:.3f}  p = {p_r:.4f}  "
              f"adds signal: {signal}")

    # reverse: does each baseline add beyond D?
    print(f"\n{'=' * 90}")
    print("REVERSE RESIDUAL: does each baseline add information beyond D?")
    print("=" * 90)

    for name, vals in predictors:
        if name == "D (halving depth)":
            continue
        v = np.array(vals, float)
        X = np.column_stack([np.ones(len(D)), D])
        beta = np.linalg.lstsq(X, v, rcond=None)[0]
        v_resid = v - X @ beta
        auc_r, p_r = auc_mann_whitney(v_resid[Y == 0], v_resid[Y == 1])
        signal = "YES" if p_r < 0.05 else "no"
        print(f"  {name:25s} | D  residual AUC = {auc_r:.3f}  p = {p_r:.4f}  "
              f"adds signal: {signal}")

    # rank all predictors
    print(f"\n{'=' * 90}")
    print("RANKING (all diseases, by AUC)")
    print("=" * 90)
    ranked = sorted(results, key=lambda r: -r[1])
    for i, (name, a1, p1, a2, p2) in enumerate(ranked, 1):
        marker = " <-- D" if name == "D (halving depth)" else ""
        print(f"  {i:2d}. {name:25s} AUC = {a1:.3f}{marker}")

    print(f"\n{'=' * 90}")
    print("RANKING (non-cancer only, by AUC)")
    print("=" * 90)
    ranked_nc = sorted(results, key=lambda r: -r[3])
    for i, (name, a1, p1, a2, p2) in enumerate(ranked_nc, 1):
        marker = " <-- D" if name == "D (halving depth)" else ""
        print(f"  {i:2d}. {name:25s} AUC = {a2:.3f}{marker}")


if __name__ == "__main__":
    main()
