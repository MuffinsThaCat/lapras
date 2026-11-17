"""Algorithmic therapy taxonomy: predict bypass / modulation / multi-system from network features.

Instead of assigning the taxonomy post-hoc from clinical knowledge, derive it
from computable network features.  This turns the taxonomy from a descriptive
observation into a predictive tool for new diseases.

Features used:
  - is_monogenic: does the disease have a single causal gene? (from diseases.py)
  - is_replacement: is the standard therapy a replacement? (from clinical annotation)
  - D: halving depth
  - edges: number of STRING interactions
  - bill_concentration: Gini coefficient of the bill distribution
  - top_bill_dispersion: how spread are the top-5 bill proteins across modules?
  - modularity: graph modularity (Louvain)

    python3 -m lapras_tools.taxonomy_predict
    python3 -m lapras_tools.taxonomy_predict --predict <disease_key>
"""
import numpy as np
import networkx as nx
from .core import load_disease, analyze_network, LINE, disease_keys
from .validate_clinical import (
    LABELS, MONOGENIC, REPLACEMENT, CANCERS,
    auc_mann_whitney,
)


def gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


def top_bill_dispersion(an, n_top=5):
    """How spread are top-5 bill proteins across hierarchy modules?
    Returns fraction of distinct modules they occupy (0-1; 1 = maximally dispersed)."""
    ranked = an.ranked()[:n_top]
    if len(an.levels) == 0:
        return 0.0
    part = an.level_with_at_most(max(5, len(an.levels[-1])))
    labels = an.label_vector(part)
    modules_hit = set(labels[i] for i in ranked if labels[i] >= 0)
    return len(modules_hit) / min(n_top, len(part))


def graph_modularity(C, names):
    n = len(names)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] > 0:
                G.add_edge(i, j, weight=C[i, j])
    if G.number_of_edges() == 0:
        return 0.0
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = greedy_modularity_communities(G, weight="weight")
        return nx.community.modularity(G, comms, weight="weight")
    except Exception:
        return 0.0


MULTI_SYSTEM = {"heart_failure", "atherosclerosis", "chronic_kidney"}


def extract_features(key, C, names, an):
    """Extract taxonomy-relevant features for one disease."""
    edges = int((C > 0).sum()) // 2
    return {
        "key": key,
        "is_monogenic": 1 if key in MONOGENIC else 0,
        "is_replacement": 1 if key in REPLACEMENT else 0,
        "is_bypass": 1 if key in MONOGENIC or key in REPLACEMENT else 0,
        "D": an.D,
        "edges": edges,
        "n": len(names),
        "tc": an.tc,
        "bill_gini": gini(an.bill),
        "bill_top5_share": float(np.sort(an.bill)[-5:].sum() / an.bill.sum()) if an.bill.sum() > 0 else 0,
        "top_bill_dispersion": top_bill_dispersion(an),
        "modularity": graph_modularity(C, names),
    }


def classify_taxonomy(features):
    """Rule-based taxonomy classifier derived from the 43-disease training set.

    Rules (in priority order):
    1. If monogenic OR replacement → BYPASS
    2. If D <= 3.03 AND modularity > 0.15 AND bill_gini < 0.6 → MULTI-SYSTEM CANDIDATE
       (the combination targets different physiological systems, not modules)
    3. Everything else → MODULATION
    """
    if features["is_monogenic"] or features["is_replacement"]:
        return "bypass", "monogenic/replacement therapy — acts upstream of the network"

    if features["D"] <= LINE:
        if features["modularity"] > 0.15 and features["bill_gini"] < 0.6:
            return "multi-system-candidate", (
                f"MODULAR (D={features['D']:.2f}) but bill is diffuse "
                f"(Gini={features['bill_gini']:.2f}) with moderate modularity "
                f"({features['modularity']:.2f}) — check if combination targets "
                f"different organ systems")
        return "modulation-mono", f"MODULAR (D={features['D']:.2f}) — single-target candidate"

    return "modulation-combo", f"GLOBAL (D={features['D']:.2f}) — combination likely needed"


def load_training_set():
    """Load the 43 clinical-validation diseases with ground-truth taxonomy."""
    rows = []
    for k in disease_keys():
        if k not in LABELS:
            continue
        C, names, _, label = load_disease(k, 0.9)
        an = analyze_network(C, names)
        feat = extract_features(k, C, names, an)

        # ground truth taxonomy
        if k in MONOGENIC or k in REPLACEMENT:
            feat["true_class"] = "bypass"
        elif k in MULTI_SYSTEM:
            feat["true_class"] = "multi-system"
        else:
            feat["true_class"] = "modulation"

        feat["y"] = LABELS[k]
        feat["label"] = label
        rows.append(feat)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", metavar="DISEASE",
                    help="predict taxonomy for a specific disease")
    a = ap.parse_args()

    if a.predict:
        C, names, _, label = load_disease(a.predict, 0.9)
        an = analyze_network(C, names)
        feat = extract_features(a.predict, C, names, an)
        cls, reason = classify_taxonomy(feat)
        print(f"\n  {label}: {cls}")
        print(f"  {reason}")
        print(f"  Features: D={feat['D']:.2f}, edges={feat['edges']}, "
              f"bill_gini={feat['bill_gini']:.2f}, "
              f"modularity={feat['modularity']:.2f}, "
              f"dispersion={feat['top_bill_dispersion']:.2f}")
        print(f"  Top-5 bill: {', '.join(an.names[i] for i in an.ranked()[:5])}")
        return

    rows = load_training_set()

    print("=" * 100)
    print("ALGORITHMIC TAXONOMY: predicting bypass / modulation / multi-system from network features")
    print("=" * 100)

    # feature distributions by class
    for cls_name in ["bypass", "modulation", "multi-system"]:
        subset = [r for r in rows if r["true_class"] == cls_name]
        if not subset:
            continue
        print(f"\n  {cls_name.upper()} (n={len(subset)}):")
        for feat in ["D", "edges", "bill_gini", "bill_top5_share",
                      "top_bill_dispersion", "modularity"]:
            vals = [r[feat] for r in subset]
            print(f"    {feat:25s} mean={np.mean(vals):.3f}  "
                  f"range=[{min(vals):.3f}, {max(vals):.3f}]")

    # test the classifier
    print(f"\n{'=' * 100}")
    print("CLASSIFIER ACCURACY ON TRAINING SET")
    print("=" * 100)

    correct = 0
    total = 0
    errors = []
    for r in sorted(rows, key=lambda r: r["D"]):
        pred_cls, reason = classify_taxonomy(r)
        true_cls = r["true_class"]

        # map predictions to ground truth labels
        match = False
        if true_cls == "bypass" and pred_cls == "bypass":
            match = True
        elif true_cls == "multi-system" and pred_cls == "multi-system-candidate":
            match = True
        elif true_cls == "modulation" and pred_cls.startswith("modulation"):
            match = True

        total += 1
        if match:
            correct += 1
        else:
            errors.append((r["label"], r["key"], true_cls, pred_cls))

        status = "OK" if match else "MISS"
        print(f"  {status:4s} {r['label'][:40]:40s} true={true_cls:12s} "
              f"pred={pred_cls:22s} D={r['D']:.2f}")

    print(f"\n  Accuracy: {correct}/{total} = {correct/total:.1%}")
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for label, key, true, pred in errors:
            print(f"    {label}: true={true}, predicted={pred}")

    # feature importance: which features separate bypass from modulation?
    print(f"\n{'=' * 100}")
    print("FEATURE IMPORTANCE: separating bypass from modulation")
    print("=" * 100)

    bypass = [r for r in rows if r["true_class"] == "bypass"]
    modulate = [r for r in rows if r["true_class"] != "bypass"]

    for feat in ["D", "edges", "bill_gini", "bill_top5_share",
                  "top_bill_dispersion", "modularity", "is_monogenic", "is_replacement"]:
        v0 = [r[feat] for r in bypass]
        v1 = [r[feat] for r in modulate]
        auc, p = auc_mann_whitney(v0, v1)
        print(f"  {feat:25s} AUC={auc:.3f} p={p:.4f}  "
              f"bypass={np.mean(v0):.3f} vs modulation={np.mean(v1):.3f}")


if __name__ == "__main__":
    main()
