"""Size-matched analysis: does D add value within diseases of similar edge count?

The edge-count confound says D adds nothing beyond network size. This script
tests whether coupling STRUCTURE matters by finding disease pairs with similar
edge counts but different D values and different clinical outcomes.

    python3 -m lapras_tools.validate_size_matched
"""
import numpy as np
from .core import load_disease, analyze_network, LINE, disease_keys
from .validate_clinical import LABELS, auc_mann_whitney


def load_all():
    rows = []
    for k in disease_keys():
        if k not in LABELS:
            continue
        C, names, _, label = load_disease(k, 0.9)
        an = analyze_network(C, names)
        edges = int((C > 0).sum()) // 2
        rows.append(dict(
            key=k, label=label, D=an.D, edges=edges, n=len(names),
            y=LABELS[k], tc=an.tc,
        ))
    return rows


def main():
    rows = load_all()
    rows.sort(key=lambda r: r["edges"])

    print("=" * 90)
    print("SIZE-MATCHED ANALYSIS: does D add value within similar-size diseases?")
    print("=" * 90)

    # find discordant pairs: similar edges, different outcomes
    print("\n  DISCORDANT PAIRS: similar edge count, different clinical outcome")
    print("  (edge difference <= 15, one mono and one combo)")
    print(f"\n  {'Disease A':35s} {'D':>5} {'e':>5} {'y':>5}   {'Disease B':35s} {'D':>5} {'e':>5} {'y':>5}   ΔD")
    print("  " + "-" * 115)

    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i]["y"] == rows[j]["y"]:
                continue
            edge_diff = abs(rows[i]["edges"] - rows[j]["edges"])
            if edge_diff > 15:
                continue
            d_diff = rows[j]["D"] - rows[i]["D"]
            mono = rows[i] if rows[i]["y"] == 0 else rows[j]
            combo = rows[j] if rows[j]["y"] == 1 else rows[i]
            concordant = combo["D"] > mono["D"]
            pairs.append((mono, combo, concordant))

            tag = "D agrees" if concordant else "D DISAGREES"
            print(f"  {mono['label'][:35]:35s} {mono['D']:5.2f} {mono['edges']:5d} {'mono':>5}   "
                  f"{combo['label'][:35]:35s} {combo['D']:5.2f} {combo['edges']:5d} {'combo':>5}   "
                  f"{combo['D'] - mono['D']:+5.2f} [{tag}]")

    if pairs:
        n_concordant = sum(1 for _, _, c in pairs if c)
        print(f"\n  {n_concordant}/{len(pairs)} concordant pairs "
              f"({n_concordant/len(pairs):.0%}): D correctly ranks combo > mono "
              f"when edge counts are similar")

    # bin by edge count and test D within bins
    print(f"\n{'=' * 90}")
    print("WITHIN-BIN AUC: D's predictive power within edge-count tertiles")
    print("=" * 90)

    edges = np.array([r["edges"] for r in rows])
    tertiles = np.percentile(edges, [33, 67])

    bins = [
        ("low edges (< {:.0f})".format(tertiles[0]),
         [r for r in rows if r["edges"] < tertiles[0]]),
        ("mid edges ({:.0f}-{:.0f})".format(tertiles[0], tertiles[1]),
         [r for r in rows if tertiles[0] <= r["edges"] < tertiles[1]]),
        ("high edges (>= {:.0f})".format(tertiles[1]),
         [r for r in rows if r["edges"] >= tertiles[1]]),
    ]

    for label, subset in bins:
        if len(subset) < 4:
            continue
        Y = np.array([r["y"] for r in subset])
        D = np.array([r["D"] for r in subset])
        E = np.array([r["edges"] for r in subset])
        n0, n1 = sum(Y == 0), sum(Y == 1)
        if n0 == 0 or n1 == 0:
            print(f"\n  {label}: n={len(subset)} ({n0} mono, {n1} combo) — "
                  f"can't compute AUC (one class missing)")
            continue
        auc_d, p_d = auc_mann_whitney(D[Y == 0], D[Y == 1])
        auc_e, p_e = auc_mann_whitney(E[Y == 0], E[Y == 1])
        print(f"\n  {label}: n={len(subset)} ({n0} mono, {n1} combo)")
        print(f"    edge range: [{E.min():.0f}, {E.max():.0f}]")
        print(f"    D AUC = {auc_d:.3f} (p = {p_d:.4f})")
        print(f"    edge AUC = {auc_e:.3f} (p = {p_e:.4f})")
        signal = "D adds signal" if auc_d > 0.6 and p_d < 0.1 else "D does not add signal"
        print(f"    -> {signal} within this size range")

    # the clinically interesting cases: same size, opposite outcomes
    print(f"\n{'=' * 90}")
    print("CASE STUDIES: size-matched diseases with opposite outcomes")
    print("=" * 90)

    for r in rows:
        if r["y"] != 0:
            continue
        matches = [r2 for r2 in rows if r2["y"] == 1
                   and abs(r2["edges"] - r["edges"]) <= 10]
        if matches:
            m = min(matches, key=lambda r2: abs(r2["edges"] - r["edges"]))
            d_correct = r["D"] < m["D"]
            tag = "D correct" if d_correct else "D WRONG"
            print(f"\n  {r['label']} (mono, D={r['D']:.2f}, {r['edges']} edges)")
            print(f"  vs {m['label']} (combo, D={m['D']:.2f}, {m['edges']} edges)")
            print(f"  edge diff = {abs(m['edges'] - r['edges'])}, D diff = {m['D'] - r['D']:+.2f} [{tag}]")


if __name__ == "__main__":
    main()
