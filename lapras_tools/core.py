"""Shared Lapras engine for the companion tools.

Wraps coupling.py (TC, bill, Pisot compiler, halving depth) behind one quiet
function, so every tool gets: bill ranking, hierarchy of modules, and D from
any symmetric non-negative coupling matrix.  Nothing here prints or writes.
"""
import os, sys
from dataclasses import dataclass, field
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from coupling import (  # noqa: E402
    tc_gaussian, per_component_bill, cov_from_coupling, regularize,
    compile_gap, node_residuals, halving_depth,
)

LINE = 3.03
DATA_DIR = os.path.join(ROOT, "data")


@dataclass
class Analysis:
    names: list
    C: np.ndarray
    bill: np.ndarray
    D: float
    tc: float
    levels: list                  # levels[k] = partition (list of index lists), finest first
    R: np.ndarray                 # global residual per level
    meta: dict = field(default_factory=dict)

    @property
    def verdict(self):
        return "GLOBAL" if self.D > LINE else "MODULAR"

    def ranked(self):
        return np.argsort(-self.bill, kind="stable")   # ties resolve by node order, deterministically

    def level_with_at_most(self, k):
        """Finest level having <= k groups (falls back to the coarsest level)."""
        for part in self.levels:
            if len(part) <= k:
                return part
        return self.levels[-1]

    def label_vector(self, part):
        lab = np.full(len(self.names), -1, int)
        for j, grp in enumerate(part):
            for i in grp:
                lab[i] = j
        return lab


def analyze_network(C, names, q=0.9):
    C = np.array(C, dtype=float)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0.0)
    n = C.shape[0]
    if n < 3 or not (C > 0).any():
        raise ValueError(f"need >=3 nodes and at least one positive coupling (got n={n}, "
                         f"{int((C > 0).sum() // 2)} edges)")
    S, _ = regularize(cov_from_coupling(C, q=q))
    tc = tc_gaussian(S)
    bill = per_component_bill(S)
    res = compile_gap(C)
    levels = [g for _, g in res if len(g) > 1] or [[[i] for i in range(n)]]
    _, R = node_residuals(C, levels)
    D, _ = halving_depth(R)
    return Analysis(list(names), C, bill, float(D), float(tc), levels, R)


# ---------------------------------------------------------------- stats

def _ranks(a):
    """Average ranks (ties share the mean rank)."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    """Tie-aware Spearman correlation; nan if either input is constant."""
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- disease I/O

def load_disease(key, score=0.9):
    """Real STRING network for one of the 31 Lapras diseases."""
    from diseases import DISEASES
    from batch import load_string_network
    if key not in DISEASES:
        raise SystemExit(f"unknown disease '{key}'. keys: {', '.join(DISEASES)}")
    d = DISEASES[key]
    C, names, targets = load_string_network(
        os.path.join(DATA_DIR, f"{key}.tsv"), d["known_targets"], min_score=score)
    return C, names, targets, d["label"]


def disease_keys():
    from diseases import DISEASES
    return list(DISEASES.keys())
