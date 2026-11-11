"""Coupling analysis: total correlation, hierarchical decomposition, and
target ranking for protein interaction networks.

Self-contained.  Requires only numpy.
"""
import numpy as np


# ============================================================================
# Total correlation
# ============================================================================

def tc_gaussian(S):
    """TC = sum_i 1/2 ln(S_ii) - 1/2 ln det S  (nats)."""
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        return np.inf
    return 0.5 * (np.log(np.diag(S)).sum() - logdet)


def per_component_bill(S):
    """DeltaTC_i = 1/2 (ln Sigma_ii + ln Omega_ii), Omega = Sigma^{-1}.
    Ranks components by coupling contribution."""
    Omega = np.linalg.inv(S)
    return 0.5 * (np.log(np.maximum(np.diag(S), 1e-300))
                  + np.log(np.maximum(np.diag(Omega), 1e-300)))


# ============================================================================
# Coupling → covariance
# ============================================================================

def cov_from_coupling(C, q=0.9):
    """Sigma = (I - c*C)^{-1} with c = q / ||C||_2.  PD for q < 1."""
    n = C.shape[0]
    c = q / max(np.linalg.norm(C, 2), 1e-15)
    return np.linalg.inv(np.eye(n) - c * C)


def regularize(S, eps=1e-6):
    """Shrink S toward identity until min eigenvalue >= eps.
    Returns (S_reg, shrinkage_t)."""
    lam_min = np.linalg.eigvalsh(S)[0]
    if lam_min >= eps:
        return S, 0.0
    t = (eps - lam_min) / (1.0 - lam_min + 1e-15)
    t = max(t, eps)
    return (1 - t) * S + t * np.eye(S.shape[0]), t


# ============================================================================
# Hierarchical decomposition
# ============================================================================

def _phase1(C):
    """Phase 1: merge by mean coupling, module size <= 3."""
    n = C.shape[0]
    S = np.array(C, dtype=float)
    sz = [1] * n
    mods = [[i] for i in range(n)]
    while True:
        k = len(mods)
        s = np.array(sz, dtype=float)
        M = S / np.outer(s, s)
        ok = ((s[:, None] + s[None, :] <= 3)
              & np.triu(np.ones((k, k), dtype=bool), 1)
              & (M > 0))
        if not ok.any():
            break
        vals = np.where(ok, M, -1.0)
        mx = vals.max()
        x, y = divmod(int(np.argmax(vals >= mx - 1e-12)), k)
        mods[x] = mods[x] + mods[y]
        del mods[y]
        S[x, :] += S[y, :]
        S[:, x] += S[:, y]
        S = np.delete(np.delete(S, y, 0), y, 1)
        sz[x] += sz[y]
        del sz[y]
    return mods


def compile_gap(C):
    """Pisot compiler: Phase 1 (mean-coupling merge, size<=3) then
    Phase 2 (greedy pairing by total coupling, rounds until 2 groups).

    Returns list of (fraction, partition) at each level.
    fraction = inter-module coupling / previous inter-module coupling.
    partition = list of groups, each a list of component indices.
    """
    C = np.array(C, dtype=float)
    n = C.shape[0]
    Cl = C.tolist()
    mods = _phase1(C)
    total = float(np.sum(np.triu(C, k=1)))

    def gsum(a, b):
        return sum(Cl[i][j] for i in a for j in b)

    def inter(groups):
        return sum(gsum(groups[a], groups[b])
                   for a in range(len(groups))
                   for b in range(a + 1, len(groups)))

    prev = inter(mods)
    res = [(prev / total if total > 0 else 0.0, [list(m) for m in mods])]
    groups = [list(m) for m in mods]

    while len(groups) > 2 and prev > 0:
        pairs = sorted(
            ((gsum(groups[a], groups[b]), a, b)
             for a in range(len(groups))
             for b in range(a + 1, len(groups))),
            reverse=True,
        )
        used, new = set(), []
        for _, a, b in pairs:
            if a in used or b in used:
                continue
            used |= {a, b}
            new.append(groups[a] + groups[b])
        new += [groups[i] for i in range(len(groups)) if i not in used]
        cur = inter(new)
        res.append((cur / prev, new))
        prev, groups = cur, new

    return res


# ============================================================================
# Residuals and halving depth
# ============================================================================

def node_residuals(C, parts):
    """Per-node residual R_i(k) = fraction of node i's squared coupling
    that crosses block boundaries at level k.

    parts: list of partitions (each a list of groups).
    Returns (Rn, R) where Rn[k,i] is per-node and R[k] is global.
    """
    C2 = C ** 2
    s = C2.sum(1)
    s = np.where(s > 0, s, 1.0)
    n = C.shape[0]
    K = len(parts)
    Rn = np.empty((K, n))
    R = np.empty(K)
    for k, g in enumerate(parts):
        lab = np.empty(n, int)
        for j, grp in enumerate(g):
            for idx in grp:
                lab[idx] = j
        cross = C2 * (lab[:, None] != lab[None, :])
        Rn[k] = cross.sum(1) / s
        denom = C2.sum()
        R[k] = cross.sum() / denom if denom > 0 else 0.0
    return Rn, R


def halving_depth(R):
    """Levels to bring R(k) to 1/4.  Returns (D, exact?).
    D < 3.03 → hierarchically containable.
    D > 3.03 → globally entangled."""
    prev = 1.0
    for k in range(len(R)):
        cur = R[k]
        if cur <= 0.25:
            if cur == prev:
                return float(k), True
            t = ((np.log(0.25) - np.log(prev))
                 / (np.log(max(cur, 1e-300)) - np.log(prev)))
            return k + t, (k < len(R) - 1)
        prev = cur
    return np.inf, False


# ============================================================================
# Edge sensitivity
# ============================================================================

def edge_sensitivity(C, parts, R):
    """dR(k)/dC_ab = 2*C_ab*(I_k - R(k)) / Phi_0  for every edge.

    Returns (sens, edges) where sens is (K, n_edges) and
    edges is list of (a, b) index pairs.
    """
    n = C.shape[0]
    C2 = C ** 2
    Phi0 = np.triu(C2, 1).sum()
    if Phi0 == 0:
        return None, []
    K = len(parts)
    iu = np.triu_indices(n, 1)
    edges = list(zip(iu[0].tolist(), iu[1].tolist()))
    c_vals = C[iu]
    sens = np.zeros((K, len(edges)))
    for ki, g in enumerate(parts):
        lab = np.empty(n, int)
        for j, grp in enumerate(g):
            for idx in grp:
                lab[idx] = j
        cross = lab[iu[0]] != lab[iu[1]]
        sens[ki] = 2 * c_vals * (cross.astype(float) - R[ki]) / Phi0
    return sens, edges


# ============================================================================
# Centrality baselines
# ============================================================================

def degree_centrality(C):
    return (C > 0).sum(1).astype(float)

def strength_centrality(C):
    return np.abs(C).sum(1)

def betweenness_centrality(C):
    """Approximate betweenness from shortest paths on 1/weight graph."""
    try:
        import networkx as nx
        n = C.shape[0]
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                if C[i, j] > 0:
                    G.add_edge(i, j, weight=1.0 / C[i, j])
        bc = nx.betweenness_centrality(G, weight="weight")
        return np.array([bc.get(i, 0.0) for i in range(n)])
    except ImportError:
        return strength_centrality(C)
