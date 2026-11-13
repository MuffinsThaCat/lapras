"""Tool 6 - Pandemic circuit breaker.

Transmission-network nodes (airports, cities, regions) -> bill (who to close),
D (is spread modular - close one hub - or global - distribute the intervention),
and a metapopulation SIR check of whether closing high-bill nodes actually
slows spread more than closing high-degree / high-betweenness ones.

Data: OpenFlights airports.dat / routes.dat (real route network, ~2014, route
COUNTS not passenger volumes).  Node population defaults to route strength; with
--population geonames it is the real population around each airport (GeoNames cities
>= 15,000 within 40 km, each city assigned to its nearest airport node).  Any edge list
works too:  --edges file.csv  (a,b,weight).

    python3 -m coupling_other_fields.pandemic                    # top-300 airports: bill, D, modules
    python3 -m coupling_other_fields.pandemic --experiment       # intervention comparison
    python3 -m coupling_other_fields.pandemic --experiment --population geonames
    python3 -m coupling_other_fields.pandemic --experiment --population geonames --flows gravity   # MODELLED flows (sensitivity)
    python3 -m coupling_other_fields.pandemic --edges my.csv
"""
import argparse, csv, os, collections
import numpy as np
from lapras_tools.core import analyze_network, LINE

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_openflights(top_n=300):
    air = {}
    with open(os.path.join(DATA, "airports.dat"), encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) > 4 and len(r[4]) == 3:
                air[r[4]] = (r[2], r[3])
    w = collections.Counter()
    with open(os.path.join(DATA, "routes.dat"), encoding="utf-8") as f:
        for r in csv.reader(f):
            a, b = r[2], r[4]
            if a != b and a in air and b in air:
                w[tuple(sorted((a, b)))] += 1
    strength = collections.Counter()
    for (a, b), x in w.items():
        strength[a] += x; strength[b] += x
    keep = [a for a, _ in strength.most_common(top_n)]
    idx = {a: i for i, a in enumerate(keep)}
    C = np.zeros((len(keep), len(keep)))
    for (a, b), x in w.items():
        if a in idx and b in idx:
            C[idx[a], idx[b]] = C[idx[b], idx[a]] = x
    return C, keep, {a: air[a] for a in keep}


def catchment_population(names, radius_km=40.0):
    """Real population around each airport: sum of GeoNames city populations (cities >= 15,000
    people, data/cities_pop.csv) within `radius_km` of the airport (OpenFlights lat/lon).  An
    airport with no such city gets 15,000.  Replaces the route-strength proxy used by default.
    Each city is assigned to its nearest airport among the nodes only; GeoNames 'section of
    populated place' rows (districts) are dropped to avoid double counting.  The radius is a
    modelling choice; cities under 15,000 are missing; airports outside the node set (the other
    7,000 airports) do not claim cities, so a city near a small airport may be assigned to a
    farther major one within the radius."""
    coords = {}
    with open(os.path.join(DATA, "airports.dat"), encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) > 7 and len(r[4]) == 3:
                coords[r[4]] = (float(r[6]), float(r[7]))
    cl = np.genfromtxt(os.path.join(DATA, "cities_pop.csv"), delimiter=",", skip_header=1)
    lat, lon, pop = np.radians(cl[:, 0]), np.radians(cl[:, 1]), cl[:, 2]
    # distance from every city to every airport; each city counts for its NEAREST airport only
    # (so Haneda and Narita do not both claim Tokyo), and only within radius_km
    D = np.empty((len(names), len(lat)))
    for k, n in enumerate(names):
        la, lo = np.radians(coords[n][0]), np.radians(coords[n][1])
        a = np.sin((lat - la) / 2) ** 2 + np.cos(la) * np.cos(lat) * np.sin((lon - lo) / 2) ** 2
        D[k] = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    nearest = D.argmin(axis=0)
    within = D[nearest, np.arange(len(lat))] <= radius_km
    out = np.full(len(names), 15000.0)
    sums = np.bincount(nearest[within], weights=pop[within], minlength=len(names))
    out = np.where(sums > 0, sums, out)
    return out


def gravity_flows(C, names, population):
    """MODELLED passenger flows (real volumes are not available): keep the route network's links but
    weight each by sqrt(pop_i * pop_j) / (1 + distance_km / 1000), a standard gravity form.  Used
    only as a sensitivity check of the pandemic conclusions to the flow assumption, not as data."""
    coords = {}
    with open(os.path.join(DATA, "airports.dat"), encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) > 7 and len(r[4]) == 3:
                coords[r[4]] = (np.radians(float(r[6])), np.radians(float(r[7])))
    la = np.array([coords[n][0] for n in names]); lo = np.array([coords[n][1] for n in names])
    a = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    pop = np.asarray(population, float)
    W = np.sqrt(np.outer(pop, pop)) / (1 + dist / 1000.0)
    return np.where(C > 0, W, 0.0)


def load_bts_db1b(path, top_n=300):
    """REAL passenger volumes for the US domestic network: BTS DB1B Market (10% ticket sample).

    path: Origin_and_Destination_Survey_DB1BMarket_YYYY_Q.zip from transtats.bts.gov/PREZIP.
    Link weight = sampled passengers on the undirected airport pair (both directions summed),
    nodes = the top_n airports by total passengers.  Airport names/coordinates come from OpenFlights.
    Returns (C, names, meta) like load_openflights."""
    import zipfile
    import pandas as pd
    with zipfile.ZipFile(path) as z:
        member = [m for m in z.namelist() if m.lower().endswith(".csv")][0]
        with z.open(member) as f:
            df = pd.read_csv(f, usecols=["Origin", "Dest", "Passengers"], low_memory=False)
    df = df.dropna()
    lo = np.where(df["Origin"] < df["Dest"], df["Origin"], df["Dest"])
    hi = np.where(df["Origin"] < df["Dest"], df["Dest"], df["Origin"])
    g = pd.DataFrame({"a": lo, "b": hi, "p": df["Passengers"].astype(float)}).groupby(["a", "b"])["p"].sum()
    g = g[g.index.get_level_values(0) != g.index.get_level_values(1)]
    tot = pd.concat([g.groupby(level=0).sum(), g.groupby(level=1).sum()]).groupby(level=0).sum()
    air = {}
    with open(os.path.join(DATA, "airports.dat"), encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) > 4 and len(r[4]) == 3:
                air[r[4]] = (r[2], r[3])
    keep = [n for n in tot.sort_values(ascending=False).index if n in air][:top_n]
    idx = {n: i for i, n in enumerate(keep)}
    C = np.zeros((len(keep), len(keep)))
    for (u, v), p in g.items():
        if u in idx and v in idx:
            C[idx[u], idx[v]] = C[idx[v], idx[u]] = p
    return C, keep, {n: air[n] for n in keep}


def load_edges(path):
    names, rows = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            a, b, x = r["a"], r["b"], float(r.get("weight", 1) or 1)
            rows.append((a, b, x))
            for n in (a, b):
                if n not in names:
                    names.append(n)
    idx = {n: i for i, n in enumerate(names)}
    C = np.zeros((len(names), len(names)))
    for a, b, x in rows:
        if a != b:
            C[idx[a], idx[b]] += x; C[idx[b], idx[a]] += x
    return C, names, {n: ("", "") for n in names}


# ---------------------------------------------------------------- epidemic model

def simulate(C, seed_nodes, beta=0.3, gamma=0.1, mix=0.02, days=400, pop=None, mask=None,
             norm=None, reroute=1.0):
    """Deterministic metapopulation SIR.  Node i's force of infection is
    beta * (I_i + mix * sum_j P_ij I_j), P = the coupling normalised by node strength.
    Node sizes are proportional to `pop` (default: coupling strength of C).
    mask: boolean vector; the attack share is measured over these nodes only.
    norm / reroute (what happens to travel when links are closed): `norm` is each node's ORIGINAL
    strength.  reroute = 0: travel to closed airports is CANCELLED (rows of P sum to less than 1, so
    total mixing falls); reroute = 1: it is REDISTRIBUTED over the surviving neighbours (rows
    re-normalised to 1, total mixing unchanged: the pre-nineteenth-pass behaviour, kept as a
    sensitivity and as the default when `norm` is not given).
    Returns (attack share over the masked population, day by which 50% of the
    whole population has been infected or inf)."""
    n = C.shape[0]
    s = C.sum(1)
    # populations come from `pop` (the ORIGINAL network) so closing a node does not shrink it
    base = np.asarray(pop, float) if pop is not None else s
    N = np.where(base > 0, base, 1.0); N = N / N.sum()
    P_redist = C / np.maximum(s[:, None], 1e-12)
    if norm is None:
        P = P_redist
    else:
        P = (1 - reroute) * (C / np.maximum(np.asarray(norm, float)[:, None], 1e-12)) + reroute * P_redist
    S = N.copy(); I = np.zeros(n); R = np.zeros(n)
    for k in seed_nodes:
        I[k] = 1e-4 * N[k]; S[k] -= I[k]
    t50 = np.inf
    for day in range(days):
        lam = beta * (I / N + mix * (P @ (I / N)))
        new = np.minimum(S, S * lam)
        rec = gamma * I
        S -= new; I += new - rec; R += rec
        if t50 == np.inf and (N.sum() - S.sum()) >= 0.5:
            t50 = day
    m = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    return float(1 - S[m].sum() / N[m].sum()), t50


def close_nodes(C, closed):
    C2 = C.copy()
    C2[list(closed), :] = 0; C2[:, list(closed)] = 0
    return C2


def betweenness(C):
    import networkx as nx
    G = nx.Graph()
    n = C.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] > 0:
                G.add_edge(i, j, weight=1.0 / C[i, j])
    bc = nx.betweenness_centrality(G, weight="weight")
    return np.array([bc[i] for i in range(n)])


def bill_round_robin(an, max_modules=8):
    """Lapras-idiomatic distribution: best-bill node of each module in turn,
    then second-best of each, ... (the same one-per-module rule as the Combination Designer)."""
    part = an.level_with_at_most(max_modules)
    queues = [sorted(g, key=lambda i: -an.bill[i]) for g in part]
    queues.sort(key=lambda q: -sum(an.bill[i] for i in q))
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return np.array(out)


def rankings(C, an, population=None):
    r = {"bill": an.ranked(), "bill/module": bill_round_robin(an),
         "strength": np.argsort(-C.sum(1)),
         "degree": np.argsort(-(C > 0).sum(1)), "betweenness": np.argsort(-betweenness(C))}
    if population is not None:
        r["population"] = np.argsort(-np.asarray(population))        # close the biggest metros
    return r


def experiment(C, names, ks=(5, 10, 20), n_seeds=60, seed=0, population=None, reroute=0.0, **sim_kw):
    """Paired design: every strategy is scored on the SAME outbreak seed nodes, drawn
    from nodes that no strategy closes.  Reports mean attack share and the paired
    standard error of (strategy - bill) so differences can be judged against noise.
    population: per-node population (e.g. catchment_population); default is route strength."""
    an = analyze_network(C, names)
    ranks = rankings(C, an, population)
    rng = np.random.default_rng(seed)
    n = len(names)
    never = set(range(n)) - {int(i) for o in ranks.values() for i in o[:max(ks)]}
    allpool = np.array(sorted(never))
    # reserve a random group that only the "random" strategy may close; it is excluded
    # from the measured set and from the seeds, so no strategy closes what is measured
    held = rng.choice(allpool, min(max(ks) * 3, len(allpool) // 3), replace=False)
    pool = np.array(sorted(set(allpool.tolist()) - set(held.tolist())))
    seeds = [int(s) for s in rng.choice(pool, min(n_seeds, len(pool)), replace=False)]
    # Fixed populations from the ORIGINAL network, and attack share measured over the
    # reference set of nodes that NO strategy closes: closing a big hub must not
    # "win" just by removing its own population from (or protecting it in) the tally.
    mask = np.zeros(n, bool); mask[pool] = True
    sim_kw = dict(sim_kw, pop=(C.sum(1) if population is None else np.asarray(population, float)), mask=mask,
                  norm=C.sum(1), reroute=reroute)       # travel to a closed node is cancelled (reroute=0) or redistributed (1)
    base = np.array([simulate(C, [s], **sim_kw)[0] for s in seeds])
    print(f"{len(seeds)} outbreak seeds and {len(pool)} reference nodes shared by all strategies; "
          f"no intervention: attack share {base.mean():.3f}")
    print("shown: mean attack share (lower = better); [paired diff vs bill +/- SE] (negative = beats bill)\n")
    methods = list(ranks) + ["random"]
    print(f"  {'k':>3}  " + "  ".join(f"{m:>22}" for m in methods))
    for k in ks:
        res = {}
        for m, order in ranks.items():
            C2 = close_nodes(C, [int(i) for i in order[:k]])
            res[m] = np.array([simulate(C2, [s], **sim_kw)[0] for s in seeds])
        rnd = []
        for _ in range(8):
            C2 = close_nodes(C, [int(i) for i in rng.choice(held, k, replace=False)])
            rnd.append([simulate(C2, [s], **sim_kw)[0] for s in seeds])
        res["random"] = np.mean(rnd, axis=0)
        cells = []
        for m in methods:
            d = res[m] - res["bill"]
            se = d.std(ddof=1) / np.sqrt(len(d))
            cells.append(f"{res[m].mean():.3f}" + ("" if m == "bill" else f" [{d.mean():+.3f}±{se:.3f}]"))
        print(f"  {k:>3}  " + "  ".join(f"{c:>22}" for c in cells))


def report(C, names, meta, top=15):
    an = analyze_network(C, names)
    print(f"{len(names)} nodes, {int((C > 0).sum() // 2)} links, D = {an.D:.2f} -> {an.verdict}: "
          + ("spread is globally entangled; closing one hub will not contain it, distribute the intervention."
             if an.D > LINE else "spread is modular; contain the module."))
    lab = an.label_vector(an.level_with_at_most(8))
    print(f"  {'Rank':>4} {'Node':6} {'Bill':>7} {'Module':>6}  Place")
    for r, i in enumerate(an.ranked()[:top]):
        c, co = meta[names[i]]
        print(f"  {r+1:>4} {names[i]:6} {an.bill[i]:7.3f} {lab[i]:>6}  {', '.join(x for x in (c, co) if x)}")
    return an


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--edges")
    ap.add_argument("--bts", metavar="DB1B.zip",
                    help="REAL US domestic passenger volumes: BTS DB1BMarket zip (data/db1b_2023_q1.zip)")
    ap.add_argument("--experiment", action="store_true")
    ap.add_argument("--mix", type=float, default=0.002,
                    help="inter-node mixing; 0.02 saturates (everyone infected whatever you close)")
    ap.add_argument("--days", type=int, default=120, help="horizon at which attack share is read")
    ap.add_argument("--reroute", type=float, default=0.0,
                    help="share of travel to a closed airport that is redistributed to surviving neighbours "
                         "(0 = cancelled, default; 1 = the pre-nineteenth-pass behaviour)")
    ap.add_argument("--population", choices=["strength", "geonames"], default="strength",
                    help="node populations: route strength (default) or real GeoNames catchment populations "
                         "(OpenFlights airports only)")
    ap.add_argument("--flows", choices=["routes", "gravity"], default="routes",
                    help="link weights: route counts (default) or MODELLED gravity flows (needs --population geonames)")
    a = ap.parse_args()
    if a.population == "geonames" and a.edges:
        ap.error("--population geonames needs the OpenFlights airports, not --edges")
    if a.flows == "gravity" and a.population != "geonames":
        ap.error("--flows gravity needs --population geonames")
    if a.bts and a.flows == "gravity":
        ap.error("--bts already has real flows; do not combine with --flows gravity")
    if a.bts:
        C, names, meta = load_bts_db1b(a.bts, a.top)
        print(f"BTS DB1B: {len(names)} US airports, {int((C > 0).sum() // 2)} airport pairs, "
              f"{C.sum() / 2:,.0f} sampled passengers (10% ticket sample); link weights = passengers")
    else:
        C, names, meta = load_edges(a.edges) if a.edges else load_openflights(a.top)
    pop =catchment_population(names) if a.population == "geonames" else None
    if a.flows == "gravity":
        C = gravity_flows(C, names, pop)          # bill, D, strength, betweenness and the SIR all use these weights
        print("link weights: MODELLED gravity flows sqrt(pop_i*pop_j)/(1+d/1000 km) on the existing routes")
    report(C, names, meta)
    if pop is not None:
        order = np.argsort(-pop)[:5]
        print(f"catchment populations (GeoNames, 40 km): " + ", ".join(f"{names[i]} {pop[i]/1e6:.1f}M" for i in order))
    if a.experiment:
        print()
        experiment(C, names, mix=a.mix, days=a.days, population=pop, reroute=a.reroute)


if __name__ == "__main__":
    main()
