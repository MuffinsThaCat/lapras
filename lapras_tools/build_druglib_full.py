"""Bulk ChEMBL drug library: EVERY drug with a curated mechanism and max phase >= 3.

    python3 -m lapras_tools.build_druglib_full     # ~5-10 minutes, writes data/druglib_chembl_full.csv

Unlike build_druglib.py (which re-derives my hand-picked 167 drugs one at a time), this pages
through ChEMBL's mechanism and drug_indication tables in bulk, so the library is not limited to
drugs I thought of.  Same CSV columns as druglib_chembl.csv, plus seed-only targets carried over
for drugs that are in the hand-written seed (e.g. MTOR for everolimus, which ChEMBL records only
as FKBP1A).  Drug names are ChEMBL preferred names, lower-cased for lookup.
"""
import csv, json, os, time, urllib.parse, urllib.request
from .druglib import seed_library

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "druglib_chembl_full.csv")
BASE = "https://www.ebi.ac.uk"
MIN_PHASE = 3


def fetch(url, tries=6):
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            err = e
            time.sleep(3 * (a + 1))
    raise RuntimeError(f"failed: {url}: {err}")


SKIPPED_WINDOWS = []


def pages(endpoint, key, **params):
    """Yield every row of a paged ChEMBL list endpoint.  Offset-based; when a window keeps failing
    (ChEMBL throws sporadic HTTP 500s on particular pages) the window shrinks 200 -> 50 -> 10 rows,
    and a window that still fails is skipped and recorded in SKIPPED_WINDOWS."""
    offset, total = 0, None
    while total is None or offset < total:
        d = None
        for limit in (200, 50, 10):
            url = (f"{BASE}/chembl/api/data/{endpoint}.json?" +
                   urllib.parse.urlencode(dict(params, limit=limit, offset=offset)))
            try:
                d = fetch(url, tries=3)
                break
            except RuntimeError:
                continue
        if d is None:
            if total is None:
                raise RuntimeError(f"first page of {endpoint} failed")
            SKIPPED_WINDOWS.append((endpoint, offset))
            offset += 10                       # step past the bad rows
            continue
        total = d["page_meta"]["total_count"]
        for row in d[key]:
            yield row
        offset += max(len(d[key]), 1) if d[key] else total   # empty window: stop


def batched(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


SKIPPED = []
CK = os.path.join(HERE, "data", "chembl_bulk_checkpoint.json")


def safe_batch(endpoint, key, param, ids):
    """Fetch rows for `ids`; if a batch keeps failing (ChEMBL throws sporadic HTTP 500s), halve it
    until single ids, and skip (and report) any id that still fails."""
    url = (f"{BASE}/chembl/api/data/{endpoint}.json?" +
           urllib.parse.urlencode({param: ",".join(ids), "limit": 100}))
    try:
        return fetch(url, tries=3)[key]
    except RuntimeError:
        if len(ids) == 1:
            SKIPPED.append(ids[0])
            return []
        mid = len(ids) // 2
        return safe_batch(endpoint, key, param, ids[:mid]) + safe_batch(endpoint, key, param, ids[mid:])


def checkpoint_load():
    return json.load(open(CK)) if os.path.exists(CK) else {}


def checkpoint_save(state):
    json.dump(state, open(CK, "w"))


def main():
    print("mechanisms ...", flush=True)
    mech = list(pages("mechanism", "mechanisms"))
    print(f"  {len(mech)} mechanism rows" +
          (f"  WARNING: skipped windows (offsets): {[o for _, o in SKIPPED_WINDOWS]}" if SKIPPED_WINDOWS else ""))

    ck = checkpoint_load()
    # 1) targets first, so molecules/indications are fetched only for drugs that touch a protein
    #    on at least one Lapras disease network (the rest are unusable by the tools; the ChEMBL
    #    API is slow, ~5 s per call)
    tids = sorted({m["target_chembl_id"] for m in mech if m.get("target_chembl_id")})
    print(f"targets: {len(tids)}", flush=True)
    if "genes" in ck:
        genes = ck["genes"]
        print("  (resumed from checkpoint)")
    else:
        genes = {}
        for chunk in batched(tids, 50):
            for t in safe_batch("target", "targets", "target_chembl_id__in", chunk):
                g = []
                for comp in t.get("target_components", []):
                    for s in comp.get("target_component_synonyms", []):
                        if s["syn_type"] == "GENE_SYMBOL" and s["component_synonym"].upper() not in g:
                            g.append(s["component_synonym"].upper())
                genes[t["target_chembl_id"]] = g
        ck["genes"] = genes; checkpoint_save(ck)

    from .interactions import union_network
    from diseases import DISEASES
    net_genes = set(union_network(0.9)[1])
    for d in DISEASES.values():
        net_genes |= {p.upper() for p in d["proteins"]}
    mech_net = [m for m in mech if any(g in net_genes for g in genes.get(m.get("target_chembl_id"), []))]
    parents = sorted({m.get("parent_molecule_chembl_id") or m["molecule_chembl_id"] for m in mech_net})
    print(f"  {len(net_genes)} network genes; {len(mech_net)} mechanism rows touch them -> {len(parents)} parent molecules", flush=True)

    # 2) molecules (name, max phase)
    if "mol" in ck and ck.get("mol_n") == len(parents):
        mol = {k: tuple(v) for k, v in ck["mol"].items()}
        print("  (molecules resumed from checkpoint)")
    else:
        mol = {}
        for n, chunk in enumerate(batched(parents, 50)):
            for m in safe_batch("molecule", "molecules", "molecule_chembl_id__in", chunk):
                mol[m["molecule_chembl_id"]] = (m.get("pref_name"), float(m.get("max_phase") or 0))
            print(f"  molecule batch {n + 1}/{-(-len(parents) // 50)}", flush=True)
        ck["mol"] = mol; ck["mol_n"] = len(parents); checkpoint_save(ck)
    keep = {p for p, (n, ph) in mol.items() if n and ph >= MIN_PHASE}
    print(f"  kept {len(keep)} with a name and max_phase >= {MIN_PHASE}")
    if SKIPPED:
        print(f"  WARNING: {len(SKIPPED)} ids could not be fetched and were skipped: {SKIPPED[:10]}")

    drug = {}
    for m in mech:
        p = m.get("parent_molecule_chembl_id") or m["molecule_chembl_id"]
        if p not in keep:
            continue
        d = drug.setdefault(p, dict(targets=[], actions=[]))
        for g in genes.get(m.get("target_chembl_id"), []):
            if g not in d["targets"]:
                d["targets"].append(g)
        if m.get("action_type"):
            d["actions"].append(m["action_type"].lower())

    print("indications ...", flush=True)
    ind = ck.get("ind", {})
    todo = [p for p in sorted(drug) if p not in ind]
    for n, chunk in enumerate(batched(todo, 15)):
        got = {p: {} for p in chunk}
        try:
            for r in pages("drug_indication", "drug_indications",
                           parent_molecule_chembl_id__in=",".join(chunk), max_phase_for_ind__gte=1):
                p = r.get("parent_molecule_chembl_id") or r["molecule_chembl_id"]
                h = (r.get("mesh_heading") or r.get("efo_term") or "").strip()
                if h and p in got:
                    got[p][h] = max(got[p].get(h, 0), int(float(r.get("max_phase_for_ind") or 0)))
        except RuntimeError as e:
            print(f"  indication batch {n} failed, drugs left without indications: {chunk[:3]}...", flush=True)
            continue
        ind.update(got)
        ck["ind"] = ind
        if n % 5 == 0:
            checkpoint_save(ck)
        print(f"  indication batch {n + 1}/{-(-len(todo) // 15)}", flush=True)
    checkpoint_save(ck)

    seed = {d.name.lower(): d for d in seed_library().values()}
    rows = []
    for p, d in drug.items():
        name = mol[p][0].lower()
        seed_only = []
        if name in seed:
            seed_only = [t for t in seed[name].targets if t not in d["targets"]]
        tgts = d["targets"] + seed_only
        if not tgts:
            continue
        i = "|".join(f"{h}@{ph}" for h, ph in sorted(ind.get(p, {}).items(), key=lambda x: -x[1]))
        rows.append(dict(drug=name, targets="|".join(tgts), action=(d["actions"] or [""])[0],
                         status="approved" if mol[p][1] >= 4 else "investigational",
                         indications=i, chembl_id=p, seed_only="|".join(seed_only)))
    rows.sort(key=lambda r: r["drug"])
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["drug", "targets", "action", "status", "indications", "chembl_id", "seed_only"])
        w.writeheader(); w.writerows(rows)
    ap = sum(r["status"] == "approved" for r in rows)
    print(f"wrote {len(rows)} drugs ({ap} approved) -> {OUT}")
    missing = [n for n in seed if n not in {r['drug'] for r in rows}]
    print(f"seed drugs absent from the bulk library: {len(missing)} {missing[:15]}")


if __name__ == "__main__":
    main()
