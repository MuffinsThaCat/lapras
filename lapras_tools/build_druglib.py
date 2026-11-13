"""Rebuild the drug library from ChEMBL and audit the hand-written seed against it.

    python3 -m lapras_tools.build_druglib            # fetch (cached), write data/druglib_chembl.csv + audit report

For every drug name in the seed library this asks ChEMBL for: molecule (max_phase),
mechanisms (target gene symbols, action type) and drug indications (MeSH headings +
max phase).  Output columns: drug,targets,action,status,indications,chembl_id.
Indications are '|'-joined "heading@phase" strings used by the Combination Designer's
indication filter.  Network calls are cached in data/chembl_cache.json.
"""
import csv, json, os, sys, time, urllib.parse, urllib.request
from .druglib import seed_library

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "data", "chembl_cache.json")
OUT_PATH = os.path.join(HERE, "data", "druglib_chembl.csv")
BASE = "https://www.ebi.ac.uk/chembl/api/data"

_cache = {}


def get(path, **params):
    key = path + "?" + urllib.parse.urlencode(sorted(params.items()))
    if key in _cache:
        return _cache[key]
    url = f"{BASE}/{path}.json?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                d = json.load(r)
            _cache[key] = d
            return d
        except Exception as e:                       # network hiccup / 5xx
            time.sleep(1.5 * (attempt + 1))
            err = e
    print(f"  ! giving up on {url}: {err}", file=sys.stderr)
    return None


def molecule(name):
    """Resolve a drug name to its PARENT molecule record (mechanisms/indications hang off
    salt forms, so queries are made by parent id)."""
    d = get("molecule/search", q=name, limit=20)
    if not d:
        return None
    cands = [m for m in d["molecules"]
             if (m.get("pref_name") or "").lower().startswith(name.lower())]
    if not cands:
        return None
    best = min(cands, key=lambda m: len(m["pref_name"]))          # 'ERLOTINIB' over 'ERLOTINIB HYDROCHLORIDE'
    parent = (best.get("molecule_hierarchy") or {}).get("parent_chembl_id") or best["molecule_chembl_id"]
    if parent != best["molecule_chembl_id"]:
        pm = get(f"molecule/{parent}")
        if pm:
            best = pm
    best = dict(best)
    best["parent_id"] = parent
    best["max_phase"] = max([float(best.get("max_phase") or 0)] +
                            [float(m.get("max_phase") or 0) for m in cands
                             if (m.get("molecule_hierarchy") or {}).get("parent_chembl_id") == parent])
    return best


def target_genes(tid):
    d = get(f"target/{tid}")
    genes = []
    for comp in (d or {}).get("target_components", []):
        for syn in comp.get("target_component_synonyms", []):
            if syn["syn_type"] == "GENE_SYMBOL":
                genes.append(syn["component_synonym"].upper())
    return genes


def fetch_drug(name):
    m = molecule(name)
    if not m:
        return None
    cid = m["parent_id"]
    mech = (get("mechanism", parent_molecule_chembl_id=cid, limit=50) or {}).get("mechanisms", [])
    targets, actions = [], []
    for x in mech:
        if x.get("target_chembl_id"):
            for g in target_genes(x["target_chembl_id"]):
                if g not in targets:
                    targets.append(g)
        if x.get("action_type"):
            actions.append(x["action_type"].lower())
    ind = (get("drug_indication", parent_molecule_chembl_id=cid, limit=200) or {}).get("drug_indications", [])
    if not ind:
        ind = (get("drug_indication", molecule_chembl_id=cid, limit=200) or {}).get("drug_indications", [])
    inds = {}
    for x in ind:
        h = (x.get("mesh_heading") or x.get("efo_term") or "").strip()
        if h:
            inds[h] = max(inds.get(h, 0), int(float(x.get("max_phase_for_ind") or 0)))
    return dict(chembl_id=cid, phase=int(float(m.get("max_phase") or 0)), targets=targets,
                action=actions[0] if actions else "", indications=inds)


def main():
    global _cache
    if os.path.exists(CACHE_PATH):
        _cache = json.load(open(CACHE_PATH))
    seed = seed_library()
    rows, report = [], []
    for i, (key, d) in enumerate(sorted(seed.items())):
        f = fetch_drug(d.name)
        if i % 20 == 0:
            json.dump(_cache, open(CACHE_PATH, "w"))
        if f is None:
            report.append(f"NOT FOUND in ChEMBL: {d.name}")
            continue
        status = "approved" if f["phase"] == 4 else "investigational"
        seed_t, ch_t = set(d.targets) | set(d.indirect), set(f["targets"])
        if ch_t and not (seed_t & ch_t):
            report.append(f"TARGET MISMATCH {d.name}: seed {sorted(seed_t)} vs ChEMBL {sorted(ch_t)}")
        elif ch_t and not set(d.targets) <= ch_t:
            report.append(f"seed has extra targets {d.name}: {sorted(set(d.targets) - ch_t)} (ChEMBL: {sorted(ch_t)})")
        if not ch_t:
            report.append(f"no ChEMBL mechanism targets for {d.name} (seed kept: {d.targets})")
        if status != d.status:
            report.append(f"STATUS {d.name}: seed {d.status} vs ChEMBL max_phase {f['phase']}")
        ind = "|".join(f"{h}@{p}" for h, p in sorted(f["indications"].items(), key=lambda x: -x[1]))
        seed_only = [t for t in d.targets if t not in f["targets"]]
        rows.append(dict(drug=d.name, targets="|".join(f["targets"] + seed_only),
                         action=f["action"] or d.action, status=status, indications=ind,
                         chembl_id=f["chembl_id"], seed_only="|".join(seed_only)))
        print(f"  {d.name:20s} {f['chembl_id']:14s} phase {f['phase']}  {','.join(f['targets'][:4])}", flush=True)
    json.dump(_cache, open(CACHE_PATH, "w"))
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["drug", "targets", "action", "status", "indications", "chembl_id", "seed_only"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} drugs -> {OUT_PATH}\n\nAUDIT OF THE HAND-WRITTEN SEED ({len(report)} findings):")
    for r in report:
        print("  " + r)


if __name__ == "__main__":
    main()
