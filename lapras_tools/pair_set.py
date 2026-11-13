"""Cached approved-combination pair set: label-mined pairs (validate_combos_large.parse) + fixed-dose
products (validate_fdc.pairs_from_cache).

parse() runs a regex over 10,451 labels x ~1,500 drug names (~2 minutes).  Several validation
scripts need the same pair set, so the result is cached in data/pair_set_cache.json and reused
until any input changes: the two raw caches, the drug library files, or the source of the parser
and FDC modules (checked by modification time, so edits to validate_combos_large.parse invalidate it).
"""
import json, os
from .druglib import default_library

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "pair_set_cache.json")
_INPUTS = ["data/fda_combination_indications.json", "data/fda_fdc_products.json", "data/druglib_chembl_full.csv",
           "data/druglib_chembl.csv", "validate_combos_large.py", "validate_fdc.py", "druglib.py"]


def _stamp():
    return {f: os.path.getmtime(os.path.join(HERE, f)) for f in _INPUTS if os.path.exists(os.path.join(HERE, f))}


def get_pairs(lib=None, verbose=True):
    """-> set of sorted (drugA, drugB) tuples."""
    stamp = _stamp()
    if os.path.exists(OUT):
        c = json.load(open(OUT))
        if c.get("stamp") == stamp:
            return {tuple(p) for p in c["pairs"]}
    from .validate_combos_large import parse
    from .validate_fdc import pairs_from_cache
    lib = lib or default_library()
    if verbose:
        print("(rebuilding the combination pair set; ~2 min, cached afterwards)", flush=True)
    pairs = {tuple(sorted(p)) for p in parse(lib)} | {tuple(sorted(p)) for p in pairs_from_cache(lib)}
    json.dump({"stamp": stamp, "pairs": sorted(pairs)}, open(OUT, "w"))
    return pairs
