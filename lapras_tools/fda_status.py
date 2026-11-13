"""Currently-marketed flag from Drugs@FDA.

ChEMBL's max_phase 4 means 'approved at some point', so withdrawn drugs (albiglutide, ezogabine...)
sit in the library as 'approved'.  This records, for every library drug, whether ANY Drugs@FDA
product containing it has marketing status Prescription / Over-the-counter (i.e. currently marketed
in the US), and the designer ranks marketed drugs first and labels the rest.

    python3 -m lapras_tools.fda_status           # download (~60 calls) + write data/fda_marketed.json

Limits: US only (a drug marketed elsewhere but never FDA-approved, e.g. some Japanese or European
agents, reads as 'not marketed'); drugs absent from Drugs@FDA altogether (older biologics approved
as BLAs are not all in this dataset) read as 'unknown', not 'withdrawn'; name matching strips trailing
salt words and may miss unusual names.
"""
import json, os, urllib.parse, urllib.request
from .druglib import default_library
from .validate_fdc import normalise, API

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "data", "fda_marketed.json")


def build():
    lib = default_library()
    names = set(lib)
    status = {}                                        # lib name -> set of statuses seen
    n_products = 0
    for sort in ("application_number:asc", "application_number:desc"):
        for skip in range(0, 25001, 1000):
            q = urllib.parse.urlencode({"limit": 1000, "skip": skip, "sort": sort})
            try:
                d = json.load(urllib.request.urlopen(f"{API}?{q}", timeout=120))
            except Exception as e:
                print(f"  stop {sort} at skip {skip}: {e}")
                break
            for r in d["results"]:
                for p in r.get("products", []):
                    st = p.get("marketing_status") or "unknown"
                    for ing in p.get("active_ingredients", []):
                        nm = normalise(ing["name"], names)
                        if nm:
                            status.setdefault(nm, set()).add(st)
                    n_products += 1
            print(f"  {sort} skip {skip}: {len(status)} library drugs seen", flush=True)
    out = {}
    for n in lib:
        s = status.get(n)
        if not s:
            out[n] = "unknown"
        elif s & {"Prescription", "Over-the-counter"}:
            out[n] = "marketed"
        else:
            out[n] = "not marketed"                    # only Discontinued / None-listed products
    json.dump(out, open(PATH, "w"))
    c = {k: sum(1 for v in out.values() if v == k) for k in ("marketed", "not marketed", "unknown")}
    print(f"saved {PATH}: {c}")


def load():
    return json.load(open(PATH)) if os.path.exists(PATH) else {}


if __name__ == "__main__":
    build()
