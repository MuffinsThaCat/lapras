"""Download STRING interaction data for all diseases in diseases.py.

Saves one TSV per disease into data/<disease_key>.tsv in the same format
as string_aging_full.tsv.  Respects STRING API rate limits (1 req/sec).
"""
import os, sys, time, json, urllib.request, urllib.parse

from diseases import DISEASES

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

STRING_API = "https://string-db.org/api"
SPECIES = 9606  # Homo sapiens
MIN_SCORE = 400  # medium confidence


def resolve_identifiers(proteins):
    """Map gene symbols to STRING identifiers, return mapping."""
    params = urllib.parse.urlencode({
        "identifiers": "\r".join(proteins),
        "species": SPECIES,
        "limit": 1,
        "caller_identity": "drug_discovery_pipeline",
    })
    url = f"{STRING_API}/json/get_string_ids?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "DrugDiscoveryPipeline/1.0"})
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())
    mapping = {}
    for entry in data:
        query = entry.get("queryItem", "")
        preferred = entry.get("preferredName", "")
        string_id = entry.get("stringId", "")
        if query and preferred and string_id:
            mapping[query] = {"preferredName": preferred, "stringId": string_id}
    return mapping


def download_network(proteins, outpath):
    """Download interaction network for a set of proteins from STRING."""
    params = urllib.parse.urlencode({
        "identifiers": "\r".join(proteins),
        "species": SPECIES,
        "required_score": MIN_SCORE,
        "network_type": "functional",
        "caller_identity": "drug_discovery_pipeline",
    })
    url = f"{STRING_API}/tsv/network?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "DrugDiscoveryPipeline/1.0"})
    resp = urllib.request.urlopen(req, timeout=120)
    data = resp.read().decode("utf-8")
    with open(outpath, "w") as f:
        f.write(data)
    lines = data.strip().split("\n")
    return len(lines) - 1  # subtract header


def main():
    skip_existing = "--force" not in sys.argv
    diseases = sys.argv[1:] if len(sys.argv) > 1 and sys.argv[1] != "--force" else list(DISEASES.keys())
    diseases = [d for d in diseases if d in DISEASES]

    total = len(diseases)
    for i, key in enumerate(diseases):
        d = DISEASES[key]
        outpath = os.path.join(DATA_DIR, f"{key}.tsv")

        if skip_existing and os.path.exists(outpath) and os.path.getsize(outpath) > 100:
            print(f"  [{i+1}/{total}] {d['label']:40s} — already exists, skipping")
            continue

        print(f"  [{i+1}/{total}] {d['label']:40s} — downloading...", end="", flush=True)
        try:
            n_edges = download_network(d["proteins"], outpath)
            print(f" {n_edges} interactions")
        except Exception as e:
            print(f" ERROR: {e}")

        time.sleep(1.1)  # rate limit

    print(f"\nDone. TSV files in {DATA_DIR}/")


if __name__ == "__main__":
    main()
