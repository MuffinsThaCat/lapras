"""Second cancer for the patient-tool finding: lung adenocarcinoma vs squamous cell carcinoma.

In breast cancer (validate_tcga.py) the network-transformed bill predicted ER status no better than
the 47 expression weights it is computed from (AUC 0.949 vs 0.939).  This repeats the test in a
different tumour type, using the Lapras lung-cancer network and the real histology label that comes
with the TCGA project itself (TCGA-LUAD vs TCGA-LUSC):

    python3 -m lapras_tools.validate_tcga_lung --fetch 120     # n tumours per project (~4 MB each, threaded)
    python3 -m lapras_tools.validate_tcga_lung                 # analysis

Reports cross-validated AUC for LUAD-vs-LUSC from (a) the cohort-relative expression weights of the
network genes, (b) the bill values, (c) patient D, and (d) a raw marker (TP63 / NKX2-1 / KRT5
expression) as an upper-bound reference that does not use the network at all.
"""
import argparse, json, os, random, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .core import load_disease, analyze_network
from .patient import reweight, weights_relative
from .validate_tcga import GDC, list_files_for, read_star

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECTS = ("TCGA-LUAD", "TCGA-LUSC")


def npz_path(project):
    return os.path.join(HERE, "data", f"tcga_{project.split('-')[1].lower()}_tpm.npz")


def fetch(n):
    def one(item):
        fid, pid = item
        try:
            return pid, read_star(urllib.request.urlopen(f"{GDC}/data/{fid}", timeout=180).read().decode())
        except Exception as e:
            return pid, None
    for proj in PROJECTS:
        files = list_files_for(proj, n)
        print(f"{proj}: downloading {len(files)} tumours", flush=True)
        tables, ids = [], []
        with ThreadPoolExecutor(4) as ex:
            for k, (pid, t) in enumerate(ex.map(one, files)):
                if t:
                    tables.append(t); ids.append(pid)
                if k % 20 == 0:
                    print(f"  {proj} {k}/{len(files)}", flush=True)
        genes = sorted(set().union(*[t.keys() for t in tables]))
        M = np.array([[t.get(g, 0.0) for g in genes] for t in tables], dtype=np.float32)
        os.makedirs(os.path.dirname(npz_path(proj)), exist_ok=True)
        np.savez_compressed(npz_path(proj), genes=np.array(genes), ids=np.array(ids), tpm=M)
        print(f"saved {M.shape[0]} x {M.shape[1]} -> {npz_path(proj)}", flush=True)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def analyse():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    zs = [np.load(npz_path(p)) for p in PROJECTS]
    genes0 = [str(g) for g in zs[0]["genes"]]
    common = sorted(set(genes0) & set(str(g) for g in zs[1]["genes"]))
    gi = {g: i for i, g in enumerate(common)}
    mats = []
    for z in zs:
        g = {str(x): i for i, x in enumerate(z["genes"])}
        mats.append(z["tpm"][:, [g[x] for x in common]].astype(float))
    tpm = np.vstack(mats)
    y = np.array([0] * len(mats[0]) + [1] * len(mats[1]))      # 0 = LUAD, 1 = LUSC
    C, names, _, _ = load_disease("lung_cancer")
    present = [n for n in names if n in gi]
    print(f"{len(mats[0])} LUAD + {len(mats[1])} LUSC tumours; {len(present)}/{len(names)} lung-network genes found")
    W, B, D = [], [], []
    for r in range(len(tpm)):
        expr = {g: float(tpm[r, gi[g]]) for g in names if g in gi}
        w, _ = weights_relative(expr, names, common, tpm, exclude=r)
        a = analyze_network(reweight(C, w), names)
        W.append(w); B.append(a.bill); D.append(a.D)
    W, B, D = np.array(W), np.array(B), np.array(D)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)

    def cv(X):
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        p = cross_val_predict(LogisticRegression(max_iter=3000, C=0.5), X, y, cv=skf, method="predict_proba")[:, 1]
        return auc(y, p)

    print(f"cross-validated AUC for LUAD vs LUSC:")
    print(f"  {len(names)} expression weights (what the bill is computed FROM): {cv(W):.3f}")
    print(f"  {len(names)} bill values (network-transformed):                   {cv(B):.3f}")
    print(f"  patient D alone:                                              {max(auc(y, D), 1 - auc(y, D)):.3f}")
    for mk in ("TP63", "NKX2-1", "KRT5", "NAPSA"):
        if mk in gi:
            s = tpm[:, gi[mk]]
            print(f"  raw marker {mk:7s} alone (no network):                       {max(auc(y, s), 1 - auc(y, s)):.3f}")
    print(f"  D: LUAD {D[y==0].mean():.2f}, LUSC {D[y==1].mean():.2f}; above the 3.03 line: "
          f"{(D > 3.03).mean()*100:.0f}% of tumours")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", type=int, metavar="N")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.fetch)
    elif all(os.path.exists(npz_path(p)) for p in PROJECTS):
        analyse()
    else:
        ap.error("no cached tables; run --fetch 120 first")


if __name__ == "__main__":
    main()
