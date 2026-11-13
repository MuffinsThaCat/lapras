"""Real-data check of the patient-specific tool on TCGA-BRCA RNA-seq (GDC open access).

    python3 -m lapras_tools.validate_tcga --fetch 80    # download n primary-tumour STAR-count files (~4 MB each)
    python3 -m lapras_tools.validate_tcga               # analyse the cached table

Questions asked (all with a shuffled-expression control, because D is a noisy step function):
  1. Do real patients' D values differ from D under random reweighting (same weights, genes shuffled)?
  2. Does biology show up?  Groups are defined from the data itself - ESR1-high versus
     ESR1-low/PGR-low/ERBB2-low ("triple-negative-like") - since receptor status is not in the
     open expression files.  Compared: D, ESR1's bill rank, and top-10 bill-set similarity
     within vs across groups.
The compact table (log-TPM for every gene, float32) is cached in data/tcga_brca_tpm.npz.
"""
import argparse, json, os, random, urllib.parse, urllib.request
import numpy as np
from .core import load_disease, analyze_network
from .patient import reweight

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "data", "tcga_brca_tpm.npz")
GDC = "https://api.gdc.cancer.gov"


def list_files(n, seed=0):
    return list_files_for("TCGA-BRCA", n, seed)


def list_files_for(project, n, seed=0):
    """n open-access primary-tumour STAR-count files of a TCGA project, one per patient, shuffled."""
    filt = {"op": "and", "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": project}},
        {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
        {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
        {"op": "=", "content": {"field": "cases.samples.sample_type", "value": "Primary Tumor"}},
        {"op": "=", "content": {"field": "access", "value": "open"}}]}
    q = urllib.parse.urlencode({"filters": json.dumps(filt), "fields": "file_id,cases.submitter_id",
                                "size": "2000", "format": "json"})
    hits = json.load(urllib.request.urlopen(f"{GDC}/files?{q}", timeout=60))["data"]["hits"]
    seen, uniq = set(), []
    for h in hits:                                   # one file per patient
        pid = h["cases"][0]["submitter_id"]
        if pid not in seen:
            seen.add(pid); uniq.append((h["file_id"], pid))
    random.Random(seed).shuffle(uniq)
    return uniq[:n]


def read_star(text):
    expr = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or line.startswith("gene_id") or line.startswith("N_"):
            continue
        p = line.split("\t")
        if len(p) < 7:
            continue
        expr[p[1].upper()] = max(expr.get(p[1].upper(), 0.0), float(p[6]))     # tpm_unstranded
    return expr


def fetch_clinical():
    """Cohort-wide TCGA-BRCA clinical table -> {patient: {er, pr, her2, vital, days_to_death, days_followup}}."""
    filt = {"op": "and", "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
        {"op": "=", "content": {"field": "file_name", "value": "nationwidechildrens.org_clinical_patient_brca.txt"}}]}
    q = urllib.parse.urlencode({"filters": json.dumps(filt), "fields": "file_id", "size": "1", "format": "json"})
    fid = json.load(urllib.request.urlopen(f"{GDC}/files?{q}", timeout=60))["data"]["hits"][0]["file_id"]
    text = urllib.request.urlopen(f"{GDC}/data/{fid}", timeout=120).read().decode()
    lines = [l.split("\t") for l in text.splitlines()]
    head = lines[0]
    col = {h: i for i, h in enumerate(head)}
    out = {}
    for r in lines[3:]:                                # two descriptor rows follow the header
        if len(r) < len(head):
            continue
        g = lambda k: r[col[k]] if k in col else ""
        out[r[col["bcr_patient_barcode"]]] = dict(
            er=g("er_status_by_ihc"), pr=g("pr_status_by_ihc"), her2=g("her2_status_by_ihc"),
            vital=g("vital_status"), d_death=g("death_days_to"), d_follow=g("last_contact_days_to"))
    return out


def fetch(n):
    files = list_files(n)
    tables, ids = [], []
    old_genes = None
    if os.path.exists(NPZ):                            # incremental: keep what is already cached
        z = np.load(NPZ)
        old_genes, old_ids, old_tpm = list(z["genes"]), list(z["ids"]), z["tpm"]
        for k, pid in enumerate(old_ids):
            tables.append(dict(zip(old_genes, old_tpm[k].tolist()))); ids.append(pid)
        files = [(f, p) for f, p in files if p not in set(old_ids)]
        print(f"  {len(old_ids)} cached, {len(files)} to download")
    for k, (fid, pid) in enumerate(files):
        try:
            txt = urllib.request.urlopen(f"{GDC}/data/{fid}", timeout=120).read().decode()
        except Exception as e:
            print(f"  skip {pid}: {e}"); continue
        tables.append(read_star(txt)); ids.append(pid)
        print(f"  {k+1}/{len(files)} {pid} ({len(tables[-1])} genes)", flush=True)
    genes = sorted(set().union(*[t.keys() for t in tables]))
    M = np.array([[t.get(g, 0.0) for g in genes] for t in tables], dtype=np.float32)
    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    np.savez_compressed(NPZ, genes=np.array(genes), ids=np.array(ids), tpm=M)
    print(f"saved {M.shape[0]} patients x {M.shape[1]} genes -> {NPZ}")


def weights(tpm_row, genes_idx, names, floor=0.05):
    """Percentile of each network gene within this patient's whole transcriptome."""
    vals = np.sort(tpm_row)
    w = np.array([np.searchsorted(vals, tpm_row[genes_idx[n]], side="right") / len(vals) if n in genes_idx else np.nan
                  for n in names])
    w[np.isnan(w)] = np.nanmedian(w)
    return np.clip(w, floor, 1.0)


def jaccard(a, b):
    return len(a & b) / len(a | b)


def analyse(scheme="transcriptome", n_shuffle=1):
    """scheme: 'transcriptome' (percentile within the patient's own transcriptome) or
    'cohort' (percentile among the other patients, leave-one-out)."""
    from scipy import stats
    from .patient import weights_relative
    z = np.load(NPZ)
    genes, ids, tpm = list(z["genes"]), list(z["ids"]), z["tpm"].astype(float)
    gi = {g: i for i, g in enumerate(genes)}
    C, names, _, _ = load_disease("breast_cancer")
    base = analyze_network(C, names)
    present = [n for n in names if n in gi]
    print(f"\n{'='*78}\nWEIGHTING: {scheme}\n{'='*78}")
    print(f"{len(ids)} real TCGA-BRCA primary tumours; {len(present)}/{len(names)} network genes found; generic D = {base.D:.2f}")

    rng = np.random.default_rng(0)
    esr1, pgr, erbb2 = (tpm[:, gi[g]] for g in ("ESR1", "PGR", "ERBB2"))
    hi = esr1 >= np.percentile(esr1, 66)
    tn = (esr1 <= np.percentile(esr1, 33)) & (pgr <= np.percentile(pgr, 50)) & (erbb2 <= np.percentile(erbb2, 66))
    print(f"groups from the data: ESR1-high n={hi.sum()}, triple-negative-like n={tn.sum()}")

    Dp, Dn, tops, esr_rank = [], [], [], []
    for r in range(len(ids)):
        if scheme == "cohort":
            expr_r = {g: float(tpm[r, gi[g]]) for g in names if g in gi}
            w, _ = weights_relative(expr_r, names, genes, tpm, exclude=r)
        else:
            w = weights(tpm[r], gi, names)
        a = analyze_network(reweight(C, w), names)
        Dp.append(a.D)
        for _ in range(n_shuffle):
            Dn.append(analyze_network(reweight(C, rng.permutation(w)), names).D)
        tops.append(set(a.ranked()[:10].tolist()))
        esr_rank.append(int(np.where(a.ranked() == names.index("ESR1"))[0][0]) + 1)
    Dp, Dn = np.array(Dp), np.array(Dn)
    fin = lambda x: x[np.isfinite(x)]
    print(f"\n1) Real patients vs shuffled-expression control (same weight vectors, genes permuted)")
    print(f"   real     D: mean {fin(Dp).mean():.2f}  sd {fin(Dp).std():.2f}  range [{fin(Dp).min():.2f}, {fin(Dp).max():.2f}]  (inf: {np.isinf(Dp).sum()})")
    print(f"   shuffled D: mean {fin(Dn).mean():.2f}  sd {fin(Dn).std():.2f}  range [{fin(Dn).min():.2f}, {fin(Dn).max():.2f}]  (inf: {np.isinf(Dn).sum()})")
    ks = stats.ks_2samp(fin(Dp), fin(Dn))
    print(f"   KS test real vs shuffled: stat {ks.statistic:.2f}, p = {ks.pvalue:.3f}")
    print(f"   patients above the 3.03 line: {(Dp > 3.03).mean()*100:.0f}% real, {(Dn > 3.03).mean()*100:.0f}% shuffled")

    print(f"\n2) Does biology show up?")
    mw = stats.mannwhitneyu(fin(Dp[hi]), fin(Dp[tn]))
    print(f"   D, ESR1-high: {fin(Dp[hi]).mean():.2f}   triple-negative-like: {fin(Dp[tn]).mean():.2f}   Mann-Whitney p = {mw.pvalue:.3f}")
    er = np.array(esr_rank)
    print(f"   ESR1 bill rank (1 = top): ESR1-high median {np.median(er[hi]):.0f}, triple-negative-like median {np.median(er[tn]):.0f}  "
          f"(generic rank {list(base.ranked()).index(names.index('ESR1'))+1})")
    mw2 = stats.mannwhitneyu(er[hi], er[tn])
    print(f"   Mann-Whitney on ESR1 rank p = {mw2.pvalue:.2g}")

    def mean_j(ix, jx):
        return np.mean([jaccard(tops[i], tops[j]) for i in ix for j in jx if i != j])
    ih, it = list(np.where(hi)[0]), list(np.where(tn)[0])
    print(f"   top-10 bill-set Jaccard: within ESR1-high {mean_j(ih, ih):.2f}, within TN-like {mean_j(it, it):.2f}, "
          f"across groups {mean_j(ih, it):.2f}  (generic top-10 vs patients: "
          f"{np.mean([jaccard(set(base.ranked()[:10].tolist()), t) for t in tops]):.2f})")


def clinical_cached():
    path = os.path.join(HERE, "data", "tcga_brca_clinical.json")
    if os.path.exists(path):
        return json.load(open(path))
    c = fetch_clinical()
    json.dump(c, open(path, "w"))
    return c


def analyse_labelled():
    """Same tumours, but with REAL receptor status from the TCGA clinical table.

    Questions: (1) does the cohort-relative bill ranking separate ER+ from ER-negative tumours?
    (2) does the bill vector (47 numbers) predict ER status BETTER than the 47 expression weights it
    is computed from?  If not, the network adds nothing beyond expression for this purpose.
    (3) does patient D relate to receptor status or survival?"""
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from .patient import weights_relative
    z = np.load(NPZ)
    genes, ids, tpm = list(z["genes"]), [str(i) for i in z["ids"]], z["tpm"].astype(float)
    gi = {g: i for i, g in enumerate(genes)}
    clin = clinical_cached()
    C, names, _, _ = load_disease("breast_cancer")
    esr_i = names.index("ESR1")

    rows = []
    for r, pid in enumerate(ids):
        c = clin.get(pid)
        if not c or c["er"] not in ("Positive", "Negative"):
            continue
        expr_r = {g: float(tpm[r, gi[g]]) for g in names if g in gi}
        w, _ = weights_relative(expr_r, names, genes, tpm, exclude=r)
        a = analyze_network(reweight(C, w), names)
        rank = np.empty(len(names)); rank[a.ranked()] = np.arange(len(names))
        tn = c["er"] == "Negative" and c["pr"] == "Negative" and c["her2"] == "Negative"
        rows.append(dict(pid=pid, er=int(c["er"] == "Positive"), tnbc=tn, w=w, bill=a.bill.copy(), D=a.D,
                         esr_rank=rank[esr_i], vital=c["vital"], r=r))
    er = np.array([x["er"] for x in rows])
    D = np.array([x["D"] for x in rows])
    print(f"\n{'='*78}\nREAL RECEPTOR STATUS (TCGA clinical table), cohort-relative weights\n{'='*78}")
    print(f"{len(rows)} tumours with ER status: {er.sum()} ER+, {(1-er).sum()} ER-negative "
          f"(of which {sum(x['tnbc'] for x in rows)} triple-negative by IHC)")
    esr_rank = np.array([x["esr_rank"] for x in rows])
    print(f"1) ESR1 bill rank (1=top): ER+ median {np.median(esr_rank[er==1]):.0f}, ER-negative median {np.median(esr_rank[er==0]):.0f}; "
          f"AUC of ESR1 rank for ER status = {roc_auc_score(er, -esr_rank):.2f}")
    raw = np.array([tpm[x["r"], gi["ESR1"]] for x in rows])
    print(f"   (upper-bound reference: raw ESR1 TPM alone, AUC = {roc_auc_score(er, raw):.2f})")
    print(f"2) Cross-validated (5-fold x1) logistic regression predicting ER status:")
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    def cv_auc(X, label):
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), X, label, cv=skf, method="predict_proba")[:, 1]
        return roc_auc_score(label, p)
    Wm = np.array([x["w"] for x in rows]); Bm = np.array([x["bill"] for x in rows])
    print(f"   47 expression weights (what the bill is computed FROM): AUC {cv_auc(Wm, er):.3f}")
    print(f"   47 bill values (network-transformed):                   AUC {cv_auc(Bm, er):.3f}")
    print(f"   patient D alone:                                        AUC {roc_auc_score(er, D):.3f} (ER+ tumours have higher D)")
    tnbc = np.array([x["tnbc"] for x in rows]); mask = (er == 1) | tnbc
    if tnbc.sum() >= 5:
        lab2 = er[mask]
        print(f"   ER+ vs triple-negative only (n={mask.sum()}, {tnbc.sum()} TNBC): weights {cv_auc(Wm[mask], lab2):.3f}, "
              f"bill {cv_auc(Bm[mask], lab2):.3f}, D {roc_auc_score(lab2, D[mask]):.3f}")
    mw = stats.mannwhitneyu(D[er == 1], D[er == 0])
    print(f"3) D: ER+ {D[er==1].mean():.2f}, ER-negative {D[er==0].mean():.2f} (Mann-Whitney p = {mw.pvalue:.2g})")
    dead = np.array([x["vital"] == "Dead" for x in rows])
    if 5 <= dead.sum() < len(dead):
        mw2 = stats.mannwhitneyu(D[dead], D[~dead])
        print(f"   survival: {dead.sum()} deaths of {len(rows)} (short follow-up; underpowered): D dead {D[dead].mean():.2f} vs alive "
              f"{D[~dead].mean():.2f}, p = {mw2.pvalue:.2f}")


def analyse_cutoff(cutoffs=(0.0, 0.05, 0.10, 0.25, 0.40, 0.50)):
    """Sensitivity of the patient tool's expression-eligibility cutoff (--min-expression).

    For each cutoff and each tumour with known ER status: does the proposed combination include an
    ESR1-targeting drug?  (Should be common in ER+ and rare in ER-negative tumours.)  Also: how many
    modules end up with no eligible drug, and how similar is the plan to the one at the default 0.25."""
    from .patient import weights_relative
    from .druglib import default_library, DISEASE_KEYWORDS
    from .combo_designer import design
    z = np.load(NPZ)
    genes, ids, tpm = list(z["genes"]), [str(i) for i in z["ids"]], z["tpm"].astype(float)
    gi = {g: i for i, g in enumerate(genes)}
    clin = clinical_cached()
    C, names, _, _ = load_disease("breast_cancer")
    lib = default_library()
    kw = DISEASE_KEYWORDS["breast_cancer"]
    esr_drugs = {d.name for d in lib.values() if "ESR1" in d.targets}
    pats = []
    for r, pid in enumerate(ids):
        c = clin.get(pid)
        if not c or c["er"] not in ("Positive", "Negative"):
            continue
        expr_r = {g: float(tpm[r, gi[g]]) for g in names if g in gi}
        w, _ = weights_relative(expr_r, names, genes, tpm, exclude=r)
        pats.append((c["er"] == "Positive", w, analyze_network(reweight(C, w), names)))
    print(f"\n{'='*78}\nEXPRESSION-ELIGIBILITY CUTOFF SENSITIVITY ({len(pats)} tumours with ER status)\n{'='*78}")
    plans = {}
    for cut in cutoffs:
        pl = []
        for er, w, a in pats:
            plan = design(a, lib, 5, keywords=kw, eligible=(w >= cut))
            pl.append((er, {p["drug"].name for p in plan["picks"]}, len(plan["gaps"])))
        plans[cut] = pl
    ref = plans[0.25]
    print(f"{'cutoff':>7} {'ER+ with ESR1 drug':>20} {'ER- with ESR1 drug':>20} {'mean empty modules':>19} {'plan Jaccard vs 0.25':>21}")
    for cut, pl in plans.items():
        pos = [bool(s & esr_drugs) for er, s, g in pl if er]
        neg = [bool(s & esr_drugs) for er, s, g in pl if not er]
        jac = np.mean([len(s & r[1]) / max(len(s | r[1]), 1) for (_, s, _), r in zip(pl, ref)])
        print(f"{cut:7.2f} {np.mean(pos)*100:19.0f}% {np.mean(neg)*100:19.0f}% {np.mean([g for _, _, g in pl]):19.2f} {jac:21.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", action="store_true", help="only run the --min-expression sensitivity analysis")
    ap.add_argument("--fetch", type=int, metavar="N")
    a = ap.parse_args()
    if a.cutoff:
        analyse_cutoff()
        return
    if a.fetch:
        fetch(a.fetch)
    elif os.path.exists(NPZ):
        analyse("transcriptome")
        analyse("cohort")
        analyse_labelled()
    else:
        ap.error(f"no cached table; run with --fetch 80 first ({NPZ})")


if __name__ == "__main__":
    main()
