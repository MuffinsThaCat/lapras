"""Pharmacokinetic interaction layer: CYP / P-gp roles mined from each drug's own FDA label.

The network-based interaction score ignores pharmacokinetics (a drug that inhibits CYP3A4 raises
the exposure of a CYP3A4 substrate, whatever their targets).  This extracts, per drug, which
enzymes it is a SUBSTRATE of and which it INHIBITS or INDUCES, from the cached openFDA
"Drug Interactions" text (data/fda_label_interactions.json, built by validate_ddi --fetch), then flags
pairs where one drug perturbs an enzyme the other depends on.

Heuristics (regex over lower-cased label text, deliberately simple and noisy):
  * sentence mentions the drug's own name + 'inhibit' + enzyme   -> inhibitor of that enzyme
  * ... + 'induc' + enzyme                                       -> inducer
  * ... + 'substrate'/'metabolized' + enzyme                     -> substrate
  * sentence without the drug's own name: 'inhibitors/inducers of CYPx' or 'CYPx inhibitors/inducers'
    -> the label is warning about perpetrators, so the drug is a SUBSTRATE of CYPx
  * 'substrates of CYPx' / 'CYPx substrates' without own name -> the drug perturbs CYPx

CIRCULARITY (disclosed): roles come from the same label sections that supply validate_ddi's ground
truth.  The role features describe a drug alone and never use partner names, but a drug whose label
says 'avoid strong CYP3A4 inhibitors' will also name ketoconazole, so the layer partly rediscovers
the text it is tested on.  Read validate_ddi's PK numbers as an upper bound on independent accuracy.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "fda_label_interactions.json")
ENZ = re.compile(r"cyp\s?(\d[a-z]\d{1,2})|(p-?gp\b|p-glycoprotein|abcb1)")


def _enzymes(s):
    out = set()
    for m in ENZ.finditer(s):
        out.add("cyp" + m.group(1) if m.group(1) else "pgp")
    return out


def extract_roles(name, text):
    """-> dict(substrate=set, inhibitor=set, inducer=set) of enzymes."""
    roles = dict(substrate=set(), inhibitor=set(), inducer=set())
    nm = re.escape(name)
    # the drug's own name must be the grammatical subject of the verb: 'X inhibits ...', 'X is a strong
    # inhibitor of ...', 'X may induce ...'; no 'with', 'and', commas or 'use' in between (that is
    # 'coadministration of X with strong CYP3A4 inhibitors', where X is the victim)
    subj = r"(?<![a-z])" + nm + r"(?![a-z])\s+(?:also\s+|may\s+|can\s+)?(?:is\s+(?:a|an)\s+(?:[\w-]+\s+){0,3}|are\s+(?:[\w-]+\s+){0,3}|)"
    perp_inh = re.compile(subj + r"(?:inhibits|inhibitor)\b")
    perp_ind = re.compile(subj + r"(?:induces|inducer|induce)\b")
    perp_inh2 = re.compile(subj + r"(?:inhibit)\b")
    victim = re.compile(r"\b(?:inhibitors?|inducers?)\b")
    for s in re.split(r"(?<=[.;])\s+|\n", text):
        enz = _enzymes(s)
        if not enz:
            continue
        is_perp_inh = bool(perp_inh.search(s) or perp_inh2.search(s))
        is_perp_ind = bool(perp_ind.search(s))
        if is_perp_inh:
            roles["inhibitor"] |= enz
        if is_perp_ind:
            roles["inducer"] |= enz
        if not (is_perp_inh or is_perp_ind):
            if victim.search(s):
                roles["substrate"] |= enz               # label warns about perpetrators -> this drug is the victim
            elif re.search(r"(?<![a-z])" + nm + r"(?![a-z])[^.;]{0,60}(substrate|metaboli[sz]ed|metabolism)", s) or \
                    re.search(r"(substrate|metaboli[sz]ed)[^.;]{0,60}(?<![a-z])" + nm + r"(?![a-z])", s):
                roles["substrate"] |= enz
    return roles


def load_roles(lib_names=None, path=CACHE):
    if not os.path.exists(path):
        return {}
    labels = json.load(open(path))
    return {n: extract_roles(n, t) for n, t in labels.items()
            if t and (lib_names is None or n in lib_names)}


def pk_pair(roles, a, b):
    """Enzymes through which one drug perturbs the other: list of (enzyme, perpetrator, victim, kind)."""
    ra, rb = roles.get(a), roles.get(b)
    if not ra or not rb:
        return []
    out = []
    for perp, vic, rp, rv in ((a, b, ra, rb), (b, a, rb, ra)):
        for kind in ("inhibitor", "inducer"):
            for e in rp[kind] & rv["substrate"]:
                out.append((e, perp, vic, kind))
    return out
