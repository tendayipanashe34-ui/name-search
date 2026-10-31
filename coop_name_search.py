#!/usr/bin/env python3
"""
Cooperative name search + duplicate detection (CRISP-DM, pair-classification approach)

Follows the capstone deck design:
  Query -> Normalize -> Retrieve (exact, then TF-IDF char n-grams) -> Classify (match probability) -> Review

Models compared: Logistic Regression, Random Forest, Gradient Boosting.
Metrics: accuracy, precision, recall, F1, ROC-AUC, plus top-k retrieval vs exact-match baseline.

DATA
  The Ushirika CSVs contain aggregate statistics only (no cooperative names). They are used to
  shape a synthetic register: size and region mix (cooperatives_total_jun2025), SACCOS share
  (saccos_registered_dec2023), and crop vocabulary (master file, entity_type == 'crop').
  When you receive the approved, anonymised member/cooperative extract, pass it with
  --register path.csv (columns: name, region [optional]) and everything else runs unchanged.
  Duplicate labels are always simulated from realistic noise, so reported scores are a
  pipeline check, not evidence of real-world accuracy. Re-evaluate on reviewed real pairs.

USAGE
  python coop_name_search.py train  --master MASTER.csv --regional REGIONAL.csv [--register real.csv]
  python coop_name_search.py search "Umoja Kahawa Amcos" --region Kagera [--top 5]
  python coop_name_search.py check  "Tafala Women Cooperative Society"   # duplicate screen for a proposed name

Similarity is evidence, not proof of identity. Nothing here merges, deletes or approves records.
"""
import argparse
import random
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path(__file__).resolve().parent / "coop_search_outputs"
GENERIC = {"ltd", "limited", "cooperative", "society", "saccos", "amcos", "union", "co", "operative"}
BAND_LIKELY, BAND_REVIEW = 0.90, 0.50

WORDS = """umoja amani upendo tumaini baraka neema ushindi maendeleo jitegemee mshikamano
uhuru mwangaza imani furaha heshima subira nguvu kilimo mavuno tafara bidii juhudi
mapinduzi kazi tegemeo faraja ujamaa harambee mwanga pamoja chemchem msingi nuru
wakulima wafugaji wavuvi wanawake vijana mkombozi kilele mlima bonde ziwa mto jua
kijiji kijani zawadi hazina tunu azimio sifa ridhaa shukrani amka songa mbele""".split()
OTHER_KINDS = ["Dairy", "Fishing", "Women", "Youth", "Livestock", "Housing", "Transport",
               "Irrigation", "Mining", "Consumer"]
DEFAULT_CROPS = ["Coffee", "Cotton", "Cashew", "Tea", "Tobacco", "Sisal", "Maize", "Sesame"]


# --------------------------------------------------------------------------- normalisation
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", s)          # s.a.c.c.o.s -> saccos
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = []
    for t in s.split():
        t = {"limited": "ltd", "sacco": "saccos", "coop": "cooperative", "co": "cooperative",
             "soc": "society", "operative": "cooperative"}.get(t, t)
        toks.append(t)
    return " ".join(toks)


def core_tokens(norm: str):
    return [t for t in norm.split() if t not in GENERIC]


def coop_type(norm: str) -> str:
    return "amcos" if "amcos" in norm.split() else "saccos" if "saccos" in norm.split() else "other"


def phon_token(t: str) -> str:
    """Crude phonetic key for East/Southern African naming: l/r, b/v, c/k, ph/f, e/i, o/u merge."""
    for a, b in [("ph", "f"), ("sh", "s"), ("ch", "c"), ("ck", "k"), ("th", "t"), ("ee", "i"), ("oo", "u")]:
        t = t.replace(a, b)
    t = t.translate(str.maketrans({"c": "k", "v": "b", "l": "r", "y": "i", "w": "u", "z": "s",
                                   "q": "k", "e": "i", "o": "u"}))
    return re.sub(r"(.)\1+", r"\1", t)


def phon_key(norm: str) -> str:
    return " ".join(phon_token(t) for t in norm.split())


# --------------------------------------------------------------------------- string metrics
def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lev_ratio(a, b):
    m = max(len(a), len(b))
    return 1.0 if m == 0 else 1 - lev(a, b) / m


def jaro_winkler(s, t, p=0.1):
    if s == t:
        return 1.0
    ls, lt = len(s), len(t)
    if not ls or not lt:
        return 0.0
    md = max(max(ls, lt) // 2 - 1, 0)
    sm, tm, m = [False] * ls, [False] * lt, 0
    for i, c in enumerate(s):
        for j in range(max(0, i - md), min(i + md + 1, lt)):
            if not tm[j] and t[j] == c:
                sm[i] = tm[j] = True
                m += 1
                break
    if m == 0:
        return 0.0
    k = tr = 0
    for i in range(ls):
        if sm[i]:
            while not tm[k]:
                k += 1
            tr += s[i] != t[k]
            k += 1
    j = (m / ls + m / lt + (m - tr // 2) / m) / 3
    pre = 0
    for a, b in zip(s[:4], t[:4]):
        if a != b:
            break
        pre += 1
    return j + pre * p * (1 - j)


def ngrams(s, n):
    s = f" {s} "
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def dice(a, b):
    return 2 * len(a & b) / (len(a) + len(b)) if (a or b) else 1.0


def soft_core_match(ca, cb):
    """Mean best Jaro-Winkler match of each core token to the other side (symmetric)."""
    if not ca or not cb:
        return 0.0
    f = lambda x, y: np.mean([max(jaro_winkler(t, u) for u in y) for t in x])
    return (f(ca, cb) + f(cb, ca)) / 2


# --------------------------------------------------------------------------- pair features
FEATURES = ["lev_ratio", "jaro_winkler", "token_jaccard", "token_sort_ratio", "overlap_coef",
            "core_jaccard", "core_soft_match", "char3_jaccard", "char2_dice", "tfidf_cosine",
            "phonetic_equal", "phonetic_lev", "len_diff", "first_token_match", "initials_match",
            "region_agree", "type_agree", "exact_norm"]


def pair_features(a_raw, b_raw, ra, rb, vec) -> pd.DataFrame:
    na, nb = [normalize(x) for x in a_raw], [normalize(x) for x in b_raw]
    va, vb = vec.transform(na), vec.transform(nb)
    cos = np.asarray(va.multiply(vb).sum(axis=1)).ravel()
    rows = []
    for i, (x, y) in enumerate(zip(na, nb)):
        tx, ty = x.split(), y.split()
        sx, sy = set(tx), set(ty)
        cx, cy = core_tokens(x), core_tokens(y)
        px, py = phon_key(x), phon_key(y)
        r1, r2 = normalize(ra[i]), normalize(rb[i])
        rows.append([
            lev_ratio(x, y), jaro_winkler(x, y), jaccard(sx, sy),
            lev_ratio(" ".join(sorted(tx)), " ".join(sorted(ty))),
            len(sx & sy) / max(min(len(sx), len(sy)), 1),
            jaccard(set(cx), set(cy)), soft_core_match(cx, cy),
            jaccard(ngrams(x, 3), ngrams(y, 3)), dice(ngrams(x, 2), ngrams(y, 2)), cos[i],
            float(px == py), lev_ratio(px, py), abs(len(x) - len(y)) / max(len(x), len(y), 1),
            float(bool(tx and ty and tx[0] == ty[0])),
            float("".join(t[0] for t in tx) == "".join(t[0] for t in ty)),
            -1.0 if not r1 or not r2 else float(r1 == r2),
            float(coop_type(x) == coop_type(y)), float(x == y)])
    return pd.DataFrame(rows, columns=FEATURES)


def evidence(row: pd.Series) -> str:
    ev = []
    if row.exact_norm: ev.append("identical after normalisation")
    if row.phonetic_equal and not row.exact_norm: ev.append("phonetically identical")
    if row.core_soft_match >= 0.92: ev.append(f"distinctive words agree ({row.core_soft_match:.0%})")
    elif row.core_soft_match < 0.75: ev.append("distinctive words differ")
    if row.char3_jaccard >= 0.7: ev.append(f"character n-grams overlap ({row.char3_jaccard:.0%})")
    if row.token_sort_ratio > row.lev_ratio + 0.15: ev.append("same words, different order")
    if row.region_agree == 1: ev.append("same region")
    elif row.region_agree == 0: ev.append("different region")
    if not row.type_agree: ev.append("different society type")
    return "; ".join(ev) or "weak similarity only"


# --------------------------------------------------------------------------- synthetic register
def load_inputs(master, regional):
    reg = pd.read_csv(regional)
    reg = reg.dropna(subset=["cooperatives_total_jun2025"]).copy()
    reg["sacco_share"] = (reg["saccos_registered_dec2023"] / reg["cooperatives_total_jun2025"]).clip(0.03, 0.6)
    crops = DEFAULT_CROPS
    try:
        m = pd.read_csv(master)
        found = m.loc[m.entity_type == "crop", "entity"].dropna().astype(str)
        found = [c for c in found.unique() if c.lower() not in {"soya beans", "sugar cane", "cotton sulphur", "total", "others"}]
        crops = [c.replace(" Nuts", "") for c in found] or DEFAULT_CROPS
    except Exception as e:  # master is optional
        print(f"[warn] master file not used ({e}); default crops")
    return reg, sorted(set(crops))


def make_register(reg, crops, scale, rng):
    names, rows = set(), []
    for _, r in reg.iterrows():
        n = max(int(round(r.cooperatives_total_jun2025 * scale)), 3)
        for _ in range(n):
            for _try in range(50):
                w = rng.choice(WORDS).capitalize()
                if rng.random() < 0.35:
                    w += " " + rng.choice(WORDS).capitalize()
                u = rng.random()
                if u < r.sacco_share:
                    ctype, nm = "saccos", f"{w} SACCOS Ltd"
                elif u < r.sacco_share + (1 - r.sacco_share) * 0.6:
                    ctype, nm = "amcos", f"{w} {rng.choice(crops)} AMCOS Ltd"
                else:
                    ctype, nm = "other", f"{w} {rng.choice(OTHER_KINDS)} Cooperative Society Ltd"
                if normalize(nm) not in names:
                    names.add(normalize(nm))
                    rows.append((nm, r.region, ctype))
                    break
    df = pd.DataFrame(rows, columns=["name", "region", "ctype"])
    df.insert(0, "coop_id", [f"C{i:05d}" for i in range(len(df))])
    return df


# --------------------------------------------------------------------------- noise model
SUBS = [("l", "r"), ("r", "l"), ("b", "v"), ("v", "b"), ("c", "k"), ("k", "c"), ("ph", "f"),
        ("f", "ph"), ("s", "z"), ("i", "e"), ("e", "i"), ("o", "u"), ("u", "o"), ("w", "u")]
ABBR = {"ltd": ["Limited", "Ltd.", "LTD"], "saccos": ["Sacco", "S.A.C.C.O.S", "Saccos"],
        "amcos": ["A.M.C.O.S", "Amcos", "AMCOS."], "cooperative": ["Coop", "Co-operative"],
        "society": ["Soc."]}


def add_noise(name, rng, force=False):
    toks = name.split()
    ops = rng.sample(["typo", "phon", "abbr", "space", "drop", "order", "case"], k=rng.choice([1, 1, 2, 3]))
    if not force and rng.random() < 0.08:
        return name                                   # unchanged repeat submission
    for op in ops:
        idx = [i for i, t in enumerate(toks) if t.lower() not in GENERIC and len(t) >= 3] or list(range(len(toks)))
        i = rng.choice(idx)
        t = toks[i]
        if op == "typo" and len(t) >= 3:
            p, k = rng.randrange(len(t)), rng.choice(["sub", "del", "ins", "swap"])
            ch = rng.choice("abcdefghijklmnopqrstuvwxyz")
            t = {"sub": t[:p] + ch + t[p + 1:], "del": t[:p] + t[p + 1:], "ins": t[:p] + ch + t[p:],
                 "swap": t[:p] + t[p + 1:p + 2] + t[p:p + 1] + t[p + 2:]}[k]
            toks[i] = t
        elif op == "phon":
            cand = [(a, b) for a, b in SUBS if a in t.lower()]
            if cand:
                a, b = rng.choice(cand)
                toks[i] = re.sub(a, b, t, count=1, flags=re.I)
        elif op == "abbr":
            j = [k for k, x in enumerate(toks) if normalize(x) in ABBR or x.lower().strip(".") in ABBR]
            if j:
                k = rng.choice(j)
                key = normalize(toks[k]) if normalize(toks[k]) in ABBR else toks[k].lower().strip(".")
                toks[k] = rng.choice(ABBR[key])
        elif op == "space" and len(toks) > 1:
            k = rng.randrange(len(toks) - 1)
            toks[k:k + 2] = [toks[k] + toks[k + 1]]
        elif op == "drop" and len(toks) > 2:
            gen = [k for k, x in enumerate(toks) if normalize(x) in GENERIC]
            toks.pop(rng.choice(gen) if gen and rng.random() < 0.7 else rng.randrange(len(toks)))
        elif op == "order" and len(toks) > 2:
            k = rng.randrange(len(toks) - 1)
            toks[k], toks[k + 1] = toks[k + 1], toks[k]
        elif op == "case":
            toks = [x.upper() if rng.random() < .5 else x.lower() for x in toks]
    return " ".join(toks)


# --------------------------------------------------------------------------- pair construction
def build_pairs(ents, vec, rng, n_pos=2, n_hard=3, n_same=3, n_rand=1, p_region=0.85):
    ents = ents.reset_index(drop=True)
    norm = [normalize(x) for x in ents.name]
    sim = (vec.transform(norm) @ vec.transform(norm).T).toarray()
    np.fill_diagonal(sim, -1)
    A, B, RA, RB, Y, G = [], [], [], [], [], []

    def add(q, i, j, y):
        A.append(q); B.append(ents.name[j]); Y.append(y); G.append(ents.coop_id[i])
        RA.append(ents.region[i] if rng.random() < p_region else ""); RB.append(ents.region[j])

    for i in range(len(ents)):
        for _ in range(n_pos):
            add(add_noise(ents.name[i], rng), i, i, 1)
        for j in np.argsort(-sim[i])[:n_hard + 2][:n_hard]:       # hard negatives: look-alikes
            add(add_noise(ents.name[i], rng), i, int(j), 0)
        # same-region look-alikes, so region agreement alone cannot separate the classes
        same = [int(j) for j in np.argsort(-sim[i]) if ents.region[j] == ents.region[i]][:n_hard + 2]
        for j in rng.sample(same, min(n_same, len(same))):
            add(add_noise(ents.name[i], rng), i, j, 0)
        for _ in range(n_rand):
            j = rng.randrange(len(ents))
            if j != i:
                add(add_noise(ents.name[i], rng), i, j, 0)
    return A, B, RA, RB, np.array(Y), np.array(G)


def make_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=2.0)),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.08, random_state=SEED),
    }


# --------------------------------------------------------------------------- retrieval + ranking
class NameSearch:
    def __init__(self, register, vec, model):
        self.reg, self.vec, self.model = register.reset_index(drop=True), vec, model
        self.norm = [normalize(x) for x in self.reg.name]
        self.mat = vec.transform(self.norm)

    def search(self, query, region="", top=5, k=30):
        q = vec_q = self.vec.transform([normalize(query)])
        sims = (self.mat @ vec_q.T).toarray().ravel()
        cand = np.argsort(-sims)[:k]
        X = pair_features([query] * len(cand), list(self.reg.name.iloc[cand]),
                          [region] * len(cand), list(self.reg.region.iloc[cand]), self.vec)
        p = self.model.predict_proba(X[FEATURES])[:, 1]
        out = self.reg.iloc[cand].copy()
        out["match_prob"] = p
        out["tfidf_cosine"] = sims[cand]
        out["band"] = np.where(p >= BAND_LIKELY, "LIKELY DUPLICATE - officer to confirm",
                      np.where(p >= BAND_REVIEW, "REVIEW QUEUE", "unlikely"))
        out["evidence"] = [evidence(r) for _, r in X.iterrows()]
        return out.sort_values("match_prob", ascending=False).head(top).reset_index(drop=True)


def retrieval_eval(register, test_ids, vec, models, rng, k=30):
    reg = register.reset_index(drop=True)
    norm = [normalize(x) for x in reg.name]
    mat = vec.transform(norm)
    pos = {c: i for i, c in enumerate(reg.coop_id)}
    queries = []
    for cid in test_ids:
        i = pos[cid]
        queries.append((i, add_noise(reg.name[i], rng, force=True), reg.region[i] if rng.random() < .85 else ""))
    rows = {m: [] for m in list(models) + ["TF-IDF only", "Exact match (normalised)"]}
    for i, q, r in queries:
        sims = (mat @ vec.transform([normalize(q)]).T).toarray().ravel()
        cand = np.argsort(-sims)[:k]
        rank_t = list(cand).index(i) + 1 if i in cand else None
        rows["TF-IDF only"].append(rank_t)
        rows["Exact match (normalised)"].append(1 if normalize(q) == norm[i] else None)
        X = pair_features([q] * len(cand), list(reg.name.iloc[cand]), [r] * len(cand),
                          list(reg.region.iloc[cand]), vec)[FEATURES]
        for m, mod in models.items():
            order = list(cand[np.argsort(-mod.predict_proba(X)[:, 1])])
            rows[m].append(order.index(i) + 1 if i in order else None)
    res = []
    for m, ranks in rows.items():
        r = np.array([x if x else np.inf for x in ranks], dtype=float)
        res.append({"method": m, "top1": np.mean(r <= 1), "top3": np.mean(r <= 3),
                    "top5": np.mean(r <= 5), "MRR": np.mean(np.where(np.isinf(r), 0, 1 / r))})
    return pd.DataFrame(res).round(4)


# --------------------------------------------------------------------------- train / evaluate
def train(args):
    OUT.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    reg_stats, crops = load_inputs(args.master, args.regional)
    if args.register:
        register = pd.read_csv(args.register)
        register["region"] = register.get("region", "").fillna("") if "region" in register else ""
        register["ctype"] = [coop_type(normalize(n)) for n in register.name]
        register.insert(0, "coop_id", [f"C{i:05d}" for i in range(len(register))])
        print(f"Using supplied register: {len(register)} names")
    else:
        register = make_register(reg_stats, crops, args.scale, rng)
        register.to_csv(OUT / "register_synthetic.csv", index=False)
        print(f"Synthetic register from CSV statistics: {len(register)} cooperatives, "
              f"{register.region.nunique()} regions, crops={crops}")

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    vec.fit([normalize(x) for x in register.name])
    # pair building is O(n^2) in memory, so very large registers are subsampled for training only
    fit_reg = register if len(register) <= 4000 else register.sample(4000, random_state=SEED)

    # split by cooperative (not by pair) so no name leaks between train and test
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(fit_reg, groups=fit_reg.coop_id))
    tr_e, te_e = fit_reg.iloc[tr_i], fit_reg.iloc[te_i]
    Xtr_raw, Xte_raw = build_pairs(tr_e, vec, rng), build_pairs(te_e, vec, rng)
    Xtr = pair_features(*Xtr_raw[:4], vec)[FEATURES]; ytr, gtr = Xtr_raw[4], Xtr_raw[5]
    Xte = pair_features(*Xte_raw[:4], vec)[FEATURES]; yte = Xte_raw[4]
    print(f"Pairs: train {len(ytr)} ({ytr.mean():.0%} duplicates), test {len(yte)} ({yte.mean():.0%} duplicates)")

    results, fitted, probs = [], {}, {}
    base_pred = Xte["exact_norm"].values
    results.append({"model": "Exact-match baseline", "cv_f1": np.nan, "accuracy": accuracy_score(yte, base_pred),
                    "precision": precision_score(yte, base_pred, zero_division=0), "recall": recall_score(yte, base_pred),
                    "f1": f1_score(yte, base_pred), "roc_auc": np.nan})
    for name, model in make_models().items():
        cv = []
        for a, b in GroupKFold(5).split(Xtr, ytr, gtr):
            m = make_models()[name].fit(Xtr.iloc[a], ytr[a])
            cv.append(f1_score(ytr[b], m.predict(Xtr.iloc[b])))
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]; pred = (p >= 0.5).astype(int)
        fitted[name], probs[name] = model, p
        results.append({"model": name, "cv_f1": np.mean(cv), "accuracy": accuracy_score(yte, pred),
                        "precision": precision_score(yte, pred), "recall": recall_score(yte, pred),
                        "f1": f1_score(yte, pred), "roc_auc": roc_auc_score(yte, p)})
    res = pd.DataFrame(results).round(4)
    res.to_csv(OUT / "model_metrics.csv", index=False)
    print("\nPair classification (held-out cooperatives):\n", res.to_string(index=False))

    best = res.iloc[1:].sort_values("cv_f1", ascending=False).model.iloc[0]   # chosen on CV, not on test
    print(f"\nSelected model (best CV F1): {best}")

    band = pd.cut(probs[best], [-1, BAND_REVIEW, BAND_LIKELY, 2], labels=["unlikely", "review", "likely"])
    bt = pd.DataFrame({"band": band, "y": yte}).groupby("band", observed=True).y.agg(pairs="size", true_dup_rate="mean").round(3)
    print("\nConfidence bands (test pairs):\n", bt.to_string())

    ret = retrieval_eval(register, te_e.coop_id.tolist(), vec, fitted, rng)
    ret.to_csv(OUT / "retrieval_metrics.csv", index=False)
    print("\nTop-k retrieval for noisy queries of held-out cooperatives:\n", ret.to_string(index=False))

    fi = None
    mdl = fitted[best]
    if hasattr(mdl, "feature_importances_"): fi = mdl.feature_importances_
    elif hasattr(mdl, "named_steps"): fi = np.abs(mdl.named_steps["logisticregression"].coef_[0])
    imp = pd.Series(fi, index=FEATURES).sort_values(ascending=False).round(4)
    imp.to_csv(OUT / "feature_importance.csv", header=["importance"])
    print("\nTop features:\n", imp.head(8).to_string())

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.5, 4.5))
        for n, p in probs.items():
            f, t, _ = roc_curve(yte, p); plt.plot(f, t, label=f"{n} (AUC {roc_auc_score(yte, p):.3f})")
        plt.plot([0, 1], [0, 1], "k:", lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.title("ROC - held-out cooperatives"); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(OUT / "roc_curves.png", dpi=160); plt.close()
    except Exception as e:
        print(f"[warn] ROC plot skipped: {e}")

    # review queue: possible duplicates inside the register itself (human decides; nothing is merged)
    final = NameSearch(register, vec, fitted[best])
    q = []
    for _, r in register.sample(min(300, len(register)), random_state=SEED).iterrows():
        h = final.search(r["name"], r["region"], top=2)
        h = h[h.coop_id != r.coop_id].head(1)
        if len(h) and h.match_prob.iloc[0] >= BAND_REVIEW:
            q.append({"coop_id": r.coop_id, "name": r["name"], "candidate_id": h.coop_id.iloc[0],
                      "candidate_name": h.name.iloc[0], "match_prob": round(h.match_prob.iloc[0], 3),
                      "band": h.band.iloc[0], "evidence": h.evidence.iloc[0], "officer_decision": ""})
    pd.DataFrame(q).to_csv(OUT / "review_queue_sample.csv", index=False)

    joblib.dump({"register": register, "vec": vec, "model": fitted[best], "model_name": best}, OUT / "name_search_model.joblib")
    print(f"\nSaved to {OUT}/ : model_metrics.csv, retrieval_metrics.csv, feature_importance.csv, "
          f"roc_curves.png, review_queue_sample.csv, name_search_model.joblib")


def load_engine():
    b = joblib.load(OUT / "name_search_model.joblib")
    return NameSearch(b["register"], b["vec"], b["model"]), b["model_name"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--master", default="tanzania_ushirika_master.csv")
    t.add_argument("--regional", default="tanzania_ushirika_regional_analysis.csv")
    t.add_argument("--register", help="CSV of real names (columns: name, region)")
    t.add_argument("--scale", type=float, default=0.25, help="fraction of real cooperative counts to simulate")
    for c in ("search", "check"):
        s = sub.add_parser(c)
        s.add_argument("query")
        s.add_argument("--region", default="")
        s.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    else:
        eng, mname = load_engine()
        res = eng.search(a.query, a.region, a.top)
        pd.set_option("display.width", 200, "display.max_colwidth", 70)
        print(f"Model: {mname}   Query: {a.query!r}   Region: {a.region or '(not given)'}\n")
        print(res[["coop_id", "name", "region", "match_prob", "band", "evidence"]].round(3).to_string(index=False))
        if a.cmd == "check":
            top = res.iloc[0]
            verdict = ("Name is too close to an existing society - refer to officer."
                       if top.match_prob >= BAND_REVIEW else "No close existing name found.")
            print(f"\n{verdict} (similarity is evidence, not proof; an officer makes the decision)")


if __name__ == "__main__":
    main()
