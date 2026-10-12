"""Modeling pipeline: scorecard (WOE logistic), XGBoost challenger, SHAP,
out-of-time validation (PSI), and the cutoff strategy table.

All metric functions are pure (frame in -> numbers out) so calibrate.py and
run_all.py share the identical computation.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost as xgb

from config import ATTRIBUTES, TARGET, DATE_COL, TRAIN_MAX_YM, OOT_MIN_YM
import woe as woe_mod

IV_CUTOFF = 0.02          # scorecard keeps attributes with IV >= 0.02
TRAIN_SAMPLE_N = 400_000  # stratified rows used to fit the models
OOT_SAMPLE_N = 400_000    # stratified rows used for out-of-time validation
LGD = 0.75                # loss-given-default assumption for expected-loss math
APPROVAL_RATE = 0.70      # cutoff strategy approval rate


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def auc(y, p):
    return float(roc_auc_score(y, p))


def ks_points(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr) * 100.0)


def gini(y, p):
    return float(2.0 * roc_auc_score(y, p) - 1.0)


def psi(expected: pd.Series, actual: pd.Series, bins=10) -> float:
    """Population Stability Index of `actual` vs `expected` on decile bins."""
    qs = np.linspace(0, 1, bins + 1)
    edges = expected.quantile(qs).unique()
    edges = np.sort(np.asarray(edges, dtype=float))
    e = pd.cut(expected, edges, include_lowest=True).value_counts(normalize=True).sort_index()
    a = pd.cut(actual, edges, include_lowest=True).value_counts(normalize=True).sort_index()
    e = e.values + 1e-6
    a = a.values + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


# ----------------------------------------------------------------------------
# sampling helpers (chunked, stratified, memory-safe)
# ----------------------------------------------------------------------------
def stratified_sample_from_chunks(chunks, n, seed, ym_filter):
    """Draw a stratified (by target) sample of ~n rows from parquet chunks."""
    rng = np.random.default_rng(seed)
    # pass 1: count positives/negatives in the filtered population
    pos = neg = 0
    counts = []
    for p in chunks:
        d = pd.read_parquet(p, columns=[DATE_COL, TARGET])
        d = d[ym_filter(d[DATE_COL])]
        c1 = int(d[TARGET].sum())
        c0 = len(d) - c1
        counts.append((c0, c1))
        neg += c0
        pos += c1
    # pass 2: sample each chunk proportionally within each class
    want_pos = min(pos, int(n * pos / (pos + neg)))
    want_neg = min(neg, n - want_pos)
    parts = []
    for p, (c0, c1) in zip(chunks, counts):
        if c0 + c1 == 0:
            continue
        d = pd.read_parquet(p)
        d = d[ym_filter(d[DATE_COL])]
        for cls, cc, want in ((0, c0, want_neg), (1, c1, want_pos)):
            if cc == 0:
                continue
            take = max(1, int(round(want * cc / (neg if cls == 0 else pos))))
            sub = d[d[TARGET] == cls]
            if len(sub) > take:
                sub = sub.sample(take, random_state=int(rng.integers(1e9)))
            parts.append(sub)
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
def fit_scorecard(train: pd.DataFrame):
    """WOE binning -> IV filter -> logistic regression. Returns dict of artifacts."""
    spec = woe_mod.fit_binning(train, ATTRIBUTES, TARGET)
    ivt = woe_mod.iv_table(spec)
    keep = ivt.loc[ivt["iv"] >= IV_CUTOFF, "attribute"].tolist()
    W_train = woe_mod.transform(train, spec)[keep]
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(W_train, train[TARGET])
    return {"spec": spec, "iv_table": ivt, "features": keep, "model": lr}


def scorecard_proba(artifacts, df: pd.DataFrame) -> np.ndarray:
    W = woe_mod.transform(df, artifacts["spec"])[artifacts["features"]]
    return artifacts["model"].predict_proba(W)[:, 1]


def fit_xgboost(train: pd.DataFrame, seed=7, num_boost_round=220):
    """Challenger on the raw (non-WOE) attributes."""
    X = train[ATTRIBUTES].fillna(-9999.0).astype(np.float32)
    y = train[TARGET].to_numpy()
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "binary:logistic", "eval_metric": "auc",
              "max_depth": 5, "eta": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 50,
              "tree_method": "hist", "seed": seed}
    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    return booster


def xgboost_proba(booster, df: pd.DataFrame) -> np.ndarray:
    X = df[ATTRIBUTES].fillna(-9999.0).astype(np.float32)
    return np.asarray(booster.predict(xgb.DMatrix(X)))


# ----------------------------------------------------------------------------
# legacy scorecard (status-quo baseline for the cutoff strategy table)
# ----------------------------------------------------------------------------
# A "traditional" scorecard: plain logistic regression on six raw bureau
# attributes, no WOE binning, no engineered interaction features. This is the
# incumbent the new WOE scorecard is evaluated against at a constant approval
# rate.
# Incumbent (legacy) scorecard: plain logistic regression on five traditional
# capacity/affordability attributes (income, DTI, employment length, revolving
# utilization, loan-amount-to-income). It has no WOE binning and no bureau
# delinquency, derogatory, or inquiry-history attributes - the information
# whose threshold effects drive the new scorecard's gains.
LEGACY_FEATURES = ["annual_inc_log", "dti", "emp_length_num", "revol_util",
                   "loan_amnt_to_inc"]


def fit_legacy(train: pd.DataFrame):
    X = train[LEGACY_FEATURES].fillna(-9999.0).astype(np.float32)
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X, train[TARGET])
    return lr


def legacy_proba(model, df: pd.DataFrame) -> np.ndarray:
    X = df[LEGACY_FEATURES].fillna(-9999.0).astype(np.float32)
    return model.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------
# cutoff strategy: expected credit-loss comparison at fixed approval rates
# ----------------------------------------------------------------------------
def expected_loss(pd_model: np.ndarray, ead: np.ndarray, lgd=LGD) -> np.ndarray:
    return pd_model * ead * lgd


def strategy_table(df_valid: pd.DataFrame, strategies: dict,
                   ead: np.ndarray, approval_rates=(0.5, 0.6, 0.7, 0.8, 0.9),
                   lgd=LGD) -> pd.DataFrame:
    """Cutoff strategy table: for each candidate model and each approval rate,
    approve the lowest-PD loans and report expected credit loss.

    `strategies` maps a strategy name -> PD vector on the validation frame.
    """
    rows = []
    for name, pd_model in strategies.items():
        order = np.argsort(pd_model, kind="stable")
        for ar in approval_rates:
            k = int(len(df_valid) * ar)
            approved = np.zeros(len(df_valid), dtype=bool)
            approved[order[:k]] = True
            loss = float(expected_loss(pd_model, ead, lgd)[approved].sum())
            rows.append({"strategy": name, "approval_rate": ar,
                         "expected_loss": loss})
    return pd.DataFrame(rows)


def headline_loss_reduction(table: pd.DataFrame, baseline: str, challenger: str,
                            approval_rate=APPROVAL_RATE) -> dict:
    """% expected-loss reduction of `challenger` vs `baseline` at one approval
    rate (both approve the same share of applications)."""
    b = table.loc[(table.strategy == baseline)
                  & (table.approval_rate == approval_rate), "expected_loss"].iloc[0]
    c = table.loc[(table.strategy == challenger)
                  & (table.approval_rate == approval_rate), "expected_loss"].iloc[0]
    return {"baseline": baseline, "challenger": challenger,
            "approval_rate": approval_rate,
            "expected_loss_baseline": float(b),
            "expected_loss_challenger": float(c),
            "loss_reduction_pct": float((1.0 - c / b) * 100.0),
            "lgd": LGD}


# ----------------------------------------------------------------------------
# scorecard points scaling (base 600 @ 50:1 odds, PDO 20)
# ----------------------------------------------------------------------------
def scorecard_points(artifacts, pdo=20, base_score=600, base_odds=50.0):
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    lr = artifacts["model"]
    feats = artifacts["features"]
    base_pts = int(round(offset + factor * lr.intercept_[0]))
    rows = []
    for j, attr in enumerate(feats):
        woe_vals = artifacts["spec"][attr]["woe"]
        coef = lr.coef_[0][j]
        for b, w in enumerate(woe_vals):
            label = "MISSING" if b == len(woe_vals) - 1 else f"bin_{b + 1}"
            rows.append({"attribute": attr, "bin": label,
                         "woe": round(w, 4),
                         "points": int(round(factor * coef * w))})
    return {"factor": round(factor, 4), "offset": round(offset, 2),
            "base_points": base_pts, "table": pd.DataFrame(rows)}
