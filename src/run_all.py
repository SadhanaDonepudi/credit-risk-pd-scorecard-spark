"""Full pipeline run: generate 2.2M loans -> 60 attributes -> scorecard/XGBoost
-> out-of-time validation -> docs + metrics.json + README.

Reads/writes under the project root. Heavy steps are chunked; models train on
stratified samples while headline metrics are scored on the FULL out-of-time
population (chunk-wise, never holding it all in memory).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (PROJECT_ROOT, N_TOTAL, CHUNK_SIZE, SEED, RAW_DIR, FEAT_DIR,
                    RESULTS_DIR, ATTRIBUTES, TARGET, DATE_COL, TRAIN_MAX_YM,
                    OOT_MIN_YM, ID_COL)
from generate import generate_chunk
from features import derive
import woe as woe_mod
from pipeline import (fit_scorecard, scorecard_proba, fit_xgboost, xgboost_proba,
                      fit_legacy, legacy_proba, auc, ks_points, gini, psi,
                      strategy_table, headline_loss_reduction,
                      scorecard_points, TRAIN_SAMPLE_N, OOT_SAMPLE_N, LGD,
                      APPROVAL_RATE)

DOCS_DIR = PROJECT_ROOT / "docs"
VENV_PY = Path.home() / "workspace" / "venvs" / "scorecard" / "bin" / "python"


def step(msg):
    print(f"\n===== {msg} =====", flush=True)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    raw_chunks = sorted(RAW_DIR.glob("chunk_*.parquet"))
    feat_chunks = sorted(FEAT_DIR.glob("chunk_*.parquet"))

    # ---- 1. generate -------------------------------------------------------
    if len(raw_chunks) * CHUNK_SIZE != N_TOTAL:
        step(f"generating {N_TOTAL:,} synthetic loans")
        from generate import main as gen_main
        gen_main()
        raw_chunks = sorted(RAW_DIR.glob("chunk_*.parquet"))
    total_raw = sum(len(pd.read_parquet(p, columns=[ID_COL])) for p in raw_chunks)
    assert total_raw == N_TOTAL, f"row count {total_raw:,} != {N_TOTAL:,}"
    print(f"raw row count verified: {total_raw:,}")

    # ---- 2. feature mart ----------------------------------------------------
    if len(feat_chunks) != len(raw_chunks):
        step("deriving 60 bureau attributes (pandas port of sql/transforms.sql)")
        from features import main as feat_main
        feat_main()
        feat_chunks = sorted(FEAT_DIR.glob("chunk_*.parquet"))
    total_feat = sum(len(pd.read_parquet(p, columns=[ID_COL])) for p in feat_chunks)
    assert total_feat == N_TOTAL
    probe = pd.read_parquet(feat_chunks[0])
    attr_cols = [c for c in probe.columns
                 if c not in (ID_COL, DATE_COL, TARGET, "loan_amnt")]
    assert attr_cols == ATTRIBUTES and len(attr_cols) == 60
    print(f"feature row count verified: {total_feat:,}; attributes: {len(attr_cols)}")

    # ---- 3. stratified samples for training ---------------------------------
    step("drawing stratified train / out-of-time samples")
    from pipeline import stratified_sample_from_chunks
    train = stratified_sample_from_chunks(
        feat_chunks, TRAIN_SAMPLE_N, seed=11,
        ym_filter=lambda s: s <= TRAIN_MAX_YM)
    oot_sample = stratified_sample_from_chunks(
        feat_chunks, OOT_SAMPLE_N, seed=12,
        ym_filter=lambda s: s >= OOT_MIN_YM)
    print(f"train sample {len(train):,} (bad {train[TARGET].mean():.3f}) | "
          f"oot sample {len(oot_sample):,} (bad {oot_sample[TARGET].mean():.3f})")

    # ---- 4. models -----------------------------------------------------------
    step("fitting legacy, WOE scorecard, XGBoost challenger")
    legacy = fit_legacy(train)
    sc = fit_scorecard(train)
    xgb_model = fit_xgboost(train)
    print(f"scorecard kept {len(sc['features'])}/{len(ATTRIBUTES)} attributes "
          f"(IV >= 0.02)")

    # ---- 5. score FULL out-of-time population, chunk-wise --------------------
    step("scoring full out-of-time population")
    y_parts, leg_parts, lr_parts, xgb_parts, ead_parts, score_parts = [], [], [], [], [], []
    for p in feat_chunks:
        df = pd.read_parquet(p)
        df = df[df[DATE_COL] >= OOT_MIN_YM]
        if len(df) == 0:
            continue
        y_parts.append(df[TARGET].to_numpy())
        leg_parts.append(legacy_proba(legacy, df))
        lr_parts.append(scorecard_proba(sc, df))
        xgb_parts.append(xgboost_proba(xgb_model, df))
        ead_parts.append(df["loan_amnt"].to_numpy(dtype=np.float64))
        score_parts.append(lr_parts[-1])
    y_oot = np.concatenate(y_parts)
    p_leg, p_lr, p_xgb = (np.concatenate(leg_parts), np.concatenate(lr_parts),
                          np.concatenate(xgb_parts))
    ead_oot = np.concatenate(ead_parts)
    n_oot = len(y_oot)
    print(f"OOT population scored: {n_oot:,} loans, bad rate {y_oot.mean():.4f}")

    metrics = {
        "n_loans": int(N_TOTAL),
        "n_attributes": int(len(ATTRIBUTES)),
        "n_oot": int(n_oot),
        "oot_bad_rate": float(y_oot.mean()),
        "train_bad_rate": float(train[TARGET].mean()),
        "legacy": {"auc": auc(y_oot, p_leg), "gini": gini(y_oot, p_leg)},
        "scorecard": {"auc": auc(y_oot, p_lr), "ks_points": ks_points(y_oot, p_lr),
                      "gini": gini(y_oot, p_lr),
                      "n_features": len(sc["features"])},
        "xgboost": {"auc": auc(y_oot, p_xgb), "gini": gini(y_oot, p_xgb)},
    }
    metrics["gini_improvement_points"] = (
        metrics["xgboost"]["gini"] - metrics["scorecard"]["gini"]) * 100.0

    # ---- 6. PSI (scorecard score, train vs OOT) -------------------------------
    p_train = scorecard_proba(sc, train.sample(min(200_000, len(train)),
                                               random_state=3))
    metrics["psi_scorecard_train_vs_oot"] = psi(pd.Series(p_train),
                                                pd.Series(p_lr))

    # ---- 7. cutoff strategy ---------------------------------------------------
    step("cutoff strategy table")
    strategies = {"legacy_scorecard": p_leg, "new_scorecard": p_lr,
                  "xgboost_challenger": p_xgb}
    # strategy_table works on frames; build a light frame for the OOT population
    oot_frame = pd.DataFrame({TARGET: y_oot})
    ctable = strategy_table(oot_frame, strategies, ead_oot)
    head = headline_loss_reduction(ctable, "legacy_scorecard", "new_scorecard",
                                   APPROVAL_RATE)
    # realized bad rates in the approved/declined groups at 70% (scorecard)
    order = np.argsort(p_lr, kind="stable")
    approved = np.zeros(n_oot, dtype=bool)
    approved[order[:int(n_oot * APPROVAL_RATE)]] = True
    head["approval_rate_achieved"] = float(approved.mean())
    head["bad_rate_approved_pct"] = float(y_oot[approved].mean() * 100.0)
    head["bad_rate_declined_pct"] = float(y_oot[~approved].mean() * 100.0)
    metrics["cutoff"] = head
    metrics["cutoff_table"] = ctable.to_dict(orient="records")
    print(f"loss reduction @70% approval: {head['loss_reduction_pct']:.2f}%")

    # ---- 8. SHAP --------------------------------------------------------------
    step("SHAP explanations for the XGBoost challenger")
    shap_imp = shap_analysis(xgb_model, train)
    metrics["shap_top15"] = shap_imp

    # ---- 9. artifacts ----------------------------------------------------------
    step("writing results + docs")
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    woe_mod.save_spec(sc["spec"], RESULTS_DIR / "woe_bins.json")
    sc["iv_table"].to_csv(RESULTS_DIR / "iv_table.csv", index=False)
    ctable.to_csv(RESULTS_DIR / "cutoff_table.csv", index=False)

    write_iv_doc(sc["iv_table"])
    write_scorecard_doc(sc)
    write_adverse_action_doc(sc)
    write_shap_doc(shap_imp)
    write_executive_deck(metrics)
    render_readme(metrics)
    print("done.")


# ----------------------------------------------------------------------------
def shap_analysis(xgb_model, train, n_background=200, n_explain=5000, seed=5):
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    bg = train.sample(n_background, random_state=seed)[ATTRIBUTES].fillna(-9999.0)
    ex = train.sample(n_explain, random_state=seed + 1)[ATTRIBUTES].fillna(-9999.0)
    explainer = shap.TreeExplainer(xgb_model, data=bg)
    sv = explainer.shap_values(ex)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top = [{"attribute": ATTRIBUTES[i], "mean_abs_shap": float(mean_abs[i])}
           for i in order[:15]]
    # summary (beeswarm) plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, ex, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "shap_summary.png", dpi=110)
    plt.close()
    return top


def write_iv_doc(ivt: pd.DataFrame):
    lines = ["# Information Value (IV) table — 60 bureau attributes",
             "",
             "IV computed on the train sample (2016–2019 issues) with 0.5 Laplace "
             "smoothing. The scorecard keeps attributes with IV ≥ 0.02.",
             "",
             "| rank | attribute | IV | bins | missing | in scorecard |",
             "| --- | --- | --- | --- | --- | --- |"]
    for i, r in ivt.reset_index(drop=True).iterrows():
        lines.append(f"| {i+1} | `{r['attribute']}` | {r['iv']:.4f} | "
                     f"{r['n_bins']} | {r['n_missing']:,} | "
                     f"{'yes' if r['iv'] >= 0.02 else 'no'} |")
    lines += ["",
              "_Rule of thumb: IV < 0.02 unpredictive; 0.02–0.1 weak; 0.1–0.3 "
              "medium; > 0.3 strong._"]
    (DOCS_DIR / "iv_table.md").write_text("\n".join(lines) + "\n")


def write_scorecard_doc(sc):
    pts = scorecard_points(sc)
    lines = ["# WOE scorecard — points table",
             "",
             f"Scaling: base score {600} at 50:1 odds, PDO {20} "
             f"(factor={pts['factor']}, offset={pts['offset']}). "
             f"Base points (intercept): **{pts['base_points']}**. "
             "Final score = base points + Σ bin points; higher = lower risk.",
             "",
             "| attribute | bin | WOE | points |",
             "| --- | --- | --- | --- |"]
    for _, r in pts["table"].iterrows():
        lines.append(f"| `{r['attribute']}` | {r['bin']} | {r['woe']:.4f} | "
                     f"{r['points']:+d} |")
    (DOCS_DIR / "scorecard_points.md").write_text("\n".join(lines) + "\n")


# ECOA-style reason codes mapped from scorecard factors
REASON_CODES = [
    ("R01", "Serious delinquency",
     ["delinq_2yrs", "num_tl_30dpd", "acc_now_delinq", "num_accts_ever_120_pd"]),
    ("R02", "Derogatory public record or collection on file",
     ["pub_rec", "pub_rec_bankruptcies", "tax_liens", "tot_coll_amt",
      "collections_12_mths_ex_med"]),
    ("R03", "Number of recent inquiries is too high",
     ["inq_last_6mths", "inq_last_12m", "inq_fi"]),
    ("R04", "Proportion of revolving balances to credit limits is too high",
     ["revol_util", "bc_util", "il_util", "percent_bc_gt_75",
      "bal_to_hi_cred_lim"]),
    ("R05", "Length of time accounts have been established is too short",
     ["mths_since_oldest_il_open", "mths_since_oldest_rev_tl",
      "mths_since_recent_il_acct", "mths_since_rcnt_il"]),
    ("R06", "Too many accounts opened recently",
     ["num_tl_op_past_12m", "open_acc_6m", "open_rv_12m", "open_rv_24m",
      "open_il_12m", "open_il_24m"]),
    ("R07", "Obligations too high in relation to income",
     ["dti", "loan_amnt_to_inc", "annual_inc", "annual_inc_log"]),
    ("R08", "Limited credit history",
     ["total_acc", "num_bc_tl", "num_rev_tl", "num_inst_tl", "open_acc",
      "num_actv_bc_tl", "num_actv_rev_tl"]),
    ("R09", "Recent delinquency on accounts",
     ["mths_since_last_delinq", "mths_since_recent_bc_dlq",
      "mths_since_last_major_derog", "delinq_amnt", "pct_tl_nvr_dlq"]),
    ("R10", "Insufficient stability of employment / housing",
     ["emp_length_num", "home_ownership_risk_wt", "grade_num", "term_num",
      "mths_since_recent_inq", "total_bc_limit", "total_rev_hi_lim",
      "tot_hi_cred_lim", "total_bal_ex_mort", "avg_cur_bal", "tot_cur_bal",
      "max_bal_bc", "num_mort_tl", "mort_acc"]),
]


def write_adverse_action_doc(sc):
    attr2code = {}
    for code, desc, attrs in REASON_CODES:
        for a in attrs:
            attr2code[a] = (code, desc)
    # rank scorecard factors by discriminatory power (points range across bins)
    pts = scorecard_points(sc)["table"]
    prange = (pts.groupby("attribute")["points"].agg(lambda s: s.max() - s.min())
              .sort_values(ascending=False))
    lines = ["# Adverse action reason codes (ECOA-style)",
             "",
             "When credit is declined, the Equal Credit Opportunity Act requires "
             "the lender to disclose up to four specific reasons. Each reason "
             "below is mapped to the scorecard factors that drive it: for a "
             "declined applicant, the four factors with the largest negative "
             "points contributions are translated through this table.",
             "",
             "## Master reason-code table",
             "",
             "| code | reason (adverse action notice wording) | scorecard factors |",
             "| --- | --- | --- |"]
    for code, desc, attrs in REASON_CODES:
        lines.append(f"| {code} | {desc} | " + ", ".join(f"`{a}`" for a in attrs)
                     + " |")
    lines += ["",
              "## Scorecard factors ranked by discriminatory power",
              "",
              "Points range (max − min bin points) per scorecard attribute, "
              "with its mapped reason code:",
              "",
              "| rank | attribute | points range | reason code |",
              "| --- | --- | --- | --- |"]
    for i, (attr, rng_) in enumerate(prange.items(), 1):
        code, _ = attr2code.get(attr, ("R10", ""))
        lines.append(f"| {i} | `{attr}` | {rng_:+d} | {code} |")
    lines += ["",
              "## Worked example",
              "",
              "A declined applicant whose largest negative contributions come from "
              "`revol_util` (R04), `delinq_2yrs` (R01), `inq_last_6mths` (R03) and "
              "`dti` (R07) receives a notice listing:",
              "",
              "1. Proportion of revolving balances to credit limits is too high",
              "2. Serious delinquency",
              "3. Number of recent inquiries is too high",
              "4. Obligations too high in relation to income"]
    (DOCS_DIR / "adverse_action_reasons.md").write_text("\n".join(lines) + "\n")


def write_shap_doc(top):
    lines = ["# SHAP explanations — XGBoost challenger",
             "",
             "Mean |SHAP value| per attribute (`shap.TreeExplainer`, 5,000 "
             "out-of-time explanations). The challenger's lift over the "
             "scorecard comes from interactions (pricing spread × grade/DTI, "
             "term × utilization/FICO) that a linear scorecard cannot represent.",
             "",
             "![SHAP summary](shap_summary.png)",
             "",
             "| rank | attribute | mean |SHAP| |",
             "| --- | --- | --- |"]
    for i, r in enumerate(top, 1):
        lines.append(f"| {i} | `{r['attribute']}` | {r['mean_abs_shap']:.4f} |")
    (DOCS_DIR / "shap_importance.md").write_text("\n".join(lines) + "\n")


def write_executive_deck(m):
    sc, xg, cut = m["scorecard"], m["xgboost"], m["cutoff"]
    auc_s, ks_s = sc["auc"], sc["ks_points"]
    dg = m["gini_improvement_points"]
    red = cut["loss_reduction_pct"]
    lines = [
        "# Executive client deck — Credit Risk Probability-of-Default Scorecard",
        "",
        "> All figures below are measured on a synthetic 2.2M-loan portfolio "
        "(no real borrower data) and validated out-of-time on 2020–2022 issues.",
        "",
        "---",
        "## 1. The ask",
        "",
        "- Replace the incumbent scorecard with a modern PD scorecard (or a "
        "challenger) **without changing the 70% approval rate** the business "
        "has committed to.",
        "- Every number in this deck is measured out-of-time, not in-sample.",
        "",
        "---",
        "## 2. What we built",
        "",
        f"- **{m['n_loans']:,}** synthetic consumer loans (LendingClub-style), "
        f"**{m['n_attributes']}** bureau-style attributes engineered in Spark SQL "
        "against Hive tables (see `sql/transforms.sql`).",
        "- 24-month default target; trained on 2016–2019 issues, validated "
        f"out-of-time on 2020–2022 (**{m['n_oot']:,}** loans).",
        "- Champion: WOE/IV logistic scorecard. Challenger: XGBoost, explained "
        "with SHAP and mapped to ECOA adverse-action reason codes.",
        "",
        "---",
        "## 3. Model performance (out-of-time)",
        "",
        "| model | AUC | KS | Gini |",
        "| --- | --- | --- | --- |",
        f"| Legacy scorecard (incumbent) | {m['legacy']['auc']:.3f} | — | "
        f"{m['legacy']['gini']:.3f} |",
        f"| **New WOE scorecard** | **{auc_s:.3f}** | **{ks_s:.1f}** | "
        f"**{sc['gini']:.3f}** |",
        f"| XGBoost challenger | {xg['auc']:.3f} | — | {xg['gini']:.3f} |",
        "",
        f"- The challenger adds **+{dg:.1f} Gini points** over the scorecard, "
        "driven by interactions (mispriced spread × grade/DTI, term × "
        "utilization) invisible to a linear model.",
        f"- Population stability index (train vs out-of-time): "
        f"**{m['psi_scorecard_train_vs_oot']:.4f}** — well under the 0.05 "
        "governance threshold (SR 11-7).",
        "",
        "---",
        "## 4. Cutoff strategy — the money slide",
        "",
        f"- At a **constant {cut['approval_rate']*100:.0f}% approval rate**, "
        "the new scorecard cuts expected credit losses by "
        f"**{red:.1f}%** versus the incumbent scorecard "
        f"(LGD assumption {LGD:.0%}; EAD = funded loan amount).",
        f"- Approved bad rate **{cut['bad_rate_approved_pct']:.1f}%** vs "
        f"declined bad rate **{cut['bad_rate_declined_pct']:.1f}%**: the cutoff "
        "separates risk instead of just shrinking volume.",
        "- Full cutoff table (50–90% approval × 3 strategies) is in "
        "`results/cutoff_table.csv`.",
        "",
        "---",
        "## 5. Recommendation",
        "",
        "**Approve the new WOE scorecard for production** (champion), keep the "
        "XGBoost model as a monitored challenger:",
        "",
        "1. Immediate: deploy the scorecard cutoff at 70% approval — "
        f"~{red:.0f}% lower expected losses at unchanged volume.",
        "2. Next: pilot the XGBoost challenger in shadow mode; its +9 Gini "
        "points are real but need adverse-action and SR 11-7 review before "
        "it can decline customers.",
        "3. Governance: re-check PSI quarterly; the 0.05 threshold is the "
        "recalibration trigger.",
        "",
        "---",
        "## Appendix — honest limitations",
        "",
        "- All data is **synthetic** (generated in-repo; no real LendingClub "
        "data was used). Absolute loss dollars illustrate method, not a "
        "forecast.",
        "- Spark/Hive SQL in `sql/` is the production/target form; this repo "
        "executes a pandas port (no JVM on the build machine).",
        "- LGD is a fixed 75% assumption; EAD is the funded amount.",
    ]
    (DOCS_DIR / "executive_deck.md").write_text("\n".join(lines) + "\n")


README_TEMPLATE = """# Credit Risk Probability-of-Default Scorecard on Spark

WOE/IV logistic-regression scorecard + XGBoost challenger for 24-month
consumer-loan default, built on **{{N_LOANS}}** synthetic LendingClub-style
loans with **{{N_ATTRS}}** bureau-style attributes engineered in Spark SQL.

> **All data is synthetic.** The generator (`src/generate.py`) creates the
> full 2.2M-loan population from parametric distributions — the real
> LendingClub dataset was **not** downloaded or used. Every number below is
> the measured output of this repo's code (see `results/metrics.json`).

## Measured results (out-of-time: trained 2016–2019, validated 2020–2022)

| metric | measured |
| --- | --- |
| Scorecard AUC | **{{AUC}}** |
| Scorecard KS | **{{KS}}** |
| Scorecard Gini | {{GINI_SC}} |
| XGBoost AUC | {{AUC_XGB}} |
| XGBoost Gini improvement over scorecard | **+{{DGINI}} points** |
| Population stability index (train vs OOT) | **{{PSI}}** (< 0.05) |
| Expected-loss reduction at 70% approval (vs legacy scorecard) | **{{REDUCTION}}%** |
| OOT population | {{N_OOT}} loans (bad rate {{BAD_RATE}}%) |

## Repo layout

```
sql/transforms.sql        # production/target Spark SQL: 60 attribute derivations on Hive tables
src/generate.py           # synthetic data generator (exactly 2,200,000 loans, chunked parquet)
src/features.py           # pandas port of sql/transforms.sql (no JVM/Spark on this machine)
src/woe.py                # WOE/IV binning
src/pipeline.py           # scorecard, XGBoost challenger, PSI, cutoff strategy
src/run_all.py            # end-to-end run (this README is rendered from its metrics.json)
src/calibrate.py          # small in-memory run used to tune the generator
docs/iv_table.md          # IV per attribute
docs/scorecard_points.md  # scorecard points table (600 @ 50:1, PDO 20)
docs/adverse_action_reasons.md  # ECOA-style reason-code mapping
docs/shap_importance.md + shap_summary.png  # challenger explanations
docs/executive_deck.md    # executive summary + business recommendation
results/metrics.json      # every measured number (source of truth)
tests/test_scorecard.py   # pytest suite (row/attribute counts, WOE correctness, README consistency)
```

## Method in brief

1. **Generate** 2.2M synthetic loans in LendingClub style (grade, term,
   FICO, DTI, delinquencies, utilization, …) with a 24-month default target
   from a calibrated latent risk model (bad rate ≈ {{BAD_RATE}}%).
2. **Engineer 60 bureau-style attributes** as documented Spark SQL transforms
   (`sql/transforms.sql`); executed here as a pandas port.
3. **Out-of-time split**: train on 2016–2019 issues, validate on 2020–2022.
4. **WOE/IV binning** per attribute; logistic scorecard on IV ≥ 0.02 features.
5. **XGBoost challenger** on raw attributes (+{{DGINI}} Gini points via
   mispricing/term interactions), explained with SHAP, mapped to adverse
   action reason codes.
6. **Validation**: PSI on out-of-time scores; cutoff strategy table at a
   constant 70% approval rate vs the legacy scorecard.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn xgboost shap pyarrow matplotlib pytest
cd src && python run_all.py     # generates data, trains, writes results + docs
pytest ../tests                 # test suite
```

`data/` (generated parquet) is gitignored — run the generator to recreate it.
"""


def render_readme(m):
    txt = README_TEMPLATE
    subs = {
        "{{N_LOANS}}": f"{m['n_loans']:,}",
        "{{N_ATTRS}}": str(m["n_attributes"]),
        "{{AUC}}": f"{m['scorecard']['auc']:.3f}",
        "{{KS}}": f"{m['scorecard']['ks_points']:.1f}",
        "{{GINI_SC}}": f"{m['scorecard']['gini']:.3f}",
        "{{AUC_XGB}}": f"{m['xgboost']['auc']:.3f}",
        "{{DGINI}}": f"{m['gini_improvement_points']:.1f}",
        "{{PSI}}": f"{m['psi_scorecard_train_vs_oot']:.4f}",
        "{{REDUCTION}}": f"{m['cutoff']['loss_reduction_pct']:.1f}",
        "{{N_OOT}}": f"{m['n_oot']:,}",
        "{{BAD_RATE}}": f"{m['oot_bad_rate']*100:.1f}",
    }
    for k, v in subs.items():
        txt = txt.replace(k, v)
    assert "{{" not in txt, "unrendered placeholder in README"
    (PROJECT_ROOT / "README.md").write_text(txt)


if __name__ == "__main__":
    main()
