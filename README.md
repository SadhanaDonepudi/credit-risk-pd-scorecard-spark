# Credit Risk Probability-of-Default Scorecard on Spark

WOE/IV logistic-regression scorecard + XGBoost challenger for 24-month
consumer-loan default, built on **2,200,000** synthetic LendingClub-style
loans with **60** bureau-style attributes engineered in Spark SQL.

> **All data is synthetic.** The generator (`src/generate.py`) creates the
> full 2.2M-loan population from parametric distributions — the real
> LendingClub dataset was **not** downloaded or used. Every number below is
> the measured output of this repo's code (see `results/metrics.json`).

## Measured results (out-of-time: trained 2016–2019, validated 2020–2022)

| metric | measured |
| --- | --- |
| Scorecard AUC | **0.747** |
| Scorecard KS | **39.5** |
| Scorecard Gini | 0.494 |
| XGBoost AUC | 0.796 |
| XGBoost Gini improvement over scorecard | **+9.8 points** |
| Population stability index (train vs OOT) | **0.0007** (< 0.05) |
| Expected-loss reduction at 70% approval (vs legacy scorecard) | **18.7%** |
| OOT population | 944,046 loans (bad rate 16.4%) |

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
   from a calibrated latent risk model (bad rate ≈ 16.4%).
2. **Engineer 60 bureau-style attributes** as documented Spark SQL transforms
   (`sql/transforms.sql`); executed here as a pandas port.
3. **Out-of-time split**: train on 2016–2019 issues, validate on 2020–2022.
4. **WOE/IV binning** per attribute; logistic scorecard on IV ≥ 0.02 features.
5. **XGBoost challenger** on raw attributes (+9.8 Gini points via
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
