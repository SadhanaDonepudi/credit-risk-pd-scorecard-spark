# Executive client deck — Credit Risk Probability-of-Default Scorecard

> All figures below are measured on a synthetic 2.2M-loan portfolio (no real borrower data) and validated out-of-time on 2020–2022 issues.

---
## 1. The ask

- Replace the incumbent scorecard with a modern PD scorecard (or a challenger) **without changing the 70% approval rate** the business has committed to.
- Every number in this deck is measured out-of-time, not in-sample.

---
## 2. What we built

- **2,200,000** synthetic consumer loans (LendingClub-style), **60** bureau-style attributes engineered in Spark SQL against Hive tables (see `sql/transforms.sql`).
- 24-month default target; trained on 2016–2019 issues, validated out-of-time on 2020–2022 (**944,046** loans).
- Champion: WOE/IV logistic scorecard. Challenger: XGBoost, explained with SHAP and mapped to ECOA adverse-action reason codes.

---
## 3. Model performance (out-of-time)

| model | AUC | KS | Gini |
| --- | --- | --- | --- |
| Legacy scorecard (incumbent) | 0.691 | — | 0.381 |
| **New WOE scorecard** | **0.747** | **39.5** | **0.494** |
| XGBoost challenger | 0.796 | — | 0.592 |

- The challenger adds **+9.8 Gini points** over the scorecard, driven by interactions (mispriced spread × grade/DTI, term × utilization) invisible to a linear model.
- Population stability index (train vs out-of-time): **0.0007** — well under the 0.05 governance threshold (SR 11-7).

---
## 4. Cutoff strategy — the money slide

- At a **constant 70% approval rate**, the new scorecard cuts expected credit losses by **18.7%** versus the incumbent scorecard (LGD assumption 75%; EAD = funded loan amount).
- Approved bad rate **8.8%** vs declined bad rate **34.2%**: the cutoff separates risk instead of just shrinking volume.
- Full cutoff table (50–90% approval × 3 strategies) is in `results/cutoff_table.csv`.

---
## 5. Recommendation

**Approve the new WOE scorecard for production** (champion), keep the XGBoost model as a monitored challenger:

1. Immediate: deploy the scorecard cutoff at 70% approval — ~19% lower expected losses at unchanged volume.
2. Next: pilot the XGBoost challenger in shadow mode; its +9 Gini points are real but need adverse-action and SR 11-7 review before it can decline customers.
3. Governance: re-check PSI quarterly; the 0.05 threshold is the recalibration trigger.

---
## Appendix — honest limitations

- All data is **synthetic** (generated in-repo; no real LendingClub data was used). Absolute loss dollars illustrate method, not a forecast.
- Spark/Hive SQL in `sql/` is the production/target form; this repo executes a pandas port (no JVM on the build machine).
- LGD is a fixed 75% assumption; EAD is the funded amount.
