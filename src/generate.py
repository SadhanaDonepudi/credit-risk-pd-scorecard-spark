"""Synthetic LendingClub-style loan data generator.

Generates EXACTLY 2,200,000 synthetic consumer loans in chunks and writes them
as parquet (efficient dtypes; never holds the full population in memory).

ALL DATA IS SYNTHETIC. It is generated from parametric distributions whose
shapes mimic public LendingClub loan data (it is NOT the real LendingClub
dataset and nothing was downloaded). The 24-month default target is drawn
from a calibrated latent risk model so that a bureau-style scorecard reaches
a realistic ~0.74 AUC.

Reproducible: chunk c is drawn with seed SEED + c.
"""
import numpy as np
import pandas as pd

from config import N_TOTAL, CHUNK_SIZE, SEED, RAW_DIR

# ----------------------------------------------------------------------------
# generator hyper-parameters (tuned so the scorecard lands near AUC ~0.74
# and the XGBoost challenger gains ~+9 Gini points via interaction effects)
# ----------------------------------------------------------------------------
BASE_RATE = 0.12        # target population 24-month default rate
INTERCEPT = -5.40       # calibrated for BASE_RATE at the chosen signal scale
LIN_SCALE = 0.30        # scales the linear (main-effect) part of log-odds
INTER_W = 3.00          # scales the interaction part (what XGBoost picks up)
NOISE_SD = 1.90         # irreducible noise in log-odds

GRADES = ["A", "B", "C", "D", "E", "F", "G"]
GRADE_BASE_RATE = {"A": 7.25, "B": 10.75, "C": 14.25, "D": 18.00,
                   "E": 21.50, "F": 24.75, "G": 28.00}
GRADE_CUTS = [-1.05, -0.45, 0.05, 0.55, 1.05, 1.50]  # on grade_score ~ N(0, 1.28)

EMP_LENGTHS = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
               "6 years", "7 years", "8 years", "9 years", "10+ years"]
HOME_OWN = ["RENT", "OWN", "MORTGAGE", "OTHER"]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
            "small_business", "car", "medical", "vacation", "wedding",
            "house", "renewable_energy", "moving", "other"]
PURPOSE_P = [0.52, 0.22, 0.07, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]
STATES = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI", "NJ", "VA",
          "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI", "MN", "CO", "SC", "AL",
          "LA", "KY", "OR", "OK", "CT", "UT", "IA", "NV", "AR", "MS", "KS", "NM",
          "NE", "ID", "WV", "HI", "NH", "ME", "RI", "MT", "DE", "SD", "ND", "AK",
          "VT", "WY", "DC"]


# Nonlinear main-effect weights (module-level so calibration can ablate them)
W_FICO, W_FICO_SUB = -0.50, -0.22
W_DTI, W_DTI_Q = 0.30, 0.20
W_RU, W_RU_KINK = 0.27, 0.26
W_DELINQ, W_INQ_SQRT, W_GRADE = 0.26, 0.16, 0.22
W_TERM60, W_INC, W_PUBREC = 0.14, -0.15, 0.18
W_HI_UTIL, W_HI_DTI, W_SUBPRIME = 0.60, 0.55, 0.48
W_MANY_INQ, W_THIN_FILE = 0.34, 0.30
# Interaction weights: spread-bucket x grade/dti (mispricing), term x util/fico.
# All terms use at least one (near-)independent or bounded factor, so they are
# orthogonal to the scorecard's main effects - the XGBoost challenger's edge.
W_INT_SPREAD_GRADE, W_INT_SPREAD_DTI = 1.15, 1.05
W_INT_TERM_RU, W_INT_TERM_FICO = 0.50, -0.45


def _amort_payment(principal, monthly_rate, n_months):
    r = np.asarray(monthly_rate, dtype=np.float64)
    p = np.asarray(principal, dtype=np.float64)
    pay = np.where(r > 1e-9, p * r / (1.0 - (1.0 + r) ** (-n_months)), p / n_months)
    return pay.astype(np.float32)


def generate_chunk(n, seed, id_offset):
    rng = np.random.default_rng(seed)

    # ---- latent borrower creditworthiness -----------------------------------
    F = rng.standard_normal(n).astype(np.float32)

    # ---- issue date: uniform over 2016-01 .. 2022-12 (out-of-time split) ----
    ym_idx = rng.integers(0, 84, n)                      # 0..83
    year = 2016 + ym_idx // 12
    month = 1 + ym_idx % 12
    issue_ym = (year * 100 + month).astype(np.int32)
    issue_d = np.char.add(np.char.add(np.char.add(year.astype(str), "-"),
                        np.char.zfill(month.astype(str), 2)), "-15")

    # ---- grade (ordered by latent quality) ----------------------------------
    gscore = (-F + rng.standard_normal(n) * 0.8).astype(np.float32)
    grade_num = np.digitize(gscore, GRADE_CUTS) + 1      # 1..7
    grade = np.array(GRADES, dtype=object)[grade_num - 1]
    sub_digit = rng.integers(1, 6, n)
    sub_grade = np.char.add(grade.astype(str), sub_digit.astype(str))

    # ---- pricing -------------------------------------------------------------
    base = np.array([GRADE_BASE_RATE[g] for g in GRADES], dtype=np.float32)[grade_num - 1]
    int_rate = np.clip(base + rng.standard_normal(n).astype(np.float32) * 1.1
                       + 0.08 * (year - 2016), 5.0, 31.0).astype(np.float32)
    term = np.where(rng.random(n) < 0.28 + 0.03 * grade_num, "60 months", "36 months")
    term_num = np.where(term == "60 months", 60, 36).astype(np.int8)

    loan_amnt = np.clip(np.exp(rng.standard_normal(n) * 0.6
                              + np.log(14000.0) + 0.10 * grade_num - 0.08 * F),
                        1000, 40000)
    loan_amnt = (np.round(loan_amnt / 25) * 25).astype(np.float32)
    funded_amnt = loan_amnt
    installment = _amort_payment(loan_amnt, int_rate / 1200.0, term_num)

    # ---- borrower ------------------------------------------------------------
    annual_inc = np.clip(np.exp(rng.standard_normal(n) * 0.55
                                + np.log(78000.0) + 0.22 * F + 0.02 * (year - 2016)),
                         8000, 3_000_000).astype(np.float32)
    dti = np.clip(rng.standard_normal(n).astype(np.float32) * 5.5
                  + 13.5 - 2.6 * F + 0.12 * (year - 2016), 0, 45).astype(np.float32)

    emp_cut = np.digitize(-F + rng.standard_normal(n) * 0.9,
                          [-1.6, -1.0, -0.5, -0.1, 0.3, 0.7, 1.0, 1.3, 1.6, 2.0])
    emp_length = np.array(EMP_LENGTHS, dtype=object)[np.clip(emp_cut, 0, 10)]
    emp_length_num = np.where(emp_cut <= 0, 0.5, np.minimum(emp_cut, 10)).astype(np.float32)

    ho_u = rng.random(n)
    home_ownership = np.where(ho_u < 0.44 + 0.06 * F, "MORTGAGE",
                      np.where(ho_u < 0.60 + 0.06 * F, "RENT",
                      np.where(ho_u < 0.95, "OWN", "OTHER")))

    verification_status = np.where(rng.random(n) < 0.32, "Verified",
                          np.where(rng.random(n) < 0.55, "Source Verified",
                                   "Not Verified"))
    purpose = rng.choice(PURPOSES, n, p=PURPOSE_P)
    application_type = np.where(rng.random(n) < 0.96, "Individual", "Joint App")
    addr_state = rng.choice(STATES, n)

    # ---- bureau tradeline picture --------------------------------------------
    fico_low = np.clip(np.round(658 + 52 * F + rng.standard_normal(n) * 17),
                       600, 800).astype(np.int16)
    fico_high = (fico_low + rng.integers(0, 5, n)).astype(np.int16)
    fico_mid = (fico_low.astype(np.float32) + fico_high.astype(np.float32)) / 2

    delinq_2yrs = np.minimum(rng.poisson(np.exp(-1.7 - 0.85 * F)), 10).astype(np.int8)
    inq_last_6mths = np.minimum(rng.poisson(np.exp(-0.95 - 0.30 * F)), 8).astype(np.int8)
    pub_rec = np.minimum(rng.poisson(np.exp(-2.7 - 0.45 * F)), 4).astype(np.int8)
    pub_rec_bankruptcies = rng.binomial(np.maximum(pub_rec, 0).astype(np.int64),
                                        0.65).astype(np.int8)

    open_acc = np.minimum(rng.poisson(9 + 1.2 * F) + 2, 60).astype(np.int8)
    total_acc = np.minimum(open_acc.astype(np.int16)
                           + rng.poisson(14 + 2 * F), 120).astype(np.int16)
    mort_acc = np.where(home_ownership == "MORTGAGE",
                        np.minimum(rng.poisson(1.2, n), 6),
                        np.minimum(rng.poisson(0.15, n), 6)).astype(np.int8)

    total_bc_limit = np.exp(rng.standard_normal(n) * 0.5
                            + np.log(22000.0) + 0.30 * F).astype(np.float32)
    revol_util = np.clip(rng.standard_normal(n).astype(np.float32) * 17
                         + 52 - 13 * F, 0, 120).astype(np.float32)
    revol_bal = (revol_util / 100.0 * total_bc_limit).astype(np.float32)

    hist_mths = np.clip(np.round(np.exp(rng.standard_normal(n) * 0.5
                                        + np.log(150.0) - 0.25 * F)), 6, 600).astype(np.int16)
    # earliest credit line date = issue date minus history length
    ecl_year = year - (hist_mths // 12)
    ecl_month = month - (hist_mths % 12)
    ecl_year = np.where(ecl_month < 1, ecl_year - 1, ecl_year)
    ecl_month = np.where(ecl_month < 1, ecl_month + 12, ecl_month)
    earliest_cr_line = np.char.add(np.char.add(ecl_year.astype(str), "-"),
                             np.char.add(np.char.zfill(ecl_month.astype(str), 2), "-01"))

    # ---- 24-month default target from latent risk model -----------------------
    # Main effects are NONLINEAR in the raw attributes (subprime FICO penalty,
    # convex DTI, utilization kink, diminishing inquiries): a WOE-binned
    # scorecard captures these curves, a raw linear model only their projection.
    z_fico = (fico_mid - 700.0) / 45.0
    z_dti = (dti - 13.5) / 6.0
    z_ru = (revol_util - 50.0) / 20.0
    z_delinq = (delinq_2yrs.astype(np.float32) - 0.35) / 0.8
    z_inq = (inq_last_6mths.astype(np.float32) - 0.5) / 0.8
    z_grade = (grade_num.astype(np.float32) - 3.2) / 1.6
    z_inc = (np.log(annual_inc) - 11.2) / 0.6
    z_pubrec = (pub_rec.astype(np.float32) - 0.08) / 0.35
    fico_sub = np.clip((660.0 - fico_mid) / 45.0, 0, None)   # subprime penalty
    dti_q = z_dti * np.abs(z_dti)                            # convex DTI
    ru_kink = np.clip(z_ru - 1.5, 0, None)                    # utilization kink
    inq_sqrt = (np.sqrt(inq_last_6mths.astype(np.float32) + 0.5) - 1.0) / 0.7
    # bureau threshold effects: step risk not visible to a raw linear model,
    # captured exactly by WOE binning
    hi_util = (revol_util >= 90.0).astype(np.float32)
    hi_dti = (dti >= 28.0).astype(np.float32)
    subprime = (fico_mid < 640.0).astype(np.float32)
    many_inq = (inq_last_6mths >= 4).astype(np.float32)
    thin_file = (total_acc.astype(np.float32) <= 6.0).astype(np.float32)
    z_spread = (int_rate - base) / 1.1      # pure pricing noise, ~N(0,1)
    # Binary spread buckets (top/bottom quartile of pricing noise): bounded, so
    # the interaction cannot create extreme heavy-tailed risk subgroups that
    # would spike the scorecard's KS; the tree challenger still finds them.
    # Binary spread buckets (top/bottom quartile of pricing noise): bounded, so
    # the interaction cannot create extreme heavy-tailed risk subgroups that
    # would spike the scorecard's KS; the tree challenger still finds them.
    spread_hi = (z_spread > 0.67).astype(np.float32)
    spread_lo = (z_spread < -0.67).astype(np.float32)
    emp_short = (emp_length_num <= 1.0).astype(np.float32)
    term60 = (term_num == 60).astype(np.float32)
    term_c = term60 - 0.35                   # centered term indicator

    lin = (LIN_SCALE * (W_FICO * z_fico + W_FICO_SUB * fico_sub
                       + W_DTI * z_dti + W_DTI_Q * dti_q
                       + W_RU * z_ru + W_RU_KINK * ru_kink
                       + W_DELINQ * z_delinq + W_INQ_SQRT * inq_sqrt
                       + W_GRADE * z_grade
                       + W_TERM60 * term60 + W_INC * z_inc + W_PUBREC * z_pubrec
                       + W_HI_UTIL * hi_util + W_HI_DTI * hi_dti
                       + W_SUBPRIME * subprime
                       + W_MANY_INQ * many_inq + W_THIN_FILE * thin_file))
    # Interaction block: uses pricing noise (z_spread) and loan term (term_c),
    # which are (near-)independent of borrower quality, so every term here is
    # orthogonal to the main effects. A linear/WOE scorecard cannot see these;
    # a tree model can. This is the XGBoost challenger's edge.
    inter = (INTER_W * (W_INT_SPREAD_GRADE * (spread_hi - spread_lo) * z_grade
                       + W_INT_SPREAD_DTI * (spread_hi - spread_lo) * z_dti
                       + W_INT_TERM_RU * term_c * z_ru
                       + W_INT_TERM_FICO * term_c * z_fico))
    logit = INTERCEPT + lin + inter + rng.standard_normal(n).astype(np.float32) * NOISE_SD
    pd_true = 1.0 / (1.0 + np.exp(-logit))
    default_24m = (rng.random(n) < pd_true).astype(np.int8)

    df = pd.DataFrame({
        "id": np.arange(id_offset, id_offset + n, dtype=np.int64),
        "member_id": rng.integers(1_000_000, 99_999_999, n, dtype=np.int64),
        "loan_amnt": loan_amnt,
        "funded_amnt": funded_amnt,
        "term": pd.Categorical(term),
        "int_rate": int_rate,
        "installment": installment,
        "grade": pd.Categorical(grade, categories=GRADES),
        "sub_grade": pd.Categorical(sub_grade),
        "emp_length": pd.Categorical(emp_length, categories=EMP_LENGTHS),
        "home_ownership": pd.Categorical(home_ownership),
        "annual_inc": annual_inc,
        "verification_status": pd.Categorical(verification_status),
        "issue_d": issue_d,
        "issue_ym": issue_ym,
        "purpose": pd.Categorical(purpose),
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "earliest_cr_line": earliest_cr_line,
        "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "application_type": pd.Categorical(application_type),
        "mort_acc": mort_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies,
        "addr_state": pd.Categorical(addr_state),
        "fico_range_low": fico_low,
        "fico_range_high": fico_high,
        "default_24m": default_24m,
    })
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    n_chunks = N_TOTAL // CHUNK_SIZE
    assert N_TOTAL % CHUNK_SIZE == 0
    total = 0
    for c in range(n_chunks):
        df = generate_chunk(CHUNK_SIZE, SEED + c, id_offset=c * CHUNK_SIZE)
        path = RAW_DIR / f"chunk_{c:02d}.parquet"
        df.to_parquet(path, index=False)
        total += len(df)
        print(f"chunk {c:02d}: {len(df):,} rows -> {path.name}", flush=True)
    print(f"TOTAL synthetic loans: {total:,}")


if __name__ == "__main__":
    main()
