"""Shared configuration for the credit-risk PD scorecard project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- dataset -----------------------------------------------------------------
N_TOTAL = 2_200_000          # EXACT row count of the synthetic loan population
CHUNK_SIZE = 110_000         # rows per parquet chunk (20 chunks total)
SEED = 20261012              # master seed; chunk c uses SEED + c

RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEAT_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---- modeling ----------------------------------------------------------------
# Out-of-time split on loan issue date: train on 2016-2019, validate on 2020-2022
TRAIN_MAX_YM = 201912
OOT_MIN_YM = 202001

TARGET = "default_24m"
ID_COL = "id"
DATE_COL = "issue_ym"

# ---- the EXACT 60 bureau-style attributes (must match sql/transforms.sql) -----
ATTRIBUTES = [
    # Group A: history depth / recency (1-8)
    "mths_since_oldest_il_open", "mths_since_recent_inq",
    "mths_since_last_delinq", "mths_since_last_major_derog",
    "mths_since_recent_bc_dlq", "mths_since_rcnt_il",
    "mths_since_oldest_rev_tl", "mths_since_recent_il_acct",
    # Group B: delinquency / derogatory (9-19)
    "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_120dpd_2m",
    "num_accts_ever_120_pd", "delinq_2yrs", "delinq_amnt",
    "pct_tl_nvr_dlq", "pub_rec", "pub_rec_bankruptcies",
    "tax_liens", "acc_now_delinq",
    # Group C: collections (20-21)
    "tot_coll_amt", "collections_12_mths_ex_med",
    # Group D: inquiries (22-24)
    "inq_last_6mths", "inq_last_12m", "inq_fi",
    # Group E: balances / limits / utilization (25-37)
    "total_bal_ex_mort", "tot_hi_cred_lim", "total_bc_limit",
    "bc_util", "revol_bal", "revol_util", "il_util",
    "avg_cur_bal", "tot_cur_bal", "max_bal_bc",
    "percent_bc_gt_75", "total_rev_hi_lim", "bal_to_hi_cred_lim",
    # Group F: account mix counts (38-49)
    "num_bc_tl", "num_rev_tl", "num_inst_tl", "num_mort_tl",
    "num_actv_bc_tl", "num_actv_rev_tl", "total_acc", "open_acc",
    "open_acc_6m", "open_il_12m", "open_il_24m", "num_tl_op_past_12m",
    # Group G: recent openings / mortgage (50-52)
    "open_rv_12m", "open_rv_24m", "mort_acc",
    # Group H: borrower capacity / loan structure (53-60)
    "annual_inc", "annual_inc_log", "dti", "emp_length_num",
    "home_ownership_risk_wt", "grade_num", "term_num", "loan_amnt_to_inc",
]
assert len(ATTRIBUTES) == 60, f"attribute list must hold exactly 60, got {len(ATTRIBUTES)}"
