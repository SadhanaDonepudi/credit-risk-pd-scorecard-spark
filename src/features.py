"""Feature mart builder: pandas port of sql/transforms.sql.

Executes the EXACT 60 attribute derivations documented in sql/transforms.sql
(the production/target Spark SQL form). Because no JVM/Spark/Hive exists on
this machine, this module computes the same expressions in pandas, chunk by
chunk, and writes the bureau-attributes mart as parquet.

Each derivation is numbered to match the SQL file (attrs 1-60).
"""
import numpy as np
import pandas as pd

from config import RAW_DIR, FEAT_DIR, ATTRIBUTES, TARGET, ID_COL, DATE_COL

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
HO_WT = {"MORTGAGE": 1.0, "OWN": 1.2, "RENT": 1.8, "OTHER": 2.2}
EMP_MAP = {"< 1 year": 0.5, "1 year": 1.0, "2 years": 2.0, "3 years": 3.0,
           "4 years": 4.0, "5 years": 5.0, "6 years": 6.0, "7 years": 7.0,
           "8 years": 8.0, "9 years": 9.0, "10+ years": 10.0}


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the 60 documented derivations. Returns mart with keys + 60 attrs."""
    d = df  # raw frame
    out = pd.DataFrame()
    out[ID_COL] = d["id"].astype(np.int64)
    out[DATE_COL] = d["issue_ym"].astype(np.int32)
    out[TARGET] = d["default_24m"].astype(np.int8)
    # exposure-at-default for the expected-loss math (NOT a model attribute)
    out["loan_amnt"] = d["loan_amnt"].astype(np.float32)

    f32 = lambda s: s.astype(np.float32)
    i16 = lambda s: s.astype(np.int16)

    delinq = d["delinq_2yrs"].astype(np.float32)
    pubrec = d["pub_rec"].astype(np.float32)
    inq = d["inq_last_6mths"].astype(np.float32)
    open_acc = d["open_acc"].astype(np.float32)
    total_acc = d["total_acc"].astype(np.float32)
    revol_bal = d["revol_bal"].astype(np.float32)
    revol_util = d["revol_util"].astype(np.float32)

    # months between issue_d and earliest_cr_line (Spark MONTHS_BETWEEN approx)
    mso = ((pd.to_datetime(d["issue_d"]) - pd.to_datetime(d["earliest_cr_line"]))
           .dt.days / 30.4375)

    # ---- Group A: history depth / recency (1-8) -------------------------------
    out["mths_since_oldest_il_open"] = f32(mso)                                   # 1
    out["mths_since_recent_inq"] = f32(np.maximum(0.0, 6.0 - inq))               # 2
    out["mths_since_last_delinq"] = f32(np.where(delinq == 0, np.nan,             # 3
                                                 24.0 * np.exp(-0.9 * delinq)))
    out["mths_since_last_major_derog"] = f32(np.where(pubrec == 0, np.nan,        # 4
                                                      48.0 * np.exp(-0.6 * pubrec)))
    out["mths_since_recent_bc_dlq"] = f32(np.where(delinq == 0, np.nan,           # 5
                                                   18.0 * np.exp(-1.1 * delinq)))
    out["mths_since_rcnt_il"] = f32(0.5 * mso)                                   # 6
    out["mths_since_oldest_rev_tl"] = f32(0.7 * mso)                             # 7
    out["mths_since_recent_il_acct"] = f32(0.3 * mso)                            # 8

    # ---- Group B: delinquency / derogatory (9-19) ------------------------------
    out["num_tl_30dpd"] = i16(delinq)                                            # 9
    out["num_tl_90g_dpd_24m"] = i16(delinq * 0.35)                               # 10
    out["num_tl_120dpd_2m"] = i16(delinq * 0.12)                                 # 11
    out["num_accts_ever_120_pd"] = i16(delinq * 0.55)                            # 12
    out["delinq_2yrs"] = i16(delinq)                                            # 13
    out["delinq_amnt"] = f32(delinq * 420.0)                                    # 14
    out["pct_tl_nvr_dlq"] = f32(100.0 * (total_acc - delinq)                    # 15
                                / np.where(total_acc == 0, np.nan, total_acc))
    out["pub_rec"] = i16(pubrec)                                                # 16
    out["pub_rec_bankruptcies"] = d["pub_rec_bankruptcies"].astype(np.int8)      # 17
    out["tax_liens"] = i16(pubrec * 0.30)                                       # 18
    out["acc_now_delinq"] = i16(delinq * 0.45)                                   # 19

    # ---- Group C: collections (20-21) ------------------------------------------
    out["tot_coll_amt"] = f32(delinq * 1850.0)                                  # 20
    out["collections_12_mths_ex_med"] = i16(delinq * 0.20)                       # 21

    # ---- Group D: inquiries (22-24) --------------------------------------------
    out["inq_last_6mths"] = i16(inq)                                            # 22
    out["inq_last_12m"] = i16(inq * 2)                                          # 23
    out["inq_fi"] = i16(inq * 0.60)                                             # 24

    # ---- Group E: balances / limits / utilization (25-37) -----------------------
    out["total_bal_ex_mort"] = f32(revol_bal * 3.2)                             # 25
    out["tot_hi_cred_lim"] = f32(revol_bal * 3.2 + 45000.0)                      # 26
    out["total_bc_limit"] = f32(np.where(revol_util == 0, np.nan,                 # 27
                                         revol_bal / (revol_util / 100.0)))
    out["bc_util"] = f32(revol_util * 0.92)                                     # 28
    out["revol_bal"] = f32(revol_bal)                                           # 29
    out["revol_util"] = f32(revol_util)                                         # 30
    out["il_util"] = f32(revol_util * 0.75)                                     # 31
    out["avg_cur_bal"] = f32(np.where(open_acc == 0, np.nan,                      # 32
                                      revol_bal / open_acc))
    out["tot_cur_bal"] = f32(revol_bal * 1.15)                                  # 33
    out["max_bal_bc"] = f32(revol_bal * 0.55)                                   # 34
    out["percent_bc_gt_75"] = f32(np.where(revol_util >= 75, 100.0, 0.0))        # 35
    out["total_rev_hi_lim"] = f32(np.where(                                      # 36
        revol_util == 0, np.nan, revol_bal / (revol_util / 100.0) * 1.35))
    out["bal_to_hi_cred_lim"] = f32((revol_bal * 3.2)                            # 37
                                    / (revol_bal * 3.2 + 45000.0))

    # ---- Group F: account mix counts (38-49) ------------------------------------
    out["num_bc_tl"] = i16(open_acc * 0.45)                                     # 38
    out["num_rev_tl"] = i16(open_acc * 0.55)                                    # 39
    out["num_inst_tl"] = i16(total_acc * 0.40)                                  # 40
    out["num_mort_tl"] = d["mort_acc"].astype(np.int8)                          # 41
    out["num_actv_bc_tl"] = i16(open_acc * 0.38)                                # 42
    out["num_actv_rev_tl"] = i16(open_acc * 0.47)                                # 43
    out["total_acc"] = d["total_acc"].astype(np.int16)                          # 44
    out["open_acc"] = d["open_acc"].astype(np.int8)                             # 45
    out["open_acc_6m"] = i16(open_acc * 0.18)                                   # 46
    out["open_il_12m"] = i16(total_acc * 0.10)                                  # 47
    out["open_il_24m"] = i16(total_acc * 0.18)                                  # 48
    out["num_tl_op_past_12m"] = i16(total_acc * 0.08)                            # 49

    # ---- Group G: recent openings / mortgage (50-52) -----------------------------
    out["open_rv_12m"] = i16(open_acc * 0.14)                                   # 50
    out["open_rv_24m"] = i16(open_acc * 0.25)                                   # 51
    out["mort_acc"] = d["mort_acc"].astype(np.int8)                            # 52

    # ---- Group H: borrower capacity / loan structure (53-60) ----------------------
    out["annual_inc"] = f32(d["annual_inc"])                                    # 53
    out["annual_inc_log"] = f32(np.log(d["annual_inc"].astype(np.float32)))     # 54
    out["dti"] = f32(d["dti"])                                                  # 55
    out["emp_length_num"] = f32(d["emp_length"].map(EMP_MAP).astype(np.float32))# 56
    out["home_ownership_risk_wt"] = f32(                                        # 57
        d["home_ownership"].map(HO_WT).astype(np.float32))
    out["grade_num"] = d["grade"].map(GRADE_MAP).astype(np.int8)                # 58
    out["term_num"] = np.where(d["term"] == "60 months", 60, 36).astype(np.int8) # 59
    out["loan_amnt_to_inc"] = f32(d["loan_amnt"].astype(np.float32)              # 60
                                  / d["annual_inc"].astype(np.float32))

    # guard: column order and the exact-60 contract
    # (id / issue_ym / default_24m / loan_amnt are keys & target & EAD, not attributes)
    attr_cols = [c for c in out.columns
                 if c not in (ID_COL, DATE_COL, TARGET, "loan_amnt")]
    assert attr_cols == ATTRIBUTES, "derived columns do not match the 60-attribute spec"
    return out


def main():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = sorted(RAW_DIR.glob("chunk_*.parquet"))
    assert chunks, "no raw chunks found; run generate.py first"
    for p in chunks:
        df = pd.read_parquet(p)
        mart = derive(df)
        outp = FEAT_DIR / p.name
        mart.to_parquet(outp, index=False)
        print(f"{p.name}: {len(mart):,} rows x {len(ATTRIBUTES)} attributes -> {outp.name}",
              flush=True)


if __name__ == "__main__":
    main()
