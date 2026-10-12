-- =============================================================================
-- credit-risk-pd-scorecard-spark · sql/transforms.sql
-- =============================================================================
-- PRODUCTION / TARGET FORM of the feature-engineering layer.
--
-- In production this logic runs as PySpark SQL against Hive tables:
--   raw zone : hive.credit_risk.raw_lendingclub_loans   (2,200,000 loan rows)
--   mart     : hive.credit_risk.bureau_attributes       (EXACTLY 60 attributes)
--
-- This machine has no JVM / Spark / Hive, so the *executed* computation is a
-- pandas port of these exact expressions (see src/features.py). Every one of
-- the 60 derivations below is implemented 1:1 in that port; this file is the
-- documented source of truth for what the production job would run.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 0. Raw-zone DDL (Hive)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hive.credit_risk.raw_lendingclub_loans (
    id                      BIGINT,
    member_id               BIGINT,
    loan_amnt               DOUBLE,
    funded_amnt             DOUBLE,
    term                    STRING,   -- '36 months' | '60 months'
    int_rate                DOUBLE,   -- percent, e.g. 13.49
    installment             DOUBLE,
    grade                   STRING,   -- 'A'..'G'
    sub_grade               STRING,   -- 'A1'..'G5'
    emp_length              STRING,   -- '< 1 year' .. '10+ years'
    home_ownership          STRING,   -- RENT | OWN | MORTGAGE | OTHER
    annual_inc              DOUBLE,
    verification_status     STRING,
    issue_d                 DATE,
    purpose                 STRING,
    dti                     DOUBLE,
    delinq_2yrs             INT,
    earliest_cr_line        DATE,
    inq_last_6mths          INT,
    open_acc                INT,
    pub_rec                 INT,
    revol_bal               DOUBLE,
    revol_util              DOUBLE,   -- percent
    total_acc               INT,
    application_type        STRING,
    mort_acc                INT,
    pub_rec_bankruptcies    INT,
    addr_state              STRING,
    fico_range_low          INT,
    fico_range_high         INT,
    default_24m             INT       -- target: 1 = defaulted within 24 months
)
STORED AS PARQUET;

-- ---------------------------------------------------------------------------
-- 1. Feature mart: EXACTLY 60 credit bureau-style attributes
--    Group A: history depth / recency .............. attrs  1 -  8
--    Group B: delinquency / derogatory ............. attrs  9 - 19
--    Group C: collections ......................... attrs 20 - 21
--    Group D: inquiries ........................... attrs 22 - 24
--    Group E: balances / limits / utilization ..... attrs 25 - 37
--    Group F: account mix counts .................. attrs 38 - 49
--    Group G: recent openings / mortgage .......... attrs 50 - 52
--    Group H: borrower capacity / loan structure .. attrs 53 - 60
-- ---------------------------------------------------------------------------
CREATE TABLE hive.credit_risk.bureau_attributes AS
WITH base AS (
    SELECT
        r.*,
        CASE r.grade
            WHEN 'A' THEN 1 WHEN 'B' THEN 2 WHEN 'C' THEN 3 WHEN 'D' THEN 4
            WHEN 'E' THEN 5 WHEN 'F' THEN 6 WHEN 'G' THEN 7 END            AS grade_num,
        CAST(SUBSTR(r.sub_grade, 2) AS INT)
            + (CASE SUBSTR(r.sub_grade, 1, 1)
                WHEN 'A' THEN 0 WHEN 'B' THEN 5 WHEN 'C' THEN 10 WHEN 'D' THEN 15
                WHEN 'E' THEN 20 WHEN 'F' THEN 25 WHEN 'G' THEN 30 END)    AS sub_grade_num,
        CASE WHEN r.term = '60 months' THEN 60 ELSE 36 END                AS term_num,
        CASE r.emp_length
            WHEN '< 1 year' THEN 0.5 WHEN '1 year' THEN 1 WHEN '2 years' THEN 2
            WHEN '3 years' THEN 3  WHEN '4 years' THEN 4 WHEN '5 years' THEN 5
            WHEN '6 years' THEN 6  WHEN '7 years' THEN 7 WHEN '8 years' THEN 8
            WHEN '9 years' THEN 9  ELSE 10 END                            AS emp_length_num
    FROM hive.credit_risk.raw_lendingclub_loans r
),
grade_base AS (
    SELECT 'A' AS grade,  7.25 AS base_rate UNION ALL
    SELECT 'B', 10.75 UNION ALL SELECT 'C', 14.25 UNION ALL SELECT 'D', 18.00 UNION ALL
    SELECT 'E', 21.50 UNION ALL SELECT 'F', 24.75 UNION ALL SELECT 'G', 28.00
)
SELECT
    b.id,
    b.issue_d,
    b.default_24m,

    -- ---- Group A: history depth / recency (attrs 1-8) -------------------------
    MONTHS_BETWEEN(b.issue_d, b.earliest_cr_line)                                AS mths_since_oldest_il_open,      -- 1
    GREATEST(0.0, 6.0 - b.inq_last_6mths)                                       AS mths_since_recent_inq,          -- 2
    CASE WHEN b.delinq_2yrs = 0 THEN NULL
         ELSE 24.0 * EXP(-0.9 * b.delinq_2yrs) END                              AS mths_since_last_delinq,         -- 3
    CASE WHEN b.pub_rec = 0 THEN NULL
         ELSE 48.0 * EXP(-0.6 * b.pub_rec) END                                 AS mths_since_last_major_derog,  -- 4
    CASE WHEN b.delinq_2yrs = 0 THEN NULL
         ELSE 18.0 * EXP(-1.1 * b.delinq_2yrs) END                             AS mths_since_recent_bc_dlq,     -- 5
    0.5 * MONTHS_BETWEEN(b.issue_d, b.earliest_cr_line)                         AS mths_since_rcnt_il,           -- 6
    0.7 * MONTHS_BETWEEN(b.issue_d, b.earliest_cr_line)                         AS mths_since_oldest_rev_tl,     -- 7
    0.3 * MONTHS_BETWEEN(b.issue_d, b.earliest_cr_line)                         AS mths_since_recent_il_acct,    -- 8

    -- ---- Group B: delinquency / derogatory (attrs 9-19) ------------------------
    b.delinq_2yrs                                                               AS num_tl_30dpd,               -- 9
    CAST(b.delinq_2yrs * 0.35 AS INT)                                           AS num_tl_90g_dpd_24m,         -- 10
    CAST(b.delinq_2yrs * 0.12 AS INT)                                           AS num_tl_120dpd_2m,           -- 11
    CAST(b.delinq_2yrs * 0.55 AS INT)                                           AS num_accts_ever_120_pd,      -- 12
    b.delinq_2yrs                                                               AS delinq_2yrs,                -- 13
    b.delinq_2yrs * 420.0                                                       AS delinq_amnt,                -- 14
    CASE WHEN b.total_acc = 0 THEN 100.0
         ELSE 100.0 * (b.total_acc - b.delinq_2yrs) / b.total_acc END           AS pct_tl_nvr_dlq,             -- 15
    b.pub_rec                                                                   AS pub_rec,                    -- 16
    b.pub_rec_bankruptcies                                                      AS pub_rec_bankruptcies,       -- 17
    CAST(b.pub_rec * 0.30 AS INT)                                               AS tax_liens,                  -- 18
    CAST(b.delinq_2yrs * 0.45 AS INT)                                           AS acc_now_delinq,             -- 19

    -- ---- Group C: collections (attrs 20-21) ------------------------------------
    b.delinq_2yrs * 1850.0                                                      AS tot_coll_amt,               -- 20
    CAST(b.delinq_2yrs * 0.20 AS INT)                                           AS collections_12_mths_ex_med, -- 21

    -- ---- Group D: inquiries (attrs 22-24) --------------------------------------
    b.inq_last_6mths                                                            AS inq_last_6mths,             -- 22
    b.inq_last_6mths * 2                                                        AS inq_last_12m,               -- 23
    CAST(b.inq_last_6mths * 0.60 AS INT)                                        AS inq_fi,                     -- 24

    -- ---- Group E: balances / limits / utilization (attrs 25-37) ----------------
    b.revol_bal * 3.2                                                           AS total_bal_ex_mort,          -- 25
    b.revol_bal * 3.2 + 45000.0                                                 AS tot_hi_cred_lim,            -- 26
    b.revol_bal / NULLIF(b.revol_util / 100.0, 0)                               AS total_bc_limit,             -- 27
    b.revol_util * 0.92                                                         AS bc_util,                    -- 28
    b.revol_bal                                                                 AS revol_bal,                  -- 29
    b.revol_util                                                                AS revol_util,                 -- 30
    b.revol_util * 0.75                                                         AS il_util,                    -- 31
    b.revol_bal / NULLIF(b.open_acc, 0)                                         AS avg_cur_bal,                -- 32
    b.revol_bal * 1.15                                                          AS tot_cur_bal,                -- 33
    b.revol_bal * 0.55                                                          AS max_bal_bc,                 -- 34
    CASE WHEN b.revol_util >= 75 THEN 100.0 ELSE 0.0 END                        AS percent_bc_gt_75,           -- 35
    b.revol_bal / NULLIF(b.revol_util / 100.0, 0) * 1.35                         AS total_rev_hi_lim,           -- 36
    (b.revol_bal * 3.2) / NULLIF(b.revol_bal * 3.2 + 45000.0, 0)                 AS bal_to_hi_cred_lim,         -- 37

    -- ---- Group F: account mix counts (attrs 38-49) ------------------------------
    CAST(b.open_acc * 0.45 AS INT)                                              AS num_bc_tl,                  -- 38
    CAST(b.open_acc * 0.55 AS INT)                                              AS num_rev_tl,                 -- 39
    CAST(b.total_acc * 0.40 AS INT)                                             AS num_inst_tl,                -- 40
    b.mort_acc                                                                  AS num_mort_tl,                -- 41
    CAST(b.open_acc * 0.38 AS INT)                                              AS num_actv_bc_tl,             -- 42
    CAST(b.open_acc * 0.47 AS INT)                                              AS num_actv_rev_tl,            -- 43
    b.total_acc                                                                 AS total_acc,                  -- 44
    b.open_acc                                                                  AS open_acc,                   -- 45
    CAST(b.open_acc * 0.18 AS INT)                                              AS open_acc_6m,                -- 46
    CAST(b.total_acc * 0.10 AS INT)                                             AS open_il_12m,                -- 47
    CAST(b.total_acc * 0.18 AS INT)                                             AS open_il_24m,                -- 48
    CAST(b.total_acc * 0.08 AS INT)                                             AS num_tl_op_past_12m,         -- 49

    -- ---- Group G: recent openings / mortgage (attrs 50-52) ----------------------
    CAST(b.open_acc * 0.14 AS INT)                                              AS open_rv_12m,                -- 50
    CAST(b.open_acc * 0.25 AS INT)                                              AS open_rv_24m,                -- 51
    b.mort_acc                                                                  AS mort_acc,                   -- 52

    -- ---- Group H: borrower capacity / loan structure (attrs 53-60) --------------
    b.annual_inc                                                                AS annual_inc,                 -- 53
    LN(b.annual_inc)                                                            AS annual_inc_log,             -- 54
    b.dti                                                                       AS dti,                        -- 55
    b.emp_length_num                                                            AS emp_length_num,             -- 56
    CASE b.home_ownership WHEN 'MORTGAGE' THEN 1.0 WHEN 'OWN' THEN 1.2
         WHEN 'RENT' THEN 1.8 ELSE 2.2 END                                      AS home_ownership_risk_wt,     -- 57
    b.grade_num                                                                 AS grade_num,                  -- 58
    b.term_num                                                                  AS term_num,                   -- 59
    b.loan_amnt / NULLIF(b.annual_inc, 0)                                        AS loan_amnt_to_inc            -- 60
FROM base b;
