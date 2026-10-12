# Adverse action reason codes (ECOA-style)

When credit is declined, the Equal Credit Opportunity Act requires the lender to disclose up to four specific reasons. Each reason below is mapped to the scorecard factors that drive it: for a declined applicant, the four factors with the largest negative points contributions are translated through this table.

## Master reason-code table

| code | reason (adverse action notice wording) | scorecard factors |
| --- | --- | --- |
| R01 | Serious delinquency | `delinq_2yrs`, `num_tl_30dpd`, `acc_now_delinq`, `num_accts_ever_120_pd` |
| R02 | Derogatory public record or collection on file | `pub_rec`, `pub_rec_bankruptcies`, `tax_liens`, `tot_coll_amt`, `collections_12_mths_ex_med` |
| R03 | Number of recent inquiries is too high | `inq_last_6mths`, `inq_last_12m`, `inq_fi` |
| R04 | Proportion of revolving balances to credit limits is too high | `revol_util`, `bc_util`, `il_util`, `percent_bc_gt_75`, `bal_to_hi_cred_lim` |
| R05 | Length of time accounts have been established is too short | `mths_since_oldest_il_open`, `mths_since_oldest_rev_tl`, `mths_since_recent_il_acct`, `mths_since_rcnt_il` |
| R06 | Too many accounts opened recently | `num_tl_op_past_12m`, `open_acc_6m`, `open_rv_12m`, `open_rv_24m`, `open_il_12m`, `open_il_24m` |
| R07 | Obligations too high in relation to income | `dti`, `loan_amnt_to_inc`, `annual_inc`, `annual_inc_log` |
| R08 | Limited credit history | `total_acc`, `num_bc_tl`, `num_rev_tl`, `num_inst_tl`, `open_acc`, `num_actv_bc_tl`, `num_actv_rev_tl` |
| R09 | Recent delinquency on accounts | `mths_since_last_delinq`, `mths_since_recent_bc_dlq`, `mths_since_last_major_derog`, `delinq_amnt`, `pct_tl_nvr_dlq` |
| R10 | Insufficient stability of employment / housing | `emp_length_num`, `home_ownership_risk_wt`, `grade_num`, `term_num`, `mths_since_recent_inq`, `total_bc_limit`, `total_rev_hi_lim`, `tot_hi_cred_lim`, `total_bal_ex_mort`, `avg_cur_bal`, `tot_cur_bal`, `max_bal_bc`, `num_mort_tl`, `mort_acc` |

## Scorecard factors ranked by discriminatory power

Points range (max − min bin points) per scorecard attribute, with its mapped reason code:

| rank | attribute | points range | reason code |
| --- | --- | --- | --- |
| 1 | `grade_num` | +61 | R10 |
| 2 | `dti` | +42 | R07 |
| 3 | `pct_tl_nvr_dlq` | +10 | R09 |
| 4 | `bc_util` | +7 | R04 |
| 5 | `revol_util` | +7 | R04 |
| 6 | `il_util` | +7 | R04 |
| 7 | `annual_inc_log` | +5 | R07 |
| 8 | `annual_inc` | +5 | R07 |
| 9 | `loan_amnt_to_inc` | +5 | R07 |
| 10 | `emp_length_num` | +4 | R10 |
| 11 | `open_il_12m` | +3 | R06 |
| 12 | `num_actv_rev_tl` | +2 | R08 |
| 13 | `tot_coll_amt` | +2 | R02 |
| 14 | `num_rev_tl` | +2 | R08 |
| 15 | `num_tl_30dpd` | +2 | R01 |
| 16 | `delinq_2yrs` | +2 | R01 |
| 17 | `delinq_amnt` | +2 | R09 |
| 18 | `mths_since_last_delinq` | +1 | R09 |
| 19 | `total_acc` | +1 | R08 |
| 20 | `open_rv_24m` | +1 | R06 |
| 21 | `num_tl_op_past_12m` | +1 | R06 |
| 22 | `num_inst_tl` | +1 | R08 |
| 23 | `num_bc_tl` | +1 | R08 |
| 24 | `mths_since_recent_bc_dlq` | +1 | R09 |
| 25 | `mths_since_oldest_il_open` | +0 | R05 |
| 26 | `mths_since_rcnt_il` | +0 | R05 |
| 27 | `num_actv_bc_tl` | +0 | R08 |
| 28 | `mths_since_recent_il_acct` | +0 | R05 |
| 29 | `mths_since_oldest_rev_tl` | +0 | R05 |
| 30 | `open_acc` | +0 | R08 |
| 31 | `open_acc_6m` | +0 | R06 |
| 32 | `open_il_24m` | +0 | R06 |
| 33 | `total_bc_limit` | +0 | R10 |
| 34 | `total_rev_hi_lim` | +0 | R10 |

## Worked example

A declined applicant whose largest negative contributions come from `revol_util` (R04), `delinq_2yrs` (R01), `inq_last_6mths` (R03) and `dti` (R07) receives a notice listing:

1. Proportion of revolving balances to credit limits is too high
2. Serious delinquency
3. Number of recent inquiries is too high
4. Obligations too high in relation to income
