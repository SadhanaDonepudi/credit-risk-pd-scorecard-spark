# Information Value (IV) table — 60 bureau attributes

IV computed on the train sample (2016–2019 issues) with 0.5 Laplace smoothing. The scorecard keeps attributes with IV ≥ 0.02.

| rank | attribute | IV | bins | missing | in scorecard |
| --- | --- | --- | --- | --- | --- |
| 1 | `grade_num` | 0.3733 | 7 | 0 | yes |
| 2 | `dti` | 0.1380 | 11 | 0 | yes |
| 3 | `bc_util` | 0.1304 | 11 | 0 | yes |
| 4 | `revol_util` | 0.1304 | 11 | 0 | yes |
| 5 | `il_util` | 0.1304 | 11 | 0 | yes |
| 6 | `emp_length_num` | 0.1204 | 10 | 0 | yes |
| 7 | `total_acc` | 0.0680 | 11 | 0 | yes |
| 8 | `num_inst_tl` | 0.0674 | 9 | 0 | yes |
| 9 | `pct_tl_nvr_dlq` | 0.0649 | 4 | 0 | yes |
| 10 | `open_il_24m` | 0.0634 | 5 | 0 | yes |
| 11 | `loan_amnt_to_inc` | 0.0633 | 11 | 0 | yes |
| 12 | `total_bc_limit` | 0.0545 | 11 | 2,991 | yes |
| 13 | `total_rev_hi_lim` | 0.0545 | 11 | 2,991 | yes |
| 14 | `open_il_12m` | 0.0538 | 5 | 0 | yes |
| 15 | `mths_since_recent_bc_dlq` | 0.0507 | 3 | 316,439 | yes |
| 16 | `mths_since_last_delinq` | 0.0507 | 3 | 316,439 | yes |
| 17 | `num_tl_op_past_12m` | 0.0485 | 4 | 0 | yes |
| 18 | `delinq_2yrs` | 0.0444 | 3 | 0 | yes |
| 19 | `tot_coll_amt` | 0.0444 | 3 | 0 | yes |
| 20 | `delinq_amnt` | 0.0444 | 3 | 0 | yes |
| 21 | `num_tl_30dpd` | 0.0444 | 3 | 0 | yes |
| 22 | `mths_since_rcnt_il` | 0.0415 | 11 | 0 | yes |
| 23 | `mths_since_recent_il_acct` | 0.0415 | 11 | 0 | yes |
| 24 | `mths_since_oldest_il_open` | 0.0415 | 11 | 0 | yes |
| 25 | `mths_since_oldest_rev_tl` | 0.0415 | 11 | 0 | yes |
| 26 | `annual_inc` | 0.0305 | 11 | 0 | yes |
| 27 | `annual_inc_log` | 0.0305 | 11 | 0 | yes |
| 28 | `open_acc` | 0.0279 | 11 | 0 | yes |
| 29 | `num_rev_tl` | 0.0272 | 8 | 0 | yes |
| 30 | `num_actv_rev_tl` | 0.0269 | 7 | 0 | yes |
| 31 | `num_actv_bc_tl` | 0.0268 | 6 | 0 | yes |
| 32 | `num_bc_tl` | 0.0266 | 6 | 0 | yes |
| 33 | `open_rv_24m` | 0.0245 | 5 | 0 | yes |
| 34 | `open_acc_6m` | 0.0213 | 4 | 0 | yes |
| 35 | `open_rv_12m` | 0.0185 | 4 | 0 | no |
| 36 | `mths_since_recent_inq` | 0.0070 | 3 | 0 | no |
| 37 | `avg_cur_bal` | 0.0065 | 11 | 0 | no |
| 38 | `inq_last_12m` | 0.0064 | 3 | 0 | no |
| 39 | `inq_last_6mths` | 0.0064 | 3 | 0 | no |
| 40 | `mths_since_last_major_derog` | 0.0049 | 2 | 371,596 | no |
| 41 | `home_ownership_risk_wt` | 0.0032 | 4 | 0 | no |
| 42 | `revol_bal` | 0.0019 | 11 | 0 | no |
| 43 | `tot_hi_cred_lim` | 0.0019 | 11 | 0 | no |
| 44 | `total_bal_ex_mort` | 0.0019 | 11 | 0 | no |
| 45 | `max_bal_bc` | 0.0019 | 11 | 0 | no |
| 46 | `tot_cur_bal` | 0.0019 | 11 | 0 | no |
| 47 | `bal_to_hi_cred_lim` | 0.0019 | 11 | 0 | no |
| 48 | `mort_acc` | 0.0007 | 4 | 0 | no |
| 49 | `num_mort_tl` | 0.0007 | 4 | 0 | no |
| 50 | `collections_12_mths_ex_med` | 0.0000 | 2 | 0 | no |
| 51 | `num_tl_120dpd_2m` | 0.0000 | 2 | 0 | no |
| 52 | `num_accts_ever_120_pd` | 0.0000 | 2 | 0 | no |
| 53 | `num_tl_90g_dpd_24m` | 0.0000 | 2 | 0 | no |
| 54 | `inq_fi` | 0.0000 | 2 | 0 | no |
| 55 | `tax_liens` | 0.0000 | 2 | 0 | no |
| 56 | `acc_now_delinq` | 0.0000 | 2 | 0 | no |
| 57 | `pub_rec` | 0.0000 | 2 | 0 | no |
| 58 | `pub_rec_bankruptcies` | 0.0000 | 2 | 0 | no |
| 59 | `percent_bc_gt_75` | 0.0000 | 2 | 0 | no |
| 60 | `term_num` | 0.0000 | 2 | 0 | no |

_Rule of thumb: IV < 0.02 unpredictive; 0.02–0.1 weak; 0.1–0.3 medium; > 0.3 strong._
