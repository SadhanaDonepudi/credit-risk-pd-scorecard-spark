"""In-memory calibration run (~200k rows) to tune the generator.

Prints measured AUC / KS / Gini / Gini-delta / PSI / loss-reduction (new
scorecard vs legacy scorecard at 70% approval) so the generator
hyper-parameters in generate.py can be adjusted BEFORE the full 2.2M run.
Uses the exact same pipeline code as run_all.py.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from generate import generate_chunk
from features import derive
from pipeline import (fit_scorecard, scorecard_proba, fit_xgboost,
                      xgboost_proba, fit_legacy, legacy_proba,
                      auc, ks_points, gini, psi, strategy_table,
                      headline_loss_reduction)
from config import TARGET, DATE_COL, TRAIN_MAX_YM, OOT_MIN_YM, SEED

N_CAL = 200_000


def main():
    feats = []
    for c in range(2):
        raw = generate_chunk(N_CAL // 2, SEED + 1000 + c, id_offset=c * (N_CAL // 2))
        feats.append(derive(raw))
    df = pd.concat(feats, ignore_index=True)
    print(f"calibration frame: {len(df):,} rows, bad rate {df[TARGET].mean():.3f}")

    train = df[df[DATE_COL] <= TRAIN_MAX_YM]
    oot = df[df[DATE_COL] >= OOT_MIN_YM]
    print(f"train {len(train):,} | oot {len(oot):,}")

    tr = train.sample(min(120_000, len(train)), random_state=0)
    va = oot.sample(min(80_000, len(oot)), random_state=1)
    y = va[TARGET].to_numpy()
    ead = va["loan_amnt"].to_numpy(dtype=np.float64)

    legacy = fit_legacy(tr)
    p_leg = legacy_proba(legacy, va)

    sc = fit_scorecard(tr)
    p_lr = scorecard_proba(sc, va)
    a_lr, ks_lr, g_lr = auc(y, p_lr), ks_points(y, p_lr), gini(y, p_lr)
    print(f"LEGACY     AUC={auc(y, p_leg):.4f}")
    print(f"SCORECARD  AUC={a_lr:.4f}  KS={ks_lr:.2f}  Gini={g_lr:.4f}  "
          f"kept {len(sc['features'])} attrs")

    xgb_model = fit_xgboost(tr)
    p_xgb = xgboost_proba(xgb_model, va)
    a_xgb, g_xgb = auc(y, p_xgb), gini(y, p_xgb)
    print(f"XGBOOST   AUC={a_xgb:.4f}  Gini={g_xgb:.4f}  "
          f"dGini_pts={(g_xgb - g_lr) * 100:.2f}")

    # PSI of scorecard score, train vs out-of-time
    p_tr = scorecard_proba(sc, tr.sample(min(80_000, len(tr)), random_state=2))
    print(f"PSI (scorecard, train vs OOT) = {psi(pd.Series(p_tr), pd.Series(p_lr)):.4f}")

    strat = {"legacy_scorecard": p_leg, "new_scorecard": p_lr, "xgboost": p_xgb}
    table = strategy_table(va, strat, ead)
    head = headline_loss_reduction(table, "legacy_scorecard", "new_scorecard", 0.70)
    print(f"CUTOFF@70%  legacy_loss={head['expected_loss_baseline']:,.0f}  "
          f"new_loss={head['expected_loss_challenger']:,.0f}  "
          f"reduction={head['loss_reduction_pct']:.2f}%")
    print(table.pivot(index="approval_rate", columns="strategy",
                      values="expected_loss").to_string())


if __name__ == "__main__":
    main()
