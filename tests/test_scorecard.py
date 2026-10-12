"""Test suite for the credit-risk PD scorecard repo.

Covers: exact dataset row count, exact attribute count (60), WOE/IV
correctness, model metric sanity, and README-number consistency with
results/metrics.json (the measured source of truth).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from config import ATTRIBUTES, N_TOTAL, TARGET  # noqa: E402


def _mini_frame(n=2000, seed=999):
    from generate import generate_chunk
    from features import derive
    return derive(generate_chunk(n, seed, id_offset=0))


# ----------------------------------------------------------------------------
def test_attribute_count_is_exactly_sixty():
    assert len(ATTRIBUTES) == 60
    assert len(set(ATTRIBUTES)) == 60, "attribute names must be unique"
    df = _mini_frame()
    attr_cols = [c for c in df.columns
                 if c not in ("id", "issue_ym", TARGET, "loan_amnt")]
    assert attr_cols == ATTRIBUTES
    assert len(attr_cols) == 60


def test_row_count_exact():
    raw_dir = ROOT / "data" / "raw"
    chunks = sorted(raw_dir.glob("chunk_*.parquet"))
    if chunks:
        total = sum(len(pd.read_parquet(p, columns=["id"])) for p in chunks)
        assert total == N_TOTAL, f"{total:,} != {N_TOTAL:,}"
    else:
        # no generated data present: verify the generator yields exact counts
        from generate import generate_chunk
        a = generate_chunk(5000, 1, 0)
        b = generate_chunk(3000, 2, 5000)
        assert len(a) == 5000 and len(b) == 3000
        assert a["id"].min() == 0 and b["id"].min() == 5000


def test_feature_row_count_matches_raw():
    raw_dir = ROOT / "data" / "raw"
    feat_dir = ROOT / "data" / "features"
    raw = sorted(raw_dir.glob("chunk_*.parquet"))
    feat = sorted(feat_dir.glob("chunk_*.parquet"))
    if not raw or not feat:
        pytest.skip("generated data not present")
    n_raw = sum(len(pd.read_parquet(p, columns=["id"])) for p in raw)
    n_feat = sum(len(pd.read_parquet(p, columns=["id"])) for p in feat)
    assert n_raw == n_feat == N_TOTAL


# ----------------------------------------------------------------------------
def test_woe_iv_correctness():
    """Verify the WOE/IV formula: recompute bin counts independently from the
    fitted edges and check WOE = ln((good_i/G)/(bad_i/B)) with 0.5 smoothing."""
    from woe import fit_binning
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 4000)
    # P(bad) rises steeply with x
    y = (rng.random(4000) < 1 / (1 + np.exp(-(6 * x - 3)))).astype(int)
    df = pd.DataFrame({"x": x, TARGET: y})
    spec = fit_binning(df, ["x"], TARGET)
    s = spec["x"]
    edges = np.array(s["edges"])
    nb_value_bins = len(edges) - 1
    nb_total = nb_value_bins + 1  # + missing bin
    G = float((y == 0).sum())
    B = float((y == 1).sum())
    bidx = np.digitize(x, edges[1:-1], right=False)
    for b in range(nb_value_bins):
        good_i = float(((y == 0) & (bidx == b)).sum())
        bad_i = float(((y == 1) & (bidx == b)).sum())
        exp_woe = np.log(((good_i + 0.5) / (G + 0.5 * nb_total))
                         / ((bad_i + 0.5) / (B + 0.5 * nb_total)))
        assert s["woe"][b] == pytest.approx(exp_woe, rel=1e-9)
    # IV = sum (good% - bad%) * WOE over value bins AND the missing bin
    # (no missing here, but smoothing still contributes its 0.5/0.5 term)
    woe_all = np.array(s["woe"])
    good_c = np.array([((y == 0) & (bidx == b)).sum()
                       for b in range(nb_value_bins)] + [0.0])
    bad_c = np.array([((y == 1) & (bidx == b)).sum()
                      for b in range(nb_value_bins)] + [0.0])
    good_pct = (good_c + 0.5) / (G + 0.5 * nb_total)
    bad_pct = (bad_c + 0.5) / (B + 0.5 * nb_total)
    exp_iv = float(np.sum((good_pct - bad_pct) * woe_all))
    assert s["iv"] == pytest.approx(exp_iv, rel=1e-9)
    assert s["iv"] > 0.5
    # WOE must be monotone decreasing in x (higher x -> higher bad rate)
    assert all(a >= b for a, b in
               zip(s["woe"][:nb_value_bins], s["woe"][1:nb_value_bins]))


def test_woe_transform_applies_bins():
    from woe import fit_binning, transform
    df = _mini_frame(5000)
    spec = fit_binning(df, ["dti", "revol_util"], TARGET)
    W = transform(df, spec)
    assert list(W.columns) == ["dti", "revol_util"]
    assert np.isfinite(W.to_numpy()).all()


# ----------------------------------------------------------------------------
def _metrics():
    p = ROOT / "results" / "metrics.json"
    if not p.exists():
        pytest.skip("results/metrics.json not present (run src/run_all.py)")
    return json.loads(p.read_text())


def test_metric_sanity():
    m = _metrics()
    assert m["n_loans"] == N_TOTAL
    assert m["n_attributes"] == 60
    sc, xg = m["scorecard"], m["xgboost"]
    assert 0.5 < sc["auc"] < 1.0
    assert 0.0 < xg["auc"] < 1.0
    assert 0 < sc["ks_points"] < 100
    assert 0 < sc["gini"] < 1 and 0 < xg["gini"] < 1
    assert 0 < m["gini_improvement_points"] < 100
    assert m["psi_scorecard_train_vs_oot"] >= 0
    assert 0 < m["cutoff"]["loss_reduction_pct"] < 100
    assert abs(m["cutoff"]["approval_rate_achieved"] - 0.70) < 0.01


def test_readme_number_consistency():
    """Every number in the README must equal results/metrics.json."""
    m = _metrics()
    readme = (ROOT / "README.md").read_text()
    assert "{{" not in readme, "unrendered template placeholder in README"

    def find(pat):
        mt = re.search(pat, readme)
        assert mt, f"pattern not found: {pat}"
        return float(mt.group(1).replace(",", ""))

    assert find(r"Scorecard AUC \| \*\*([\d.]+)\*\*") == pytest.approx(
        m["scorecard"]["auc"], abs=5e-4)
    assert find(r"Scorecard KS \| \*\*([\d.]+)\*\*") == pytest.approx(
        m["scorecard"]["ks_points"], abs=5e-2)
    assert find(r"Gini improvement over scorecard \| \*\*\+([\d.]+) points\*\*") == \
        pytest.approx(m["gini_improvement_points"], abs=5e-2)
    assert find(r"Population stability index \(train vs OOT\) \| "
                r"\*\*([\d.]+)\*\* \(< 0\.05\)") == pytest.approx(
        m["psi_scorecard_train_vs_oot"], abs=5e-5)
    assert find(r"at 70% approval \(vs legacy scorecard\) \| \*\*([\d.]+)%\*\*") == \
        pytest.approx(m["cutoff"]["loss_reduction_pct"], abs=5e-2)
    assert f"{m['n_loans']:,}" in readme
    assert str(m["n_attributes"]) in readme


def test_sql_documents_sixty_attributes():
    sql = (ROOT / "sql" / "transforms.sql").read_text()
    # each attribute is documented with a numbered marker "-- N" (N = 1..60)
    markers = set(int(x) for x in re.findall(r"-- (\d+)\s*$", sql, re.M))
    assert markers == set(range(1, 61)), f"missing markers: {set(range(1,61)) - markers}"
