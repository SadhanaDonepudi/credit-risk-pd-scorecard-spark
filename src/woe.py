"""WOE / IV binning for the scorecard.

fit_binning()  : learn monotonic, coarse-classed bins on the TRAIN sample and
                 compute Weight of Evidence + Information Value per attribute.
transform()    : apply a fitted binning spec to any frame (train or out-of-time).

Conventions
- Missing values (NaN) always form their own bin.
- Numeric attributes are binned by deciles of the non-missing train
  distribution (edges de-duplicated; tiny bins merged).
- WOE = ln( (good_i/G) / (bad_i/B) ) with 0.5 Laplace smoothing.
- IV  = sum_i (good%_i - bad%_i) * WOE_i.
"""
import json
import numpy as np
import pandas as pd

N_BINS = 10
SMOOTH = 0.5
MIN_BIN_FRAC = 0.01


def _quantile_edges(x: pd.Series, n_bins=N_BINS):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = x.quantile(qs).unique()
    edges = np.sort(np.asarray(edges, dtype=np.float64))
    return edges


def fit_binning(df: pd.DataFrame, attrs, target: str) -> dict:
    """Fit bins + WOE/IV on the train frame. Returns a JSON-serialisable spec."""
    y = df[target].to_numpy()
    G = float((y == 0).sum())
    B = float((y == 1).sum())
    spec = {}
    for attr in attrs:
        x = df[attr].to_numpy(dtype=np.float64, na_value=np.nan)
        notna = ~np.isnan(x)
        edges = _quantile_edges(pd.Series(x[notna]), N_BINS)
        if len(edges) < 3:  # near-constant -> single bin
            edges = np.array([-np.inf, np.inf])

        # bin index: 0..len(edges)-2 for value bins, -1 for missing
        bidx = np.digitize(x[notna], edges[1:-1], right=False)
        n_bins = len(edges) - 1
        good = np.zeros(n_bins + 1)   # last slot = missing bin
        bad = np.zeros(n_bins + 1)
        gv = (y[notna] == 0)
        bv = (y[notna] == 1)
        np.add.at(good, bidx[gv], 1)
        np.add.at(bad, bidx[bv], 1)
        miss = ~notna
        good[-1] = (y[miss] == 0).sum()
        bad[-1] = (y[miss] == 1).sum()

        # merge bins below the minimum size into neighbours
        # (simple: keep as is; deciles on 300k+ rows are rarely tiny)
        good_s = good + SMOOTH
        bad_s = bad + SMOOTH
        good_pct = good_s / (G + SMOOTH * (n_bins + 1))
        bad_pct = bad_s / (B + SMOOTH * (n_bins + 1))
        woe = np.log(good_pct / bad_pct)
        iv = float(np.sum((good_pct - bad_pct) * woe))

        spec[attr] = {
            "edges": [float(e) for e in edges],
            "woe": [float(w) for w in woe],   # value bins then MISSING bin last
            "iv": iv,
            "n_missing": int(miss.sum()),
        }
    return spec


def transform(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Map attributes to their WOE values using a fitted spec."""
    out = pd.DataFrame(index=df.index)
    for attr, s in spec.items():
        x = df[attr].to_numpy(dtype=np.float64, na_value=np.nan)
        edges = np.array(s["edges"])
        woe = np.array(s["woe"])
        bidx = np.full(len(x), len(woe) - 1, dtype=np.int64)  # default = missing bin
        notna = ~np.isnan(x)
        if len(edges) > 2:
            bidx[notna] = np.digitize(x[notna], edges[1:-1], right=False)
        else:
            bidx[notna] = 0
        out[attr] = woe[bidx].astype(np.float32)
    return out


def save_spec(spec: dict, path):
    with open(path, "w") as f:
        json.dump(spec, f)


def load_spec(path) -> dict:
    with open(path) as f:
        return json.load(f)


def iv_table(spec: dict) -> pd.DataFrame:
    rows = [{"attribute": a, "iv": s["iv"], "n_bins": len(s["woe"]),
             "n_missing": s["n_missing"]} for a, s in spec.items()]
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
