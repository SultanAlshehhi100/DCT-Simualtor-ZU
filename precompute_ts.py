"""Fit the 9 default models (3 series x daily/weekly/monthly) and save results to data/ts_*.csv.
    python precompute_ts.py
Needs the prophet package. The Streamlit app reads these files, so it also runs where prophet is not installed."""
import pandas as pd
import timeseries as ts

F, B, M, C = [], [], [], []
for s in ts.SERIES:
    for k in ts.KINDS:
        f, b, m, c = ts.fit_all(s, k)
        for df, acc in ((f, F), (b, B), (c, C)):
            acc.append(df.assign(series=s, kind=k))
        M.append(m)
        print(f"{s:13s}{k:8s} WMAPE model {m['wmape_model']:.3f}  vs same-period-last-year {m['wmape_baseline']:.3f}  (train {m['n_train']}, test {m['n_test']})")
pd.concat(F).to_csv(ts.DATA / "ts_forecast.csv", index=False)
pd.concat(B).to_csv(ts.DATA / "ts_backtest.csv", index=False)
pd.concat(C).to_csv(ts.DATA / "ts_components.csv", index=False)
pd.DataFrame(M).to_csv(ts.DATA / "ts_metrics.csv", index=False)
