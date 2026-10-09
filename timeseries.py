"""Time-series (Prophet) models of hotel guests: daily, weekly and monthly.

"Occupancy" here = guests staying in Abu Dhabi hotels per night. No room-capacity data exists, so the
occupancy INDEX is guests / (highest value in 2022-Jul 2025 at the same granularity) x 100.

Daily   : weekly seasonality (weekends) + yearly seasonality
Weekly  : event / convention / holiday regressors (share of the week's days inside each event) + yearly
Monthly : season dummies (Winter is the baseline) + trend
"""
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    PROPHET_OK = True
except Exception:  # prophet missing or failed to build -> app falls back to precomputed files
    PROPHET_OK = False

DATA = Path(__file__).resolve().parent / "data"
SERIES = ["Total", "International", "Domestic"]
KINDS = ["daily", "weekly", "monthly"]
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
          6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
TRAIN_END = pd.Timestamp("2025-01-01")      # backtest: train before, test from here to end of actuals
FORECAST_END = pd.Timestamp("2026-12-31")
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass(frozen=True)
class Params:
    seasonality_mode: str = "multiplicative"
    changepoint_prior_scale: float = 0.05
    interval_width: float = 0.80
    use_events: bool = True


DEFAULT = Params()


# ---------------------------------------------------------------- data
def load_daily() -> pd.DataFrame:
    d = pd.read_csv(DATA / "daily_guests.csv", parse_dates=["Date"]).set_index("Date")
    d["Total"] = d["International"] + d["Domestic"]
    return d


def load_events() -> pd.DataFrame:
    e = pd.read_csv(DATA / "events.csv", parse_dates=["Start", "End"])
    return e.dropna(subset=["Group", "Start", "End"])


def aggregate(daily: pd.DataFrame, series: str, kind: str, source=None) -> pd.DataFrame:
    """Mean guests per night for complete days/weeks/months. source: 'Actual' / 'Forecast' / None (all)."""
    d = daily if source is None else daily[daily["Source"] == source]
    s = d[series]
    if kind == "daily":
        out = s
    else:
        rule = "W-MON" if kind == "weekly" else "MS"
        # weekly bins start on Monday
        g = s.resample(rule, label="left", closed="left") if kind == "weekly" else s.resample(rule)
        out = g.mean()
        n = g.count()
        need = 7 if kind == "weekly" else None
        if kind == "monthly":
            need = pd.Series(out.index.days_in_month, index=out.index)
        out = out[n >= need]
    return out.rename("y").rename_axis("ds").reset_index()


def occupancy_peak(daily, series, kind):
    return float(aggregate(daily, series, kind, "Actual")["y"].max())


def event_share(ds: pd.Series, events: pd.DataFrame, group: str) -> np.ndarray:
    """For each week start in ds: share of its 7 days that fall inside the group's event windows."""
    days = set()
    for _, r in events[events["Group"] == group].iterrows():
        days.update(pd.date_range(r["Start"], r["End"]))
    return np.array([sum((t + pd.Timedelta(days=i)) in days for i in range(7)) / 7 for t in ds])


def events_in(start, end, events=None):
    events = load_events() if events is None else events
    m = (events["Start"] <= end) & (events["End"] >= start)
    return ", ".join(sorted(set(events.loc[m, "Group"])))


# ---------------------------------------------------------------- models
def _build(kind, p: Params, events):
    m = dict(seasonality_mode=p.seasonality_mode, changepoint_prior_scale=p.changepoint_prior_scale,
             interval_width=p.interval_width, daily_seasonality=False)
    if kind == "daily":
        mod = Prophet(weekly_seasonality=True, yearly_seasonality=10, **m)
    elif kind == "weekly":
        mod = Prophet(weekly_seasonality=False, yearly_seasonality=6, **m)
    else:
        mod = Prophet(weekly_seasonality=False, yearly_seasonality=False, **m)
    return mod


def _add_regressors(mod, df, kind, p, events):
    names = []
    if kind == "weekly" and p.use_events:
        for g in sorted(events["Group"].unique()):
            df[g] = event_share(df["ds"], events, g)
            mod.add_regressor(g, prior_scale=0.5, mode="multiplicative" if p.seasonality_mode == "multiplicative" else "additive")
            names.append(g)
    if kind == "monthly":
        for s in ["Spring", "Summer", "Autumn"]:
            df[s] = (df["ds"].dt.month.map(SEASON) == s).astype(float)
            mod.add_regressor(s, prior_scale=10.0, standardize=False)
            names.append(s)
    return names


def _frame(ds, kind, p, events):
    df = pd.DataFrame({"ds": pd.to_datetime(ds)})
    if kind == "weekly" and p.use_events:
        for g in sorted(events["Group"].unique()):
            df[g] = event_share(df["ds"], events, g)
    if kind == "monthly":
        for s in ["Spring", "Summer", "Autumn"]:
            df[s] = (df["ds"].dt.month.map(SEASON) == s).astype(float)
    return df


def _fit_predict(train: pd.DataFrame, future_ds, kind, p, events):
    df = train[["ds", "y"]].copy()
    mod = _build(kind, p, events)
    names = _add_regressors(mod, df, kind, p, events)
    mod.fit(df)
    fut = _frame(future_ds, kind, p, events)
    fc = mod.predict(fut)
    return mod, fc, names


def _wmape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def _future_index(last, kind):
    if kind == "daily":
        return pd.date_range(last + pd.Timedelta(days=1), FORECAST_END)
    if kind == "weekly":
        return pd.date_range(last + pd.Timedelta(days=7), FORECAST_END, freq="7D")
    return pd.date_range(last + pd.offsets.MonthBegin(1), FORECAST_END, freq="MS")


def _lag(actual: pd.DataFrame, kind):
    s = actual.set_index("ds")["y"]
    if kind == "daily":
        return s.shift(364)
    if kind == "weekly":
        return s.shift(52)
    return s.shift(12)


def fit_all(series, kind, p: Params = DEFAULT):
    """Backtest + final fit. Returns forecast, backtest, metrics and components tables."""
    if not PROPHET_OK:
        raise RuntimeError("prophet is not installed")
    daily, events = load_daily(), load_events()
    act = aggregate(daily, series, kind, "Actual")
    # --- backtest
    tr, te = act[act["ds"] < TRAIN_END], act[act["ds"] >= TRAIN_END]
    _, fcb, _ = _fit_predict(tr, te["ds"], kind, p, events)
    base = _lag(act, kind).reindex(te["ds"]).values
    bt = pd.DataFrame({"ds": te["ds"].values, "y": te["y"].values, "yhat": fcb["yhat"].values, "base": base})
    ok = ~np.isnan(bt["base"])
    metrics = dict(series=series, kind=kind, n_train=len(tr), n_test=len(te),
                   wmape_model=_wmape(bt["y"], bt["yhat"]),
                   wmape_baseline=_wmape(bt.loc[ok, "y"], bt.loc[ok, "base"]) if ok.any() else np.nan)
    # --- final fit on all actuals, forecast to end of 2026 (in-sample + future)
    fut = _future_index(act["ds"].max(), kind)
    allds = pd.concat([act["ds"], pd.Series(fut)], ignore_index=True)
    mod, fc, names = _fit_predict(act, allds, kind, p, events)
    out = pd.DataFrame({"ds": fc["ds"], "yhat": fc["yhat"], "lo": fc["yhat_lower"], "hi": fc["yhat_upper"]})
    # --- components (effects relative to trend)
    comp = []
    tr_ = fc["trend"].values
    if kind == "daily":
        w = fc[["ds", "weekly"]].copy()
        w["dow"] = w["ds"].dt.dayofweek
        for i, v in w.groupby("dow")["weekly"].mean().items():
            comp.append(("Day of week", DOW[i], float(v)))
        y = fc[["ds", "yearly"]].copy()
        y["m"] = y["ds"].dt.month
        for i, v in y.groupby("m")["yearly"].mean().items():
            comp.append(("Month of year", str(i), float(v)))
    elif kind == "weekly":
        y = fc[["ds", "yearly"]].copy()
        y["m"] = y["ds"].dt.month
        for i, v in y.groupby("m")["yearly"].mean().items():
            comp.append(("Month of year", str(i), float(v)))
        for n in names:   # effect of a full week inside the event
            comp.append(("Event effect (full week)", n, float(fc[n].iloc[fc[n].abs().argmax()])))
    else:
        for s in ["Spring", "Summer", "Autumn"]:
            comp.append(("Season vs Winter", s, float(fc[s].iloc[fc[s].abs().argmax()])))
    comp = pd.DataFrame(comp, columns=["component", "key", "effect"])
    return out, bt, metrics, comp


def get_result(series, kind, p: Params = DEFAULT):
    """Default settings use the precomputed files (fast, works without prophet). Custom settings fit live."""
    daily = load_daily()
    res = dict(daily=daily, series=series, kind=kind, live=False)
    if p == DEFAULT and (DATA / "ts_forecast.csv").exists():
        f = pd.read_csv(DATA / "ts_forecast.csv", parse_dates=["ds"])
        b = pd.read_csv(DATA / "ts_backtest.csv", parse_dates=["ds"])
        m = pd.read_csv(DATA / "ts_metrics.csv")
        c = pd.read_csv(DATA / "ts_components.csv")
        sel = lambda d: d[(d["series"] == series) & (d["kind"] == kind)].drop(columns=["series", "kind"]).reset_index(drop=True)
        res.update(forecast=sel(f), backtest=sel(b), metrics=sel(m).iloc[0].to_dict(), components=sel(c))
    elif PROPHET_OK:
        f, b, m, c = fit_all(series, kind, p)
        res.update(forecast=f, backtest=b, metrics=m, components=c, live=True)
    else:
        raise RuntimeError("Custom settings need the prophet package; the default view uses precomputed files.")
    res["hist"] = aggregate(daily, series, kind, "Actual")
    res["lgbm"] = aggregate(daily, series, kind, "Forecast")
    res["peak"] = occupancy_peak(daily, series, kind)
    return res


# ---------------------------------------------------------------- summaries
def combined(res):
    """One line: actuals to Jul 2025, then LightGBM (Aug 2025-Feb 2026), then Prophet for the rest of 2026."""
    hist = res["hist"].assign(source="Actual")
    fc = res["forecast"]
    last_actual = hist["ds"].max()
    fut = fc[fc["ds"] > last_actual].rename(columns={"yhat": "y"})[["ds", "y", "lo", "hi"]].assign(source="Prophet forecast")
    return pd.concat([hist, fut], ignore_index=True)


def add_index(df, peak):
    df = df.copy()
    df["occ_index"] = df["y"] / peak * 100
    return df


def top_periods(res, year: int, n: int, use_prophet_for_future=True):
    c = add_index(combined(res), res["peak"])
    c = c[c["ds"].dt.year == year]
    # for weeks use the week-start's year; for months the month's year
    out = c.sort_values("y", ascending=False).head(n).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out


def monthly_by_season(res, year: int):
    c = add_index(combined(res), res["peak"])
    c = c[c["ds"].dt.year == year].copy()
    c["season"] = c["ds"].dt.month.map(SEASON)
    c["month"] = c["ds"].dt.strftime("%b")
    c["season"] = pd.Categorical(c["season"], SEASON_ORDER, ordered=True)
    return c.sort_values(["season", "ds"])
