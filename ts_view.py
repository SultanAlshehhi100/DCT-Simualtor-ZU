"""Streamlit screen for the time-series (hotel occupancy) forecasts."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import timeseries as ts

BLUE, GREY, GREEN, ORANGE, RED = "#2563eb", "#8a94a6", "#2f9e44", "#d9480f", "#c92a2a"
SEASON_COL = {"Winter": "#3b82f6", "Spring": "#22c55e", "Summer": "#f59e0b", "Autumn": "#a16207"}
UNIT = {"daily": "day", "weekly": "week (starting Monday)", "monthly": "month"}


@st.cache_data(show_spinner="Fitting the model…")
def _get(series, kind, params):
    r = ts.get_result(series, kind, params)
    return r


def _fmt(v):
    return f"{v:,.0f}"


def _full_chart(res, kind):
    hist, fc, lg = res["hist"], res["forecast"], res["lgbm"]
    last = hist["ds"].max()
    fut = fc[fc["ds"] > last]
    f = go.Figure()
    f.add_scatter(x=hist["ds"], y=hist["y"], name="Actual", line=dict(color=GREY, width=1.5))
    f.add_scatter(x=fut["ds"], y=fut["hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    f.add_scatter(x=fut["ds"], y=fut["lo"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(37,99,235,.15)",
                  name="Prophet 80% range", hoverinfo="skip")
    f.add_scatter(x=fut["ds"], y=fut["yhat"], name="Prophet forecast", line=dict(color=BLUE, width=2))
    if len(lg):
        f.add_scatter(x=lg["ds"], y=lg["y"], name="LightGBM forecast (our main model)", line=dict(color=GREEN, width=2, dash="dash"))
    f.add_vline(x=last, line_dash="dot", line_color=GREY)
    f.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Guests in hotels per night",
                    legend=dict(orientation="h", y=1.12), hovermode="x unified")
    return f


def _year_chart(res, kind, year, top, mode):
    c = ts.add_index(ts.combined(res), res["peak"])
    c = c[c["ds"].dt.year == year]
    f = go.Figure()
    if kind == "monthly":
        c = c.assign(season=c["ds"].dt.month.map(ts.SEASON))
        f.add_bar(x=c["ds"].dt.strftime("%b"), y=c["y"], marker_color=[SEASON_COL[s] for s in c["season"]],
                  customdata=c[["season", "occ_index"]], hovertemplate="%{x} (%{customdata[0]})<br>%{y:,.0f} guests/night<br>index %{customdata[1]:.0f}<extra></extra>")
        f.update_layout(showlegend=False)
    else:
        f.add_scatter(x=c["ds"], y=c["y"], mode="lines", line=dict(color=BLUE, width=2), name="Guests per night")
        f.add_scatter(x=top["ds"], y=top["y"], mode="markers+text", text=top["rank"], textposition="top center",
                      marker=dict(color=RED, size=11), name=f"Top {len(top)}")
    f.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Guests in hotels per night",
                    legend=dict(orientation="h", y=1.12), hovermode="x unified" if kind != "monthly" else "closest")
    return f


def _tab(kind, series, year, topn, params):
    res = _get(series, kind, params)
    m = res["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backtest error (WMAPE)", f"{m['wmape_model']:.1%}", help="Trained on data before 2025, tested on Jan–Jul 2025. Lower is better.")
    c2.metric("Same period last year", f"{m['wmape_baseline']:.1%}", help="Naive baseline: repeat the value from a year earlier.")
    better = m["wmape_model"] < m["wmape_baseline"]
    c3.metric("Model vs baseline", "Better" if better else "Not better", delta=f"{(m['wmape_baseline'] - m['wmape_model']) * 100:+.1f} pts", delta_color="normal")
    c4.metric("Peak in history", f"{res['peak']:,.0f}", help="Highest Actual value at this granularity (= occupancy index 100).")
    if kind == "monthly":
        st.warning("Monthly has only 43 data points (36 for training the backtest), so treat this model as indicative. "
                   + ("" if better else "In the backtest it did **not** beat the same-month-last-year baseline."))
    if res["live"]:
        st.info("Showing a live fit with your custom settings.")

    st.markdown(f"**Whole timeline** – actuals (grey), our LightGBM forecast to Feb 2026 (green dashed) and the Prophet forecast to Dec 2026 (blue).")
    st.plotly_chart(_full_chart(res, kind), width="stretch")

    top = ts.top_periods(res, year, topn)
    st.markdown(f"**{year} by {UNIT[kind]}**" + (" – coloured by season" if kind == "monthly" else f" – the top {topn} are marked 1…{topn}"))
    st.plotly_chart(_year_chart(res, kind, year, top, kind), width="stretch")

    ev = load_events_cached()
    if kind == "monthly":
        mm = ts.monthly_by_season(res, year)
        st.markdown(f"**Monthly occupancy grouped by season, {year}**")
        t = mm[["season", "month", "y", "occ_index", "source"]].rename(columns={
            "season": "Season", "month": "Month", "y": "Guests per night", "occ_index": "Occupancy index", "source": "Source"})
        t["Guests per night"] = t["Guests per night"].round(0).astype(int)
        t["Occupancy index"] = t["Occupancy index"].round(0).astype(int)
        st.dataframe(t, hide_index=True, width="stretch")
        s = mm.groupby("season", observed=True).agg(avg=("y", "mean"), idx=("occ_index", "mean")).reset_index()
        s.columns = ["Season", "Avg guests per night", "Avg occupancy index"]
        s = s.round(0)
        st.dataframe(s, hide_index=True, width="stretch")
        st.markdown(f"**Top {min(topn, 12)} months of {year}**")
    else:
        st.markdown(f"**Top {topn} most occupied {'days' if kind == 'daily' else 'weeks'} of {year}**")
    t = top.copy()
    end = pd.Timedelta(days={"daily": 0, "weekly": 6, "monthly": 0}[kind])
    if kind == "monthly":
        t["events"] = [", ".join(sorted(set(ts.SEASON[d.month] for d in [r]))) for r in t["ds"]]
        t["Label"] = t["ds"].dt.strftime("%B %Y")
    else:
        t["events"] = [ts.events_in(r, r + end, ev) for r in t["ds"]]
        t["Label"] = t["ds"].dt.strftime("%a %d %b %Y")
    out = pd.DataFrame({"Rank": t["rank"], UNIT[kind].split(" ")[0].capitalize(): t["Label"],
                        "Guests per night": t["y"].round(0).astype(int), "Occupancy index": t["occ_index"].round(0).astype(int),
                        "Source": t["source"].map({"Actual": "Actual", "Prophet forecast": "Prophet forecast"}),
                        ("Season" if kind == "monthly" else "Events in this period"): t["events"]})
    st.dataframe(out, hide_index=True, width="stretch")
    st.caption("Occupancy index: 100 = the highest value in Jan 2022 – Jul 2025 at this granularity. It can exceed 100 in a forecast. "
               "Guests per night, not room occupancy %: we have no hotel capacity data.")

    with st.expander("What drives this model (components)"):
        comp = res["components"]
        for name, d in comp.groupby("component", sort=False):
            d = d.copy()
            d["effect"] = d["effect"] * 100
            fig = go.Figure(go.Bar(x=d["key"], y=d["effect"], marker_color=[GREEN if v >= 0 else RED for v in d["effect"]]))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), title=name, yaxis_title="% vs trend")
            if name == "Month of year":
                fig.update_xaxes(tickvals=[str(i) for i in range(1, 13)])
            st.plotly_chart(fig, width="stretch")
        if kind == "daily":
            st.caption("Positive = busier than the trend. In our data the busiest nights are Friday and Saturday; Sunday is the quietest.")
        if kind == "weekly":
            st.caption("Effect of a week fully inside the event. Event dates come from data/events.csv (rows marked 'Please check' are unverified).")

    with st.expander("Backtest: Jan–Jul 2025 predicted vs actual"):
        b = res["backtest"]
        f = go.Figure()
        f.add_scatter(x=b["ds"], y=b["y"], name="Actual", line=dict(color=GREY))
        f.add_scatter(x=b["ds"], y=b["yhat"], name="Model", line=dict(color=BLUE))
        f.add_scatter(x=b["ds"], y=b["base"], name="Same period last year", line=dict(color=ORANGE, dash="dot"))
        f.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.15), hovermode="x unified")
        st.plotly_chart(f, width="stretch")


@st.cache_data
def load_events_cached():
    return ts.load_events()


def render():
    st.subheader("Hotel occupancy time series")
    st.caption("Guests staying in Abu Dhabi hotels per night, by day, week and month. Actuals Jan 2022 – Jul 2025, "
               "our LightGBM forecast Aug 2025 – Feb 2026, and a Prophet time-series model carried on to Dec 2026.")
    c1, c2, c3 = st.columns(3)
    series = c1.selectbox("Guests", ts.SERIES, index=0, help="Total = International + Domestic")
    year = c2.selectbox("Year", [2022, 2023, 2024, 2025, 2026], index=4)
    topn = c3.slider("Top N periods", 3, 20, 10)

    params = ts.DEFAULT
    with st.expander("Model settings (optional)"):
        if not ts.PROPHET_OK:
            st.info("Prophet is not installed here, so the default precomputed models are shown and these settings are disabled.")
        mode = st.radio("Seasonality", ["multiplicative", "additive"], horizontal=True, disabled=not ts.PROPHET_OK,
                        help="Multiplicative: seasonal swings grow with the level.")
        cps = st.slider("Trend flexibility", 0.01, 0.5, 0.05, disabled=not ts.PROPHET_OK)
        iw = st.slider("Forecast range width", 0.5, 0.95, 0.80, disabled=not ts.PROPHET_OK)
        evs = st.checkbox("Use events in weekly model", True, disabled=not ts.PROPHET_OK)
        if ts.PROPHET_OK:
            params = ts.Params(mode, cps, iw, evs)

    d, w, m = st.tabs(["📅 Daily (weekends)", "🗓️ Weekly (events & conventions)", "📆 Monthly (seasons)"])
    with d:
        _tab("daily", series, year, topn, params)
    with w:
        _tab("weekly", series, year, topn, params)
    with m:
        _tab("monthly", series, year, min(topn, 12), params)
