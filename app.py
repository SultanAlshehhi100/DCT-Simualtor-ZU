from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import run_chain

DATA = Path(__file__).resolve().parent / "data"
BLUE, GREY, ORANGE, GREEN = "#2563eb", "#8a94a6", "#d9480f", "#2f9e44"
AIRCRAFT = {"A320 (~180 seats)": 180, "A321neo (~220)": 220, "Boeing 787-9 (~290)": 290,
            "Boeing 777-300ER (~350)": 350, "Airbus A380 (~480)": 480}

st.set_page_config(page_title="Flight → Hotel Simulator · Abu Dhabi", page_icon="✈️", layout="wide")


# --------------------------------------------------------------------------- data
@st.cache_data
def load():
    intl = pd.read_csv(DATA / "intl_monthly.csv")
    routes = pd.read_csv(DATA / "routes.csv")
    link = pd.read_csv(DATA / "market_link.csv")
    dom = pd.read_csv(DATA / "domestic_monthly.csv")
    return intl, routes, link, dom


intl, routes, link, dom = load()
MONTHS = sorted(intl["Month"].unique())
dom_fc = dom[dom["Type"] == "Forecast"].set_index("Month")["Guest_Nights"]
intl_total = intl.groupby("Month")["Guest_Nights"].sum()
GLOBAL_YIELD = float(link["Capture"].median())          # used when a market has no direct flights yet


def days_in(month: str) -> int:
    return pd.Period(month, freq="M").days_in_month


def market_values(nat: str, month: str):
    """Hotel-side numbers for one nationality-month, with sensible fallbacks."""
    row = link[(link["Nationality"] == nat) & (link["Month"] == month)]
    allrows = link[link["Nationality"] == nat]
    stay = row["Avg_Stay"].iloc[0] if len(row) and pd.notna(row["Avg_Stay"].iloc[0]) else allrows["Avg_Stay"].mean()
    yld = row["Capture"].iloc[0] if len(row) and pd.notna(row["Capture"].iloc[0]) else allrows["Capture"].median()
    nights = float(row["Guest_Nights"].iloc[0]) if len(row) else 0.0
    has_flights = bool(allrows["Has_Direct_Flight"].any())
    if pd.isna(yld):
        yld = GLOBAL_YIELD
    return float(stay), float(yld), nights, has_flights


def rng(lo, hi, val):
    """Slider limits that always contain the exact starting value (so 'no change' really means zero impact)."""
    return float(min(lo, val)), float(max(hi, val))


def fmt(x, signed=False):
    return f"{x:+,.0f}" if signed else f"{x:,.0f}"


# --------------------------------------------------------------------------- header
st.title("✈️ Flight → Hotel Guest Simulator · Abu Dhabi")
st.caption("What happens to hotel guest nights if flights change? Baseline = our forecast for Aug 2025 – Feb 2026. "
           "Every step of the chain is visible and adjustable.")

tab_sim, tab_over, tab_notes = st.tabs(["🎛️ Scenario simulator", "📊 Market overview", "ℹ️ Method & limits"])

# =========================================================================== SIMULATOR
with tab_sim:
    mode = st.radio("What do you want to test?",
                    ["Change an existing route", "Open a new route / add flights from a country"],
                    horizontal=True)

    c1, c2 = st.columns([1, 3])
    month = c1.selectbox("Month", MONTHS, index=MONTHS.index("2025-12") if "2025-12" in MONTHS else 0)
    d = days_in(month)

    # ---------------------------------------------------------------- MODE A: existing route
    if mode.startswith("Change"):
        rm = routes[routes["Month"] == month]
        countries = sorted(rm["Country"].unique())
        cols = c2.columns(3)
        country = cols[0].selectbox("Departure country", countries, index=countries.index("India") if "India" in countries else 0)
        r_c = rm[rm["Country"] == country]
        city = cols[1].selectbox("Departure city", sorted(r_c["City"].unique()))
        airline = cols[2].selectbox("Airline", sorted(r_c[r_c["City"] == city]["Airline"].unique()))
        route = r_c[(r_c["City"] == city) & (r_c["Airline"] == airline)].iloc[0]
        nat = country.upper()

        stay0, yield0, nat_nights, _ = market_values(nat, month)
        b_seats, b_pax, b_p2p = float(route["Seats"]), float(route["PAX"]), float(route["P2P"])
        b_lf = b_pax / b_seats if b_seats else 0.0
        b_tr = float(route["Transfer"]) / b_pax if b_pax else 0.0
        b_ts = float(route["Transit"]) / b_pax if b_pax else 0.0

        st.markdown(f"##### Baseline · {airline}, {city} → Abu Dhabi · {month}")
        m = st.columns(5)
        m[0].metric("Seats / month", fmt(b_seats))
        m[1].metric("Load factor", f"{b_lf:.0%}")
        m[2].metric("Point-to-point visitors", fmt(b_p2p))
        m[3].metric(f"{country} hotel guest nights", fmt(nat_nights))
        m[4].metric("Check-ins per P2P passenger", f"{yield0:.2f}")

        st.markdown("##### Scenario controls")
        k = st.columns(3)
        freq_pct = k[0].slider("Flight frequency change", -100, 200, 0, 5, format="%+d%%",
                               help="+50% = one and a half times as many flights. −100% = route cancelled.")
        size_pct = k[0].slider("Aircraft size change (seats per flight)", -50, 50, 0, 5, format="%+d%%",
                               help="+20% ≈ swapping an A320 for an A321, or adding seats to the same aircraft.")
        lo, hi = rng(0.30, 1.20, b_lf)
        lf = k[1].slider("Load factor", lo, hi, float(b_lf), 0.01, format="%.2f",
                         help="Share of seats filled. Some routes are slightly above 100% in the data (infants).")
        lo, hi = rng(0.0, 0.95, b_tr)
        transfer = k[1].slider("Transfer share of passengers", lo, hi, float(b_tr), 0.01, format="%.2f",
                               help="Passengers who change planes in Abu Dhabi and never reach a hotel.")
        lo, hi = rng(0.05, 5.0, yield0)
        yld = k[2].slider("Check-ins per P2P passenger", lo, hi, float(yield0), 0.01, format="%.2f",
                          help="Hotel check-ins for every passenger who ends their trip in Abu Dhabi. "
                               "Can exceed 1 because guests of this nationality also arrive via other countries.")
        lo, hi = rng(0.5, 10.0, stay0)
        stay = k[2].slider("Average length of stay (nights)", lo, hi, float(stay0), 0.1, format="%.1f")

        def seats_for(month_row_seats, fp=freq_pct, sp=size_pct):
            return month_row_seats * (1 + fp / 100) * (1 + sp / 100)

        base_chain = run_chain(b_seats, b_lf, b_tr, b_ts, yield0, stay0)
        scen_chain = run_chain(seats_for(b_seats), lf, transfer, b_ts, yld, stay)
        base_nights = base_chain.nights                       # = this route's share of the country's nights
        scenario_seats = seats_for(b_seats)

        def with_changes(**kw):                               # for the sensitivity chart
            p = dict(seats=scenario_seats, lf=lf, tr=transfer, ts=b_ts, yld=yld, stay=stay)
            p.update(kw)
            return run_chain(p["seats"], p["lf"], p["tr"], p["ts"], p["yld"], p["stay"]).nights - base_nights

        # same change applied to every month this route flies
        profile = []
        for mm in MONTHS:
            r2 = routes[(routes["Month"] == mm) & (routes["Country"] == country) &
                        (routes["City"] == city) & (routes["Airline"] == airline)]
            if r2.empty:
                continue
            r2 = r2.iloc[0]
            s2, y2, _, _ = market_values(nat, mm)
            ps, pp = float(r2["Seats"]), float(r2["PAX"])
            lf2 = pp / ps if ps else 0
            tr2, ts2 = (float(r2["Transfer"]) / pp if pp else 0), (float(r2["Transit"]) / pp if pp else 0)
            bn = run_chain(ps, lf2, tr2, ts2, y2, s2).nights
            sn = run_chain(seats_for(ps), np.clip(lf2 + (lf - b_lf), 0, 1.5), np.clip(tr2 + (transfer - b_tr), 0, 0.99),
                           ts2, y2 * (yld / yield0 if yield0 else 1), s2 * (stay / stay0 if stay0 else 1)).nights
            profile.append((mm, sn - bn))
        base_label = "this route"
        baseline_for_pct = nat_nights if nat_nights else np.nan
        per_flight = None

    # ---------------------------------------------------------------- MODE B: new route / add flights
    else:
        nats = sorted(intl["Nationality"].unique())
        nat = c2.columns(3)[0].selectbox("Source market (nationality)", nats, index=nats.index("GERMANY"))
        stay0, yield0, nat_nights, has_flights = market_values(nat, month)
        lk = link[(link["Nationality"] == nat) & (link["Month"] == month)]
        mkt_lf = float(lk["PAX"].iloc[0] / lk["Seats"].iloc[0]) if has_flights and len(lk) and lk["Seats"].iloc[0] else 0.80
        mkt_tr = float((lk["Transfer"].iloc[0] + lk["Transit"].iloc[0]) / lk["PAX"].iloc[0]) if has_flights and len(lk) and lk["PAX"].iloc[0] else 0.20
        mkt_tr = min(max(mkt_tr, 0.0), 0.9)
        mkt_lf = min(max(mkt_lf, 0.3), 1.2)

        st.markdown(f"##### Baseline · {nat.title()} · {month}")
        m = st.columns(4)
        m[0].metric("Hotel guest nights (forecast)", fmt(nat_nights))
        m[1].metric("Direct flights today", "Yes" if has_flights else "None in the data")
        m[2].metric("Check-ins per P2P passenger", f"{yield0:.2f}")
        m[3].metric("Average stay (nights)", f"{stay0:.1f}")
        if not has_flights:
            st.info(f"{nat.title()} has **no direct flights** in the flight data, so there's no market-specific check-in "
                    f"ratio. The simulator uses the median across all markets ({GLOBAL_YIELD:.2f}). Treat this as an assumption.")

        st.markdown("##### Scenario controls")
        k = st.columns(3)
        weekly = k[0].slider("Extra flights per week", 0, 28, 2, 1,
                             help="2 = a new twice-weekly service. 14 = twice daily.")
        ac = k[0].selectbox("Aircraft", list(AIRCRAFT) + ["Custom seats per flight"], index=2)
        spf = AIRCRAFT[ac] if ac in AIRCRAFT else k[0].number_input("Seats per flight", 50, 600, 250, 10)
        lf = k[1].slider("Load factor", 0.30, 1.20, float(mkt_lf), 0.01, format="%.2f")
        transfer = k[1].slider("Transfer share of passengers", 0.0, 0.95, float(mkt_tr), 0.01, format="%.2f",
                               help="Passengers who change planes in Abu Dhabi and never reach a hotel.")
        lo, hi = rng(0.05, 5.0, yield0)
        yld = k[2].slider("Check-ins per P2P passenger", lo, hi, float(yield0), 0.01, format="%.2f")
        lo, hi = rng(0.5, 10.0, stay0)
        stay = k[2].slider("Average length of stay (nights)", lo, hi, float(stay0), 0.1, format="%.1f")

        base_nights = 0.0
        scenario_seats = weekly * d / 7 * spf
        scen_chain = run_chain(scenario_seats, lf, transfer, 0.0, yld, stay)
        base_chain = run_chain(0, 0, 0, 0, 0, 0)

        def with_changes(**kw):
            p = dict(seats=scenario_seats, lf=lf, tr=transfer, ts=0.0, yld=yld, stay=stay)
            p.update(kw)
            return run_chain(p["seats"], p["lf"], p["tr"], p["ts"], p["yld"], p["stay"]).nights

        profile = []
        for mm in MONTHS:
            s2, y2, _, _ = market_values(nat, mm)
            seats2 = weekly * days_in(mm) / 7 * spf
            profile.append((mm, run_chain(seats2, lf, transfer, 0.0, y2 * (yld / yield0 if yield0 else 1),
                                          s2 * (stay / stay0 if stay0 else 1)).nights))
        baseline_for_pct = nat_nights if nat_nights else np.nan
        per_flight = scen_chain.nights / weekly if weekly else None

    # ---------------------------------------------------------------- results
    extra_nights = scen_chain.nights - base_nights
    extra_checkins = scen_chain.checkins - base_chain.checkins
    total_base = float(intl_total.get(month, 0) + dom_fc.get(month, 0))

    st.markdown("##### Result")
    r = st.columns(4)
    r[0].metric("Extra seats / month", fmt(scen_chain.seats - base_chain.seats, True))
    r[1].metric("Extra P2P visitors", fmt(scen_chain.p2p - base_chain.p2p, True))
    r[2].metric("Extra hotel check-ins", fmt(extra_checkins, True))
    r[3].metric("Extra guest nights", fmt(extra_nights, True),
                f"{extra_nights / baseline_for_pct:+.1%} of {nat.title()}'s nights" if baseline_for_pct == baseline_for_pct else None)
    r2 = st.columns(3)
    r2[0].metric("Share of all Abu Dhabi hotel nights (intl + domestic)", f"{extra_nights / total_base:+.2%}" if total_base else "–")
    r2[1].metric("Baseline total hotel nights this month", fmt(total_base))
    r2[2].metric("Scenario total hotel nights this month", fmt(total_base + extra_nights), fmt(extra_nights, True))
    if per_flight:
        st.caption(f"≈ **{per_flight:,.0f} extra guest nights per month for each weekly flight added** from {nat.title()}.")

    st.markdown(
        f"**Chain:** {scen_chain.seats:,.0f} seats → × {lf:.0%} load factor = **{scen_chain.pax:,.0f} passengers** → "
        f"× {scen_chain.p2p_share:.0%} not transferring = **{scen_chain.p2p:,.0f} P2P visitors** → "
        f"× {yld:.2f} check-ins each = **{scen_chain.checkins:,.0f} check-ins** → × {stay:.1f} nights = "
        f"**{scen_chain.nights:,.0f} guest nights**" + (f" (baseline {base_nights:,.0f})" if base_nights else ""))
    if yld > 1:
        st.warning("Check-ins per P2P passenger is above 1 for this market: more guests of this nationality check in than "
                   "fly in directly from the country (they arrive via other hubs). Read the result as the market-level "
                   "relationship, which may overstate what one new route alone would add.")

    # ---------------------------------------------------------------- charts
    left, right = st.columns(2)
    with left:
        if profile:
            pf = pd.DataFrame(profile, columns=["Month", "Extra"])
            fig = go.Figure(go.Bar(x=pf["Month"], y=pf["Extra"],
                                   marker_color=[BLUE if v >= 0 else ORANGE for v in pf["Extra"]],
                                   text=[fmt(v, True) for v in pf["Extra"]], textposition="outside"))
            fig.update_layout(title="Same change, every month: extra guest nights", height=360,
                              yaxis_title="Guest nights", margin=dict(t=50, b=10))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("This route doesn't operate in the other months.")
    with right:
        sens = [
            ("Flight frequency ±25%", with_changes(seats=scenario_seats * 0.75), with_changes(seats=scenario_seats * 1.25)),
            ("Aircraft size ±15%", with_changes(seats=scenario_seats * 0.85), with_changes(seats=scenario_seats * 1.15)),
            ("Load factor ±10 pts", with_changes(lf=max(lf - 0.10, 0.3)), with_changes(lf=min(lf + 0.10, 1.2))),
            ("Transfer share ±10 pts", with_changes(tr=min(transfer + 0.10, 0.95)), with_changes(tr=max(transfer - 0.10, 0))),
            ("Check-ins per P2P ±20%", with_changes(yld=yld * 0.8), with_changes(yld=yld * 1.2)),
            ("Length of stay ±1 night", with_changes(stay=max(stay - 1, 0.5)), with_changes(stay=stay + 1)),
        ]
        sens.sort(key=lambda s: abs(s[2] - s[1]))
        fig2 = go.Figure()
        fig2.add_bar(y=[s[0] for s in sens], x=[s[1] - extra_nights for s in sens], orientation="h",
                     name="Lower", marker_color=ORANGE)
        fig2.add_bar(y=[s[0] for s in sens], x=[s[2] - extra_nights for s in sens], orientation="h",
                     name="Higher", marker_color=BLUE)
        fig2.update_layout(barmode="overlay", title="Which lever matters most? Change in extra guest nights",
                           height=360, xaxis_title="vs. your scenario (guest nights)", margin=dict(t=50, b=10),
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig2, width="stretch")
        st.caption("Each bar moves one lever across a plausible range, leaving the others where you set them.")

# =========================================================================== OVERVIEW
with tab_over:
    st.subheader("Baseline forecast: hotel guest nights, Aug 2025 – Feb 2026")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_bar(x=MONTHS, y=[intl_total.get(x, 0) for x in MONTHS], name="International", marker_color=BLUE)
        fig.add_bar(x=MONTHS, y=[dom_fc.get(x, 0) for x in MONTHS], name="Domestic", marker_color=GREY)
        fig.update_layout(barmode="stack", height=340, title="International + domestic", yaxis_title="Guest nights",
                          margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
    with b:
        d2 = dom.copy()
        d2["Date"] = pd.to_datetime(d2["Month"] + "-01")
        fig = go.Figure()
        for t, colr in [("Actual", GREY), ("Forecast", ORANGE)]:
            s = d2[d2["Type"] == t]
            fig.add_scatter(x=s["Date"], y=s["Guest_Nights"], name=t, mode="lines+markers", line=dict(color=colr))
        fig.update_layout(height=340, title="Domestic: history and forecast", yaxis_title="Guest nights",
                          margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
    st.caption("Domestic demand is held constant in every scenario: international flights don't change it.")

    tot = link.groupby("Nationality").agg(Guest_Nights=("Guest_Nights", "sum"), Seats=("Seats", "sum"),
                                          Direct=("Has_Direct_Flight", "any")).reset_index()
    tot["Share"] = tot["Guest_Nights"] / tot["Guest_Nights"].sum()

    a, b = st.columns(2)
    with a:
        st.markdown("**Largest source markets (7-month guest nights)**")
        top = tot.sort_values("Guest_Nights", ascending=False).head(12).copy()
        top["Direct flights"] = np.where(top["Direct"], "Yes", "No")
        st.dataframe(top[["Nationality", "Guest_Nights", "Share", "Direct flights"]]
                     .style.format({"Guest_Nights": "{:,.0f}", "Share": "{:.1%}"}),
                     hide_index=True, width="stretch")
    with b:
        st.markdown("**Markets with no direct flight: candidates for new routes**")
        nf = tot[~tot["Direct"]].sort_values("Guest_Nights", ascending=False)
        st.dataframe(nf[["Nationality", "Guest_Nights", "Share"]]
                     .style.format({"Guest_Nights": "{:,.0f}", "Share": "{:.1%}"}),
                     hide_index=True, width="stretch")

    st.markdown("**Hotel guest nights per available seat (markets with direct flights)**")
    ps = tot[tot["Direct"] & (tot["Seats"] > 0)].copy()
    ps["Per_Seat"] = ps["Guest_Nights"] / ps["Seats"]
    ps = ps.sort_values("Per_Seat", ascending=True).tail(20)
    fig = go.Figure(go.Bar(x=ps["Per_Seat"], y=ps["Nationality"].str.title(), orientation="h", marker_color=BLUE))
    fig.update_layout(height=520, xaxis_title="Guest nights per seat", margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Top 20. A seat on a flight from the top markets fills far more hotel nights than a seat from a hub or Gulf market.")

# =========================================================================== NOTES
with tab_notes:
    st.subheader("How it works")
    st.markdown("""
**Baseline.** International guest nights come from our gradient-boosting model's forecast for Aug 2025 – Feb 2026
(held-out back-test error ≈ 11% per day, 7% per month by nationality). Domestic guest nights are the domestic team's forecast.
Flights are the monthly route totals from the flight data.

**Chain, one line per step**
```
Seats            = baseline seats × (1 + frequency change) × (1 + aircraft size change)     [new route: flights/week × days/7 × seats per flight]
Passengers       = seats × load factor
P2P visitors     = passengers × (1 − transfer share − transit share)
Hotel check-ins  = P2P visitors × check-ins per P2P passenger        (market's own ratio from the data)
Guest nights     = hotel check-ins × average length of stay           (market's own value from the data)
Impact           = scenario guest nights − baseline guest nights
```
With nothing changed the impact is exactly zero. The ratio and the length of stay are calculated from the same forecast as the
baseline, so the baseline route nights reconcile with the country total.

**Limits to state when presenting**
- **Nationality ≠ departure country.** Guest nationality and flight origin are matched by country name only. For about 40% of
  country-months, check-ins exceed point-to-point passengers (China, USA, Canada), because those guests also arrive via other hubs.
  The ratio is an association, not proof that one new route brings that many guests.
- **Not causal.** The simulator scales a historical relationship; it doesn't prove a new flight creates demand, and it ignores
  hotel capacity, prices, and visa or event effects.
- **No direct flights (12 markets).** There's no market-specific ratio, so the median across markets is used; adjust it.
- **Frequency is entered as a % change.** The "Average Weekly Frequency" column in the flight file looks unreliable on small
  routes, so absolute frequency isn't used for existing routes.
- **Domestic demand is constant** across scenarios.
- Forecast months only (Aug 2025 – Feb 2026).
""")
