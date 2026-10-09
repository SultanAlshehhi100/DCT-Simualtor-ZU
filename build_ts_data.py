"""
Build data/daily_guests.csv: one row per day, international + domestic guests staying in Abu Dhabi hotels.

    python build_ts_data.py  data_international_train.xlsx  predictions_international_test.xlsx  domestic_train_working.xlsx  domestic_test_working.xlsx

Inputs, in order:
  1. international TRAIN (daily, by nationality, Jan 2022 - Jul 2025, has Guests)
  2. international predictions (daily, by nationality, Aug 2025 - Feb 2026, our model's Guests)
  3. domestic TRAIN (daily, Jan 2022 - Jul 2025)
  4. domestic test with 'Predicted Guests' (daily, Aug 2025 - Feb 2026)
"""
import sys
from pathlib import Path

import pandas as pd

if len(sys.argv) != 5:
    sys.exit(__doc__)
intl_tr, intl_pr, dom_tr, dom_te = sys.argv[1:5]
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

i_act = pd.read_excel(intl_tr).groupby("Date")["Guests"].sum()
i_fc = pd.read_excel(intl_pr).groupby("Date")["Guests"].sum()
d_act = pd.read_excel(dom_tr).set_index("Date")["Guests"]
d_fc = pd.read_excel(dom_te).set_index("Date")["Predicted Guests"]

actual = pd.DataFrame({"International": i_act, "Domestic": d_act}).dropna()
actual["Source"] = "Actual"
fc = pd.DataFrame({"International": i_fc, "Domestic": d_fc}).dropna()
fc["Source"] = "Forecast"

daily = pd.concat([actual, fc]).sort_index()
daily.index = pd.to_datetime(daily.index)
daily.index.name = "Date"

full = pd.date_range(daily.index.min(), daily.index.max())
assert len(full) == len(daily), "missing days in the daily series"
daily[["International", "Domestic"]] = daily[["International", "Domestic"]].round().astype(int)
daily.reset_index().to_csv(OUT / "daily_guests.csv", index=False)

print(daily.groupby("Source").agg(days=("International", "size"), first=("International", lambda s: s.index.min().date()),
                                   last=("International", lambda s: s.index.max().date())))
print(daily.groupby("Source")[["International", "Domestic"]].mean().round(0))
