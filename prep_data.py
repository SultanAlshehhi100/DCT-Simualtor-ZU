"""
Step 1 of the simulator: turn the four team workbooks into small, clean CSVs in ./data

Usage (run from this folder):
    python prep_data.py  intl_predictions.xlsx  flight_data_working.xlsx  domestic_test_working.xlsx  domestic_train_working.xlsx

The four inputs, in that order:
  1. predictions_international_test.xlsx   (daily international Guests, Aug 2025 - Feb 2026, from the model)
  2. flight_data_working.xlsx              (sheets Flight_By_Route and Flight_By_Country)
  3. domestic_test_working.xlsx            (daily domestic 'Predicted Guests')
  4. domestic_train_working.xlsx           (daily domestic history, Jan 2022 - Jul 2025)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if len(sys.argv) != 5:
    sys.exit(__doc__)
intl_f, flight_f, dom_test_f, dom_train_f = sys.argv[1:5]
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- international (hotel side)
intl = pd.read_excel(intl_f)
intl["Date"] = pd.to_datetime(intl["Date"])
intl["Month"] = intl["Date"].dt.strftime("%Y-%m")
intl["Same-Day Guests"] = intl["Same-Day Guests"].fillna(0)
intl["New Arrivals"] = intl["New Arrivals"].fillna(0)

intl_m = (intl.groupby(["Month", "Nationality"])
              .agg(Guest_Nights=("Guests", "sum"),
                   Check_ins=("New Arrivals", "sum"))
              .reset_index())
# Average length of stay = guest nights / check-ins
intl_m["Avg_Stay"] = intl_m["Guest_Nights"] / intl_m["Check_ins"].replace(0, np.nan)
MONTHS = sorted(intl_m["Month"].unique())

# ---------------------------------------------------------------- flights
routes = pd.read_excel(flight_f, sheet_name="Flight_By_Route")
country = pd.read_excel(flight_f, sheet_name="Flight_By_Country")
routes = routes[routes["Month"].isin(MONTHS)].copy()
country = country[country["Month"].isin(MONTHS)].copy()

routes = routes.rename(columns={"Departure Country": "Country", "Departure City": "City",
                                "Total P2P": "P2P", "Total PAX": "PAX", "Total Seats": "Seats",
                                "Total Transfer": "Transfer", "Total Transit": "Transit"})
routes["Load_Factor"] = routes["PAX"] / routes["Seats"].replace(0, np.nan)
routes = routes[["Month", "Country", "City", "Airline", "Seats", "PAX", "P2P",
                 "Transfer", "Transit", "Load_Factor"]]

country = country.rename(columns={"Departure Country": "Country", "Total P2P": "P2P",
                                  "Total PAX": "PAX", "Total Seats": "Seats",
                                  "Total Transfer": "Transfer", "Total Transit": "Transit"})
country["Nationality"] = country["Country"].str.upper()
country = country[["Month", "Country", "Nationality", "P2P", "PAX", "Seats", "Transfer", "Transit"]]

# ---------------------------------------------------------------- link: capture rate
# capture rate = hotel check-ins / point-to-point passengers (same country name, upper-cased)
link = intl_m.merge(country, on=["Month", "Nationality"], how="left")
link["Capture"] = link["Check_ins"] / link["P2P"].replace(0, np.nan)
link["Has_Direct_Flight"] = link["P2P"].notna() & (link["P2P"] > 0)
link = link[["Month", "Nationality", "Country", "Guest_Nights", "Check_ins", "Avg_Stay",
             "P2P", "PAX", "Seats", "Transfer", "Transit", "Capture", "Has_Direct_Flight"]]

# ---------------------------------------------------------------- domestic
dom_test = pd.read_excel(dom_test_f)
dom_test["Date"] = pd.to_datetime(dom_test["Date"])
dom_test = dom_test.dropna(subset=["Predicted Guests"])
dom_f = (dom_test.assign(Month=dom_test["Date"].dt.strftime("%Y-%m"))
                 .groupby("Month")["Predicted Guests"].sum().rename("Guest_Nights").reset_index())
dom_f["Type"] = "Forecast"

dom_train = pd.read_excel(dom_train_f)
dom_train["Date"] = pd.to_datetime(dom_train["Date"])
dom_h = (dom_train.assign(Month=dom_train["Date"].dt.strftime("%Y-%m"))
                  .groupby("Month")["Guests"].sum().rename("Guest_Nights").reset_index())
dom_h["Type"] = "Actual"
domestic = pd.concat([dom_h, dom_f], ignore_index=True)

# ---------------------------------------------------------------- save + report
intl_m.to_csv(OUT / "intl_monthly.csv", index=False)
routes.to_csv(OUT / "routes.csv", index=False)
link.to_csv(OUT / "market_link.csv", index=False)
domestic.to_csv(OUT / "domestic_monthly.csv", index=False)

print("months:", MONTHS)
print("intl_monthly", intl_m.shape, "| routes", routes.shape, "| market_link", link.shape, "| domestic", domestic.shape)
print("nationalities with no direct flight:", sorted(link.loc[~link["Has_Direct_Flight"], "Nationality"].unique()))
print("capture rate (check-ins / P2P):", link["Capture"].describe().round(3).to_dict())
print("international guest nights:", int(intl_m["Guest_Nights"].sum()),
      "| domestic forecast guest nights:", int(dom_f["Guest_Nights"].sum()))
