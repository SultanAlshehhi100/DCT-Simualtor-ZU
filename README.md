# Flight → Hotel Guest Simulator (Abu Dhabi)

## Run it
    pip install -r requirements.txt
    streamlit run app.py

The `data/` folder is already built, so it runs straight away.

## Rebuild the data (only if a source workbook changes)
    python prep_data.py  predictions_international_test.xlsx  flight_data_working.xlsx  domestic_test_working.xlsx  domestic_train_working.xlsx

## Files
- `app.py`        the Streamlit screens
- `engine.py`     the calculation chain (seats → passengers → P2P → check-ins → guest nights)
- `prep_data.py`  turns the four team workbooks into the CSVs in `data/`
- `data/`         intl_monthly, routes, market_link, domestic_monthly

## Two modes
1. **Change an existing route**: pick month / country / city / airline, then change frequency (%), aircraft size (%), load factor, transfer share, check-ins per P2P passenger, length of stay.
2. **Open a new route / add flights**: pick a source market (any of 45, including the 12 with no direct flights), flights per week, aircraft, load factor, transfer share.

With nothing changed the impact is exactly zero.
