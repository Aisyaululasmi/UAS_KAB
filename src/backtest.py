import pandas as pd


def timeseries_split_years(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, one in df.groupby("Ticker"):
        one = one.sort_values("Date")
        years = sorted(pd.to_datetime(one["Date"]).dt.year.unique())
        for year in years[-3:]:
            valid = one[pd.to_datetime(one["Date"]).dt.year == year]
            train = one[pd.to_datetime(one["Date"]).dt.year < year]
            if len(train) > 300 and len(valid) > 20:
                rows.append({"ticker": ticker, "train_until": year - 1, "valid_year": year, "valid_rows": len(valid)})
    return pd.DataFrame(rows)
