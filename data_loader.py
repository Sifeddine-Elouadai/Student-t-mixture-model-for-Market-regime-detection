# data_loader.py
import pandas as pd
import yfinance as yf


def load_sp500(start="2000-01-01", end=None):
    """
    Load S&P 500 historical daily close prices from Yahoo Finance.
    Works whether yfinance returns a multi-index or not.
    """
    df = yf.download("^GSPC", start=start, end=end)

    # Check if 'Adj Close' exists
    if "Adj Close" in df.columns:
        return df["Adj Close"]
    elif ("Adj Close", "") in df.columns:
        # Sometimes yfinance returns multiindex
        return df[("Adj Close", "")]
    else:
        # fallback: use 'Close' if 'Adj Close' missing
        return df["Close"]
