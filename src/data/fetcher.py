"""Descarga de datos: precios OHLCV (yfinance) y series macro (FRED).

FRED se consulta vía el endpoint CSV público (sin API key):
https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIE
"""

import io

import pandas as pd
import requests
import yfinance as yf

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}"


def fetch_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Descarga precios de cierre ajustados diarios para una lista de ETFs.

    Parameters
    ----------
    tickers : list of str
        Símbolos de yfinance.
    start, end : str
        Rango de fechas (end=None -> hasta hoy).

    Returns
    -------
    pd.DataFrame
        Cierres ajustados, una columna por ticker, índice de fechas.
    """
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.Series):  # un solo ticker
        close = close.to_frame(tickers[0])
    return close[tickers].sort_index()


def fetch_vix(vix_ticker: str, start: str, end: str | None = None,
              name: str = "VIX") -> pd.Series:
    """Descarga el nivel de cierre de un índice de volatilidad (VIX/VIX3M)."""
    df = yf.download(vix_ticker, start=start, end=end, auto_adjust=True, progress=False)
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.name = name
    return s.sort_index()


def fetch_fred(series: list[str], start: str) -> pd.DataFrame:
    """Descarga series diarias de FRED (curva de tipos, spread de crédito).

    Returns
    -------
    pd.DataFrame
        Una columna por serie, índice de fechas, valores numéricos (NaN si '.').
    """
    frames = []
    for serie in series:
        resp = requests.get(FRED_URL.format(serie=serie), timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = ["date", serie]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df[serie] = pd.to_numeric(df[serie], errors="coerce")
        frames.append(df)
    out = pd.concat(frames, axis=1).sort_index()
    return out.loc[out.index >= pd.Timestamp(start)]
