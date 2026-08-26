"""Paso 1: descarga de datos (yfinance + FRED) y cacheo en parquet."""

from src.config.settings import DATA_DIR, ensure_dirs, load_config
from src.data.fetcher import fetch_fred, fetch_prices, fetch_vix


def main() -> None:
    cfg = load_config()
    ensure_dirs()
    d = cfg["data"]

    print("Descargando precios ETF...")
    prices = fetch_prices(d["tickers"], d["start"], d["end"])
    prices.to_parquet(DATA_DIR / "prices.parquet")
    print(f"  {prices.shape[0]} días, {prices.shape[1]} activos "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})")

    print("Descargando VIX...")
    vix = fetch_vix(d["vix_ticker"], d["start"], d["end"])
    vix.to_frame().to_parquet(DATA_DIR / "vix.parquet")

    print("Descargando series FRED...")
    fred = fetch_fred(d["fred_series"], d["start"])
    fred.to_parquet(DATA_DIR / "fred.parquet")
    print(f"  {list(fred.columns)}: {fred.dropna().index[-1].date()} último dato")
    print("OK: datos cacheados en output/data/")


if __name__ == "__main__":
    main()
