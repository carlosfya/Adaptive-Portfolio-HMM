"""Paso 2: construcción de features causales (shift(1) en todo)."""

import pandas as pd

from src.config.settings import DATA_DIR, load_config
from src.features.builder import build_features


def main() -> None:
    cfg = load_config()
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    vix = pd.read_parquet(DATA_DIR / "vix.parquet")["VIX"]
    fred = pd.read_parquet(DATA_DIR / "fred.parquet")
    vix3m_path = DATA_DIR / "vix3m.parquet"
    vix3m = pd.read_parquet(vix3m_path)["VIX3M"] if vix3m_path.exists() else None

    # El universo completo exige que todos los ETFs existan (HYG desde 2007)
    n_before = len(prices)
    prices = prices.dropna()
    print(f"Filas eliminadas por NaN en algún ticker: {n_before - len(prices)} "
          f"(inicio efectivo: {prices.index[0].date()})")
    feats = build_features(prices, vix, fred, cfg, vix3m=vix3m)
    feats.to_parquet(DATA_DIR / "features.parquet")
    print(f"Features: {feats.shape} ({feats.index[0].date()} -> {feats.index[-1].date()})")
    print(feats.describe().T[["mean", "std", "min", "max"]].round(3))


if __name__ == "__main__":
    main()
