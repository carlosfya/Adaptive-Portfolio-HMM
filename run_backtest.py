"""Paso 5: backtest causal con costes y comparación con benchmarks."""

import pandas as pd

from src.backtest.benchmarks import sixty_forty, spy_buy_hold, static_inverse_vol
from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics, metrics_table
from src.config.settings import DATA_DIR, OUTPUT_DIR, load_config
from src.data.fetcher import fetch_fred


def load_rf(cfg: dict) -> pd.Series:
    """T-bill 3M (FRED, % anual) cacheado en data/rf.parquet."""
    path = DATA_DIR / "rf.parquet"
    serie = cfg["data"].get("rf_series", "DGS3MO")
    if not path.exists():
        fetch_fred([serie], cfg["data"]["start"]).to_parquet(path)
    return pd.read_parquet(path)[serie]


def main() -> None:
    cfg = load_config()
    prices = pd.read_parquet(DATA_DIR / "prices.parquet").dropna()
    rets = prices.pct_change().dropna()
    regimes = pd.read_parquet(OUTPUT_DIR / "regimes.parquet")
    regimes = regimes.loc[regimes.index.isin(rets.index)]
    rf = load_rf(cfg)

    mode = cfg["allocation"].get("mode", "hard")
    print(f"Backtest ({mode}): {regimes.index[0].date()} -> {regimes.index[-1].date()}")
    bt = run_backtest(rets, regimes, cfg)
    dates = bt["returns"].index
    cost_bps = cfg["backtest"]["cost_bps"]
    span = cfg["allocation"]["ewma_span"]

    # El benchmark estático usa EXACTAMENTE los activos del sistema (la
    # unión de los universos por régimen, 7 ETFs), no el universo de datos
    # descargado (que incluye LQD/HYG como candidatos de ablación).
    system_assets = sorted(
        {t for u in cfg["allocation"]["regime_universes"].values() for t in u}
    )
    results = {
        f"Sistema (HMM dual, {mode})": compute_metrics(bt["returns"], bt["turnover"], rf),
        "60/40 SPY/TLT": compute_metrics(sixty_forty(rets, dates, cost_bps), rf=rf),
        "SPY B&H": compute_metrics(spy_buy_hold(rets, dates), rf=rf),
        "Estática inv-vol (7 ETFs)": compute_metrics(
            static_inverse_vol(rets[system_assets], dates, cost_bps, span), rf=rf
        ),
    }
    table = metrics_table(results)
    print("\n" + table.round(3).to_string())

    table.to_csv(OUTPUT_DIR / "metrics.csv")
    bt["returns"].to_frame().to_parquet(OUTPUT_DIR / "strategy_returns.parquet")
    bt["weights"].to_parquet(OUTPUT_DIR / "weights.parquet")
    bt["regimes"].to_frame().to_parquet(OUTPUT_DIR / "active_regimes.parquet")
    print("\nOK: resultados guardados en output/")


if __name__ == "__main__":
    main()
