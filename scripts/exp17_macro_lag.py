"""Exp. 17 — Robustez de ejecución: lag de publicación macro y retraso señal->orden.

Dos tests que blindan el realismo operativo del sistema (§5.6):

1. `macro_shift_extra=1`: el bloque macro (FRED) se observa con un día
   hábil extra de retraso, cubriendo el lag de publicación de BAA10Y.
   Requiere reconstruir las features y repetir el walk-forward completo
   (cacheado en output/experiments/, no toca los ficheros canónicos).
2. `delay_days=1`: los pesos de t se deciden con información hasta t-2
   (ejecución retrasada un día). Reutiliza los regímenes base.
"""

import pandas as pd
from common import (
    EXP_DIR,
    baseline_regimes,
    load_returns,
    save_table,
    strategy_metrics,
    variant_config,
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import DATA_DIR  # noqa: E402
from src.features.builder import build_features  # noqa: E402
from src.regimes.walk_forward_hmm import walk_forward_regimes  # noqa: E402


def _regimes_macro_lag(cfg: dict, force: bool = False) -> pd.DataFrame:
    """Walk-forward con features macro retrasadas un día extra (cacheado)."""
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    path = EXP_DIR / "regimes_exp17_macro_lag.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    prices = pd.read_parquet(DATA_DIR / "prices.parquet").dropna()
    vix = pd.read_parquet(DATA_DIR / "vix.parquet")["VIX"]
    fred = pd.read_parquet(DATA_DIR / "fred.parquet")
    vix3m_path = DATA_DIR / "vix3m.parquet"
    vix3m = pd.read_parquet(vix3m_path)["VIX3M"] if vix3m_path.exists() else None
    feats = build_features(prices, vix, fred, cfg, vix3m=vix3m)
    print("[exp17_macro_lag] walk-forward...")
    regimes = walk_forward_regimes(feats, cfg)
    regimes.to_parquet(path)
    return regimes


def main() -> None:
    rows = {}
    base_cfg = variant_config()
    base = baseline_regimes()
    rows["base"] = strategy_metrics(base, base_cfg)

    # 1) Lag de publicación macro: features nuevas -> walk-forward nuevo
    lag_cfg = variant_config({"features": {"macro_shift_extra": 1}})
    rows["macro con lag extra (t-2)"] = strategy_metrics(
        _regimes_macro_lag(lag_cfg), lag_cfg
    )

    # 2) Ejecución retrasada un día: mismos regímenes, señal desplazada
    delay_cfg = variant_config({"backtest": {"delay_days": 1}})
    rows["ejecución con retraso (t-2)"] = strategy_metrics(base, delay_cfg)

    save_table(pd.DataFrame(rows).T, "exp17_robustez_ejecucion")


if __name__ == "__main__":
    main()
