"""Exp. 9 — Overlays como extensión CONFIRMATORIA (§5.6.4, corto).

Dos reglas pre-definidas sobre el sistema (no son el plato principal):
- Veto MA200: si SPY(t-1) < MA200(t-1), la parte de renta variable
  (SPY/QQQ/DBC) se traslada a SHY.
- Vol-targeting: exposición escalada por min(1, 10% / vol EWMA realizada
  de la propia estrategia hasta t-1); el resto a SHY.
Ambas usan sólo información hasta t-1 (causales) y pagan el turnover extra.
"""

import numpy as np
import pandas as pd
from common import load_returns, save_table, variant_config

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest.metrics import compute_metrics  # noqa: E402
from src.config.settings import OUTPUT_DIR  # noqa: E402

EQUITY = ["SPY", "QQQ", "DBC"]
TARGET_VOL = 0.10


def net_returns(weights: pd.DataFrame, rets: pd.DataFrame, cost: float) -> pd.Series:
    turn = weights.diff().abs().sum(axis=1)
    turn.iloc[0] = weights.iloc[0].abs().sum()
    return (weights * rets.loc[weights.index]).sum(axis=1) - cost * turn


def main() -> None:
    cfg = variant_config()
    cost = cfg["backtest"]["cost_bps"] / 1e4
    rets = load_returns()
    w = pd.read_parquet(OUTPUT_DIR / "weights.parquet")
    base = pd.read_parquet(OUTPUT_DIR / "strategy_returns.parquet")["strategy"]

    # --- Veto MA200 (señal con precios hasta t-1) ---
    prices_spy = (1 + rets["SPY"]).cumprod()
    below = (prices_spy < prices_spy.rolling(200).mean()).shift(1).reindex(w.index)
    below = below.fillna(False).astype(bool)
    w_veto = w.copy()
    eq = w_veto.loc[below, EQUITY].sum(axis=1)
    w_veto.loc[below, EQUITY] = 0.0
    w_veto.loc[below, "SHY"] += eq

    # --- Vol-targeting sobre la estrategia (vol EWMA hasta t-1) ---
    realized = base.ewm(span=63).std().shift(1) * np.sqrt(252)
    lev = (TARGET_VOL / realized).clip(upper=1.0).fillna(1.0)
    w_vt = w.mul(lev, axis=0)
    w_vt["SHY"] += 1.0 - lev

    rows = {
        "sistema base": compute_metrics(base),
        "+ veto MA200": compute_metrics(net_returns(w_veto, rets, cost)),
        "+ vol-targeting 10%": compute_metrics(net_returns(w_vt, rets, cost)),
    }
    save_table(pd.DataFrame(rows).T, "exp09_overlays")


if __name__ == "__main__":
    main()
