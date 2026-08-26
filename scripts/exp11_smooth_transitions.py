"""Exp. 11 (exploratorio) — ¿Conviene cambiar la cartera de forma suave?

Dos familias de variantes sobre los MISMOS regímenes cacheados:

1. Suavizado exponencial de pesos: en vez de saltar al objetivo, cada día
   se avanza una fracción alpha hacia él: w_t = w_{t-1} + alpha*(objetivo - w_{t-1}).
   Es causal (el objetivo del día t se decide con info hasta t-1) y reduce
   el impacto de los cambios bruscos de la figura 5.6.
   alpha = 1 reproduce el sistema base.
2. Persistencia anti-whipsaw de 1 / 3 (base) / 5 / 10 días.

Lógica económica: suavizar reduce costes de whipsaw y el riesgo de operar
sobre una señal aún dudosa, a cambio de llegar tarde a la protección en un
crash. El experimento mide ese trade-off.
"""

import pandas as pd
from common import baseline_regimes, load_returns, save_table, strategy_metrics, variant_config

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.allocation.allocator import RegimeAllocator  # noqa: E402
from src.backtest.metrics import compute_metrics  # noqa: E402

ALPHAS = [1.0, 0.5, 0.33, 0.2, 0.1]
PERSISTENCE = [1, 3, 5, 10]


def smoothed_backtest(alpha: float, regimes: pd.Series, rets: pd.DataFrame,
                      cfg: dict) -> dict:
    """Backtest con pesos suavizados exponencialmente hacia el objetivo."""
    cost = cfg["backtest"]["cost_bps"] / 1e4
    alloc = RegimeAllocator(cfg, list(rets.columns))
    prev = pd.Series(0.0, index=rets.columns)
    out_r, out_t = [], []
    for date in regimes.index:
        hist = rets.loc[:date].iloc[:-1]
        target = alloc.step(date, regimes.loc[date], hist)
        w = prev + alpha * (target - prev)
        turnover = (w - prev).abs().sum()
        out_r.append((w * rets.loc[date]).sum() - cost * turnover)
        out_t.append(turnover)
        prev = w
    return compute_metrics(pd.Series(out_r, index=regimes.index),
                           pd.Series(out_t, index=regimes.index))


def main() -> None:
    # Experimento de la era del switch duro: fija mode=hard explícitamente
    # (la config base pasó a soft cuando la mezcla blanda se hizo final).
    cfg = variant_config({"allocation": {"mode": "hard"}})
    rets = load_returns()
    regimes = baseline_regimes()["regime"]
    regimes = regimes.loc[regimes.index.isin(rets.index)]

    rows = {}
    for a in ALPHAS:
        name = "base (salto directo)" if a == 1.0 else f"suavizado alpha={a}"
        rows[name] = smoothed_backtest(a, regimes, rets, cfg)
    for p in PERSISTENCE:
        c = variant_config({"allocation": {"mode": "hard",
                                           "persistence_days": p}})
        rows[f"persistencia {p} días"] = strategy_metrics(regimes, c)
    save_table(pd.DataFrame(rows).T, "exp11_transiciones_suaves")


if __name__ == "__main__":
    main()
