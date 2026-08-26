"""Exp. 16 (exploratorio) — Variantes de motor sobre el campeón blando.

Base: asignación blanda causal con mezcla mensual sobre las probabilidades
del modelo de features mínimas (exp12). Variantes:

- rebalanceo semanal / mensual de la mezcla.
- suavizado asimétrico: rápido hacia defensa, lento hacia riesgo
  (los crashes son rápidos; las recuperaciones, lentas).
- banda de no-negociación: no operar si el cambio total de pesos < banda.
- vol-targeting 10% integrado (escala la mezcla, resto a SHY).
- recorte de LQD del universo (candidato de la ablación exp07).
"""

import numpy as np
import pandas as pd
from common import EXP_DIR, load_returns, save_table, variant_config
from exp13_soft_allocation import inverse_vol, regime_probs

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest.metrics import compute_metrics  # noqa: E402

DEFENSIVE = ["TLT", "SHY", "GLD", "XLP", "LQD"]


def soft_engine(probs, rets, cfg, rebal="M", alpha=1.0, alpha_def=None,
                band=0.0, vol_target=None, drop=()):
    """Backtest blando generalizado (todas las señales con info <= t-1)."""
    acfg = cfg["allocation"]
    cost = cfg["backtest"]["cost_bps"] / 1e4
    universes = {r: [t for t in u if t not in drop]
                 for r, u in acfg["regime_universes"].items()}
    span, tickers = acfg["ewma_span"], list(rets.columns)
    prev = pd.Series(0.0, index=tickers)
    target = prev
    last_key = None
    out_r, out_t = [], []
    strat_hist = []  # retornos propios para el vol-targeting
    for date in probs.index:
        key = {"M": (date.year, date.month),
               "W": (date.isocalendar().year, date.isocalendar().week),
               "D": date}[rebal]
        if key != last_key:
            hist = rets.loc[:date].iloc[-400:-1]
            p = regime_probs(probs.loc[date, "vol_p1"],
                             probs.loc[date, "macro_p1"])
            target = sum(p[r] * inverse_vol(u, hist, span, tickers)
                         for r, u in universes.items())
            if vol_target is not None and len(strat_hist) > 63:
                rv = pd.Series(strat_hist).ewm(span=63).std().iloc[-1] * np.sqrt(252)
                lev = min(1.0, vol_target / rv) if rv > 0 else 1.0
                target = target * lev
                target["SHY"] += 1.0 - lev
        last_key = key
        a = alpha
        if alpha_def is not None:
            # ¿El objetivo es más defensivo que la cartera actual?
            if target[DEFENSIVE].sum() > prev[DEFENSIVE].sum():
                a = alpha_def
        w = prev + a * (target - prev)
        turnover = (w - prev).abs().sum()
        if turnover < band:
            w, turnover = prev, 0.0
        out_r.append((w * rets.loc[date]).sum() - cost * turnover)
        out_t.append(turnover)
        strat_hist.append(out_r[-1])
        prev = w
    return compute_metrics(pd.Series(out_r, index=probs.index),
                           pd.Series(out_t, index=probs.index))


def main() -> None:
    cfg = variant_config()
    rets = load_returns()
    probs = pd.read_parquet(EXP_DIR / "regimes_exp12_min_features.parquet")
    probs = probs.loc[probs.index.isin(rets.index), ["vol_p1", "macro_p1"]]

    rows = {
        "campeón (mensual)": soft_engine(probs, rets, cfg),
        "semanal": soft_engine(probs, rets, cfg, rebal="W"),
        "diaria + banda 5%": soft_engine(probs, rets, cfg, rebal="D", band=0.05),
        "mensual asimétrica (def a=1, riesgo a=0.2)": soft_engine(
            probs, rets, cfg, rebal="D", alpha=0.2, alpha_def=1.0),
        "mensual + vol-target 10%": soft_engine(probs, rets, cfg,
                                                vol_target=0.10),
        "mensual sin LQD": soft_engine(probs, rets, cfg, drop=("LQD",)),
        "semanal sin LQD": soft_engine(probs, rets, cfg, rebal="W",
                                       drop=("LQD",)),
    }
    save_table(pd.DataFrame(rows).T, "exp16_motor_blando")


if __name__ == "__main__":
    main()
