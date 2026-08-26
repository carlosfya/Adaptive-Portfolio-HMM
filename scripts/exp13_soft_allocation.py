"""Exp. 13 (exploratorio) — Asignación blanda CAUSAL por probabilidades.

La versión descartada en el intento anterior usaba probabilidades
suavizadas (miran el futuro). Aquí se reconstruye la idea con
probabilidades FILTRADAS (forward, sólo pasado), guardadas por el
walk-forward exp13_proba:

    w_t = sum_r P_t(r) * w_r(t)

donde P_t(Growth) = (1-pv)(1-pm), P_t(Crash) = pv*pm, etc. (pv, pm =
probabilidad filtrada de estrés/contracción decidida con info <= t-1) y
w_r son los pesos inverse-vol del universo de cada régimen.

Variantes:
- mezcla mensual: la mezcla se recalcula el 1er día hábil del mes.
- mezcla diaria suavizada: objetivo diario, avance alpha=0.2 hacia él.
- mezcla mensual "afilada": p^2 renormalizada (más decidida, menos mezcla).
"""

import numpy as np
import pandas as pd
from common import EXP_DIR, load_returns, save_table, variant_config

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest.metrics import compute_metrics  # noqa: E402


def regime_probs(pv: float, pm: float, sharpen: float = 1.0) -> dict:
    """Probabilidades de los 4 regímenes a partir de las de estado 1."""
    p = {"Growth": (1 - pv) * (1 - pm), "Crash": pv * pm,
         "Sideways_A": pv * (1 - pm), "Sideways_B": (1 - pv) * pm}
    if sharpen != 1.0:
        raw = {k: v ** sharpen for k, v in p.items()}
        tot = sum(raw.values())
        p = {k: v / tot for k, v in raw.items()}
    return p


def inverse_vol(universe: list[str], hist: pd.DataFrame, span: int,
                tickers: list[str]) -> pd.Series:
    w = pd.Series(0.0, index=tickers)
    vol = hist[universe].ewm(span=span).std().iloc[-1]
    if vol.isna().any() or (vol <= 0).any():
        w[universe] = 1.0 / len(universe)
        return w
    inv = 1.0 / vol
    w[universe] = inv / inv.sum()
    return w


def soft_backtest(probs: pd.DataFrame, rets: pd.DataFrame, cfg: dict,
                  monthly: bool, alpha: float = 1.0,
                  sharpen: float = 1.0) -> dict:
    acfg = cfg["allocation"]
    cost = cfg["backtest"]["cost_bps"] / 1e4
    universes, span = acfg["regime_universes"], acfg["ewma_span"]
    tickers = list(rets.columns)
    prev = pd.Series(0.0, index=tickers)
    last_month = None
    target = prev
    out_r, out_t = [], []
    for date in probs.index:
        month = (date.year, date.month)
        if (not monthly) or month != last_month:
            hist = rets.loc[:date].iloc[-400:-1]  # hasta t-1 (acotado)
            p = regime_probs(probs.loc[date, "vol_p1"],
                             probs.loc[date, "macro_p1"], sharpen)
            target = sum(p[r] * inverse_vol(u, hist, span, tickers)
                         for r, u in universes.items())
        last_month = month
        w = prev + alpha * (target - prev)
        turnover = (w - prev).abs().sum()
        out_r.append((w * rets.loc[date]).sum() - cost * turnover)
        out_t.append(turnover)
        prev = w
    return compute_metrics(pd.Series(out_r, index=probs.index),
                           pd.Series(out_t, index=probs.index))


def main() -> None:
    cfg = variant_config()
    rets = load_returns()
    probs = pd.read_parquet(EXP_DIR / "regimes_exp13_proba.parquet")
    probs = probs.loc[probs.index.isin(rets.index), ["vol_p1", "macro_p1"]]

    rows = {
        "mezcla mensual": soft_backtest(probs, rets, cfg, monthly=True),
        "mezcla diaria suavizada (a=0.2)": soft_backtest(
            probs, rets, cfg, monthly=False, alpha=0.2),
        "mezcla mensual afilada (p^2)": soft_backtest(
            probs, rets, cfg, monthly=True, sharpen=2.0),
    }
    save_table(pd.DataFrame(rows).T, "exp13_asignacion_blanda")


if __name__ == "__main__":
    main()
