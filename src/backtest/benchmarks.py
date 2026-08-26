"""Benchmarks: SPY Buy & Hold, 60/40 (SPY/TLT) y cartera estática inverse-vol.

Todos los benchmarks con rebalanceo pagan los mismos costes de fricción
que el sistema y modelan el drift de pesos entre rebalanceos (comparación
justa). La cartera estática inverse-vol sobre el universo completo aísla
el valor del switch de universo por régimen: misma regla de ponderación,
mismos activos, sin timing.
"""

import pandas as pd


def spy_buy_hold(rets: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Retornos diarios de comprar y mantener SPY."""
    return rets.loc[dates, "SPY"].rename("SPY B&H")


def sixty_forty(
    rets: pd.DataFrame, dates: pd.DatetimeIndex, cost_bps: float
) -> pd.Series:
    """Cartera 60/40 SPY/TLT con rebalanceo mensual y costes.

    Entre rebalanceos los pesos derivan con los precios (drift), como en
    una cartera real.
    """
    cost = cost_bps / 1e4
    target = pd.Series({"SPY": 0.60, "TLT": 0.40})
    w = target.copy()
    last_month = None
    out = []
    first = True
    for date in dates:
        month = (date.year, date.month)
        turnover = 0.0
        if month != last_month:
            turnover = 1.0 if first else (target - w).abs().sum()
            w = target.copy()
            first = False
        last_month = month
        r = rets.loc[date, ["SPY", "TLT"]]
        out.append((w * r).sum() - cost * turnover)
        # Drift de los pesos con los retornos del día
        w = w * (1 + r)
        w = w / w.sum()
    return pd.Series(out, index=dates, name="60/40")


def static_inverse_vol(
    rets: pd.DataFrame, dates: pd.DatetimeIndex, cost_bps: float, span: int = 63
) -> pd.Series:
    """Cartera estática inverse-vol sobre las columnas de `rets`, sin regímenes.

    Rebalanceo mensual a w ∝ 1/σ (EWMA, mismo span que el sistema), drift
    entre rebalanceos y mismos costes. Pasando los activos del sistema (la
    unión de los universos por régimen) es el benchmark interno que separa
    'diversificar' de 'saber CUÁNDO estar en cuáles'.
    """
    cost = cost_bps / 1e4
    pos = rets.index.get_indexer(dates)
    w: pd.Series | None = None
    last_month = None
    out = []
    for date, i in zip(dates, pos):
        month = (date.year, date.month)
        turnover = 0.0
        if month != last_month:
            hist = rets.iloc[max(0, i - 400):i]
            vol = hist.ewm(span=span).std().iloc[-1]
            target = 1.0 / vol
            target = target / target.sum()
            turnover = 1.0 if w is None else (target - w).abs().sum()
            w = target
        last_month = month
        r = rets.iloc[i]
        out.append((w * r).sum() - cost * turnover)
        w = w * (1 + r)
        w = w / w.sum()
    return pd.Series(out, index=dates, name="Static inv-vol")
