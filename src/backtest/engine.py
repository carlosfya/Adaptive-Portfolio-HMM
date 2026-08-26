"""Motor de backtest causal con costes de fricción y drift de pesos.

Ejecución diferida: los pesos objetivo de cada día t se deciden con
información hasta t-1 (regímenes causales + retornos hasta t-1) y se
aplican al retorno de t. El coste (cost_bps sobre el turnover
Σ|w_t - w_{t-1}|) se imputa al día del trade.

Entre rebalanceos la cartera NO se retoca: los pesos derivan con los
precios (drift), igual que en el benchmark 60/40 (comparación simétrica).
Sólo hay turnover cuando el asignador emite un objetivo nuevo (rebalanceo
mensual o, en modo hard, cambio de régimen consolidado); el turnover se
mide entonces contra los pesos ya derivados. El despliegue inicial tiene
turnover ≈ 1.

`cfg["backtest"]["delay_days"]` (por defecto 0) añade días extra de
retraso entre la señal y la ejecución: con delay_days=1 los pesos de t se
deciden con información hasta t-2 (test de realismo de ejecución, §5.6).
"""

import pandas as pd

from src.allocation.allocator import RegimeAllocator, SoftRegimeAllocator


def run_backtest(
    rets: pd.DataFrame,
    regimes: pd.DataFrame | pd.Series,
    cfg: dict,
) -> dict:
    """Ejecuta el backtest diario del sistema completo.

    El motor lo decide `cfg["allocation"]["mode"]`:
    - "soft" (sistema final): mezcla blanda causal con las probabilidades
      filtradas (`vol_p1`, `macro_p1`) del walk-forward.
    - "hard": switch discreto de universo con anti-whipsaw (ablación).

    Parameters
    ----------
    rets : pd.DataFrame
        Retornos diarios de todo el universo (todas las fechas disponibles).
    regimes : pd.DataFrame or pd.Series
        Salida out-of-sample del walk-forward: DataFrame con columnas
        `regime`, `vol_p1`, `macro_p1` (una Series de regímenes también
        vale para el modo hard).
    cfg : dict
        Configuración global.

    Returns
    -------
    dict
        `returns` (serie neta), `weights` (DataFrame, pesos vigentes tras
        drift), `turnover` (serie), `regimes` (régimen dominante por día,
        para atribución).
    """
    cost = cfg["backtest"]["cost_bps"] / 1e4
    delay = cfg["backtest"].get("delay_days", 0)
    soft = cfg["allocation"].get("mode", "hard") == "soft"
    if isinstance(regimes, pd.Series):
        regimes = regimes.to_frame("regime")
    if delay:
        # La señal de t pasa a ser la de t-delay: ejecución retrasada
        regimes = regimes.shift(delay).iloc[delay:]

    if soft:
        alloc = SoftRegimeAllocator(cfg, list(rets.columns))
    else:
        alloc = RegimeAllocator(cfg, list(rets.columns))

    dates = regimes.index
    # Posición de cada fecha en rets (evita búsquedas por etiqueta O(n)/día)
    pos = rets.index.get_indexer(dates)
    if (pos < 0).any():
        missing = dates[pos < 0]
        raise ValueError(
            f"{len(missing)} fechas de regímenes no existen en los retornos "
            f"(p.ej. {missing[0].date()}); filtra regimes con isin(rets.index)."
        )

    weights, turnovers, port_rets, active = [], [], [], []
    prev_w = pd.Series(0.0, index=rets.columns)  # pesos ya derivados (t-1)

    for date, i in zip(dates, pos):
        # Historial de retornos estrictamente hasta t-1 (causalidad)
        hist = rets.iloc[max(0, i - 400):i]
        row = regimes.loc[date]
        if soft:
            target = alloc.step(date, row["vol_p1"], row["macro_p1"], hist)
            active.append(row["regime"])  # régimen dominante (atribución)
        else:
            target = alloc.step(date, row["regime"], hist)
            active.append(alloc.active_regime)

        # Sólo se opera cuando el asignador rebalancea (flag explícito); el
        # resto de días la cartera deriva y el turnover es 0.
        w = target if alloc.rebalanced else prev_w

        turnover = (w - prev_w).abs().sum()
        r = rets.iloc[i]
        port_rets.append((w * r).sum() - cost * turnover)
        weights.append(w)
        turnovers.append(turnover)

        # Drift: los pesos derivan con los retornos del día
        grown = w * (1 + r)
        total = grown.sum()
        prev_w = grown / total if total > 0 else w

    return {
        "returns": pd.Series(port_rets, index=dates, name="strategy"),
        "weights": pd.DataFrame(weights, index=dates),
        "turnover": pd.Series(turnovers, index=dates, name="turnover"),
        "regimes": pd.Series(active, index=dates, name="active_regime"),
    }
