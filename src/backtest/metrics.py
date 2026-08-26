"""Métricas de rendimiento y riesgo del backtest.

Además de las métricas clásicas, incluye:

- Sharpe en EXCESO de la tasa libre de riesgo (T-bill 3M, FRED DGS3MO):
  imprescindible en 2022-2026, cuando el cash rinde 4-5% y una cartera
  cargada de SHY captura ese carry sin riesgo.
- Sortino con la downside deviation estándar sqrt(E[min(r,0)^2]) (la
  versión con std de los retornos negativos infra-penaliza).
- Autocorrelación lag-1 y Sharpe anualizado con el ajuste de Lo (2002):
  con autocorrelación positiva, sqrt(252) sobre-anualiza.
- VaR y CVaR diarios al 95% (mandato defensivo).
- PSR y DSR (Bailey y López de Prado, 2014): probabilidad de que el
  Sharpe sea real dado el sesgo de selección del funnel de experimentos.
"""

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def _lo_annualized_sharpe(rets: pd.Series, n_lags: int = 10) -> float:
    """Sharpe anualizado con el factor de Lo (2002), robusto a autocorrelación.

    SR_anual = SR_diario * q / sqrt(q + 2 * sum_k (q - k) * rho_k), q = 252.
    Con rho_k = 0 el factor se reduce al sqrt(252) habitual.
    """
    sr_daily = rets.mean() / rets.std()
    q = TRADING_DAYS
    denom = q + 2 * sum((q - k) * rets.autocorr(k) for k in range(1, n_lags + 1))
    if denom <= 0:  # autocorrelación negativa extrema; sin ajuste fiable
        return np.nan
    return sr_daily * q / np.sqrt(denom)


def compute_metrics(
    rets: pd.Series,
    turnover: pd.Series | None = None,
    rf: pd.Series | None = None,
) -> dict:
    """Calcula las métricas estándar sobre una serie de retornos diarios.

    Parameters
    ----------
    rets : pd.Series
        Retornos diarios netos.
    turnover : pd.Series, optional
        Serie diaria de turnover (para el turnover anualizado).
    rf : pd.Series, optional
        Tipo libre de riesgo ANUAL en porcentaje (p.ej. FRED DGS3MO). Se
        alinea por fecha, se rellena hacia delante y se pasa a diario.

    Returns
    -------
    dict
        CAGR, Vol, MaxDD, Sharpe, Sharpe exceso (si hay rf), Sharpe Lo,
        AC(1), Sortino, Calmar, VaR/CVaR 95% y turnover anualizado.
    """
    equity = (1 + rets).cumprod()
    years = len(rets) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    dd = (equity / equity.cummax() - 1).min()
    sharpe = rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)
    # Downside deviation estándar: sqrt(E[min(r,0)^2]), no std(r<0)
    downside = np.sqrt((np.minimum(rets, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = rets.mean() * TRADING_DAYS / downside if downside > 0 else np.nan
    calmar = cagr / abs(dd) if dd != 0 else np.nan
    var95 = rets.quantile(0.05)
    cvar95 = rets[rets <= var95].mean()

    out = {
        "CAGR": cagr,
        "Vol": vol,
        "MaxDD": dd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "AC(1)": rets.autocorr(1),
        "Sharpe Lo": _lo_annualized_sharpe(rets),
        "VaR 95%": var95,
        "CVaR 95%": cvar95,
    }
    if rf is not None:
        rf_daily = rf.reindex(rets.index).ffill() / 100 / TRADING_DAYS
        excess = rets - rf_daily
        out["Sharpe exceso"] = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)
    if turnover is not None:
        out["Turnover anual"] = turnover.mean() * TRADING_DAYS
    return out


def probabilistic_sharpe(rets: pd.Series, sr_benchmark_daily: float = 0.0) -> float:
    """PSR de Bailey y López de Prado (2012): P(SR verdadero > benchmark).

    Trabaja en unidades DIARIAS y corrige por skew y curtosis de los
    retornos (colas gordas reducen la confianza en el Sharpe observado).
    """
    sr = rets.mean() / rets.std()
    n = len(rets)
    g3 = stats.skew(rets)
    g4 = stats.kurtosis(rets, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr_benchmark_daily) * np.sqrt(n - 1) / denom))


def deflated_sharpe(
    rets: pd.Series, trial_sharpes_annual: list[float]
) -> tuple[float, float]:
    """DSR (Bailey y López de Prado, 2014): PSR contra el máximo esperado
    de N trials independientes con la varianza observada en el funnel.

    Parameters
    ----------
    rets : pd.Series
        Retornos diarios de la configuración FINAL reportada.
    trial_sharpes_annual : list of float
        Sharpe anualizado de TODAS las variantes probadas (el funnel
        completo, incluida la final).

    Returns
    -------
    (dsr, sr0_annual) : tuple of float
        DSR = P(SR > SR0) y el umbral SR0 anualizado (el Sharpe que se
        esperaría del mejor trial por puro azar).
    """
    daily = np.asarray(trial_sharpes_annual, dtype=float) / np.sqrt(TRADING_DAYS)
    n_trials = len(daily)
    var_sr = daily.var(ddof=1)
    sr0 = np.sqrt(var_sr) * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )
    return probabilistic_sharpe(rets, sr0), float(sr0 * np.sqrt(TRADING_DAYS))


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """Tabla comparativa de métricas (una fila por estrategia)."""
    return pd.DataFrame(results).T
