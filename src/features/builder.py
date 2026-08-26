"""Construcción de features CAUSALES para los dos HMM.

Regla de oro: shift(1) en TODAS las features. El régimen de t se decide
sólo con información disponible hasta t-1. Las series macro de FRED se
publican con retraso, así que además del shift se hace forward-fill al
calendario bursátil (el último dato conocido, nunca el futuro).
"""

import numpy as np
import pandas as pd


def build_features(
    prices: pd.DataFrame,
    vix: pd.Series,
    fred: pd.DataFrame,
    cfg: dict,
    vix3m: pd.Series | None = None,
) -> pd.DataFrame:
    """Construye la matriz de features causales para ambos HMM.

    Parameters
    ----------
    prices : pd.DataFrame
        Cierres ajustados del universo (incluye SPY y TLT).
    vix : pd.Series
        Nivel de cierre del VIX.
    fred : pd.DataFrame
        Series macro diarias (T10Y2Y, BAA10Y).
    cfg : dict
        Configuración global (bloques y ventanas en cfg['features']).

    Returns
    -------
    pd.DataFrame
        Features ya desplazadas con shift(1), sin NaN, en días de mercado.
    """
    fcfg = cfg["features"]
    rets = prices.pct_change()
    idx = prices.index

    feats = pd.DataFrame(index=idx)

    # --- Bloque de volatilidad (HMM de mercado) ---
    feats["log_vix"] = np.log(vix.reindex(idx).ffill())
    feats["rv21_spy"] = rets["SPY"].rolling(fcfg["rv_window"]).std() * np.sqrt(252)
    feats["corr_spy_tlt"] = rets["SPY"].rolling(fcfg["corr_window"]).corr(rets["TLT"])
    if vix3m is not None:
        # Estructura temporal del miedo: >0 (backwardation) = estrés agudo
        feats["vix_ts"] = np.log(
            vix.reindex(idx).ffill() / vix3m.reindex(idx).ffill()
        )

    # --- Bloque macro (HMM de ciclo, datos FRED) ---
    macro = fred.reindex(idx).ffill()
    feats["t10y2y"] = macro["T10Y2Y"]
    feats["baa10y"] = macro["BAA10Y"]
    zw = fcfg["z_window"]
    for col in ["t10y2y", "baa10y"]:
        mu = feats[col].rolling(zw).mean()
        sd = feats[col].rolling(zw).std()
        feats[f"{col}_z"] = (feats[col] - mu) / sd

    # --- Robustez al retraso de publicación de FRED (test §5.6): con
    # macro_shift_extra=1 el bloque macro se decide con info hasta t-2 ---
    extra = fcfg.get("macro_shift_extra", 0)
    if extra:
        macro_cols = ["t10y2y", "baa10y", "t10y2y_z", "baa10y_z"]
        feats[macro_cols] = feats[macro_cols].shift(extra)

    # --- Causalidad: TODO se desplaza un día ---
    feats = feats.shift(1)

    return feats.dropna()
