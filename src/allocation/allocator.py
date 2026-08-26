"""Asignador defensivo por régimen: universo por régimen + inverse-vol.

El valor del sistema está en el SWITCH DE UNIVERSO por régimen; la
ponderación es UNA sola regla (inverse-volatility con vol EWMA) idéntica en
los cuatro regímenes. Long-only, pesos suman 1, fallback equal-weight.

Dos motores comparten esa base:

- `SoftRegimeAllocator` (sistema final): mezcla blanda CAUSAL. Cada primer
  día hábil del mes, w = sum_r P(r) * w_r, donde P(r) son las
  probabilidades FILTRADAS de los dos HMM (sólo pasado) y w_r los pesos
  inverse-vol del universo del régimen r. La incertidumbre del modelo se
  traslada a la cartera: la transición entre regímenes es suave por
  construcción y no necesita filtros anti-whipsaw.
- `RegimeAllocator` (switch duro): salta al universo del régimen votado con
  persistencia anti-whipsaw. Se conserva como ablación del motor.

Ambos son *stateful* (pesos vigentes, último mes rebalanceado) y deben
invocarse en orden cronológico estricto.
"""

import numpy as np
import pandas as pd


def _erc_weights(rets: pd.DataFrame, span: int, n_iter: int = 200) -> pd.Series:
    """Risk parity clásico (ERC) con covarianza EWMA, algoritmo cíclico.

    Sólo se usa como alternativa documentada en la ablación de método
    (§5.6.1); el sistema final usa inverse-vol.
    """
    cov = rets.ewm(span=span).cov().iloc[-len(rets.columns):].to_numpy()
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(n_iter):
        rc = w * (cov @ w)  # contribución al riesgo de cada activo
        if (rc <= 0).any():
            break
        w = w * np.sqrt(rc.mean() / rc)
        w = w / w.sum()
    return pd.Series(w, index=rets.columns)


def universe_weights(
    universe: list[str],
    rets_hist: pd.DataFrame,
    tickers: list[str],
    span: int,
    weighting: str = "inverse_vol",
) -> pd.Series:
    """Pesos del ponderador único dentro de un universo (por defecto 1/σ EWMA).

    Función compartida por los dos motores (soft y hard). Devuelve un vector
    sobre el universo completo `tickers` con ceros fuera de `universe`.
    """
    w = pd.Series(0.0, index=tickers)
    if weighting == "equal_weight":
        w[universe] = 1.0 / len(universe)
        return w
    vol = rets_hist[universe].ewm(span=span).std().iloc[-1]
    if vol.isna().any() or (vol <= 0).any():  # fallback equal-weight
        w[universe] = 1.0 / len(universe)
        return w
    if weighting == "risk_parity":  # sólo para la ablación §5.6.1
        w[universe] = _erc_weights(rets_hist[universe], span)
        return w
    inv = 1.0 / vol
    w[universe] = inv / inv.sum()
    return w


class RegimeAllocator:
    """Asignador por régimen con anti-whipsaw y rebalanceo mensual.

    Parameters
    ----------
    cfg : dict
        Configuración global (usa el bloque `allocation`).
    tickers : list of str
        Universo completo de activos (columnas del vector de pesos).
    """

    def __init__(self, cfg: dict, tickers: list[str]):
        acfg = cfg["allocation"]
        self.universes = acfg["regime_universes"]
        self.span = acfg["ewma_span"]
        self.persistence = acfg["persistence_days"]
        # Ponderador ÚNICO del sistema (inverse_vol). Las alternativas
        # (equal_weight, risk_parity) existen SÓLO para la ablación §5.6.1.
        self.weighting = acfg.get("weighting", "inverse_vol")
        self.tickers = list(tickers)
        self.reset()

    def reset(self) -> None:
        """Reinicia todo el estado interno."""
        self.active_regime: str | None = None
        self.candidate: str | None = None
        self.candidate_days = 0
        self.weights = pd.Series(0.0, index=self.tickers)
        self.last_month: tuple[int, int] | None = None
        # True si el ÚLTIMO step() recalculó pesos (el motor debe operar)
        self.rebalanced = False

    def _inverse_vol_weights(
        self, universe: list[str], rets_hist: pd.DataFrame
    ) -> pd.Series:
        """Pesos según el ponderador configurado (por defecto w ∝ 1/σ EWMA)."""
        return universe_weights(
            universe, rets_hist, self.tickers, self.span, self.weighting
        )

    def step(
        self, date: pd.Timestamp, regime: str, rets_hist: pd.DataFrame
    ) -> pd.Series:
        """Devuelve los pesos vigentes para el día `date`.

        Parameters
        ----------
        date : pd.Timestamp
            Día actual t (los pesos se aplican al retorno de t).
        regime : str
            Régimen decidido con información hasta t-1.
        rets_hist : pd.DataFrame
            Retornos diarios del universo HASTA t-1 inclusive (nunca t).

        Returns
        -------
        pd.Series
            Vector de pesos (suma 1) sobre el universo completo.
        """
        self.rebalanced = False
        # --- Anti-whipsaw: el cambio debe persistir `persistence` días ---
        if self.active_regime is None:
            self.active_regime = regime
        elif regime != self.active_regime:
            if regime == self.candidate:
                self.candidate_days += 1
            else:
                self.candidate, self.candidate_days = regime, 1
            if self.candidate_days >= self.persistence:
                self.active_regime = regime
                self.candidate, self.candidate_days = None, 0
                # Cambio de régimen consolidado -> re-asignar cartera
                self.weights = self._inverse_vol_weights(
                    self.universes[self.active_regime], rets_hist
                )
                self.rebalanced = True
        else:
            self.candidate, self.candidate_days = None, 0

        # --- Rebalanceo el primer día hábil de cada mes ---
        month = (date.year, date.month)
        first_alloc = self.weights.sum() == 0
        if month != self.last_month or first_alloc:
            self.weights = self._inverse_vol_weights(
                self.universes[self.active_regime], rets_hist
            )
            self.rebalanced = True
        self.last_month = month

        return self.weights.copy()


def regime_probabilities(vol_p1: float, macro_p1: float) -> dict[str, float]:
    """Probabilidades de los 4 regímenes desde las de estado 1 de cada HMM.

    Los dos HMM son independientes por diseño, así que la probabilidad
    conjunta es el producto: P(Growth) = (1-pv)(1-pm), etc.
    """
    return {
        "Growth": (1 - vol_p1) * (1 - macro_p1),
        "Crash": vol_p1 * macro_p1,
        "Sideways_A": vol_p1 * (1 - macro_p1),
        "Sideways_B": (1 - vol_p1) * macro_p1,
    }


class SoftRegimeAllocator:
    """Mezcla blanda causal: w = sum_r P(r) * w_r, rebalanceada cada mes.

    Parameters
    ----------
    cfg : dict
        Configuración global (usa el bloque `allocation`).
    tickers : list of str
        Universo completo de activos (columnas del vector de pesos).
    """

    def __init__(self, cfg: dict, tickers: list[str]):
        acfg = cfg["allocation"]
        self.universes = acfg["regime_universes"]
        self.span = acfg["ewma_span"]
        self.weighting = acfg.get("weighting", "inverse_vol")
        # Día hábil del mes en que se rebalancea: 0 = primero (sistema);
        # >0 sólo para el test de sensibilidad al día de rebalanceo (§5.6).
        self.offset = acfg.get("rebalance_offset_days", 0)
        self.tickers = list(tickers)
        self.reset()

    def reset(self) -> None:
        """Reinicia todo el estado interno."""
        self.weights = pd.Series(0.0, index=self.tickers)
        self.last_month: tuple[int, int] | None = None
        self._day_in_month = 0
        # True si el ÚLTIMO step() recalculó pesos (el motor debe operar)
        self.rebalanced = False

    def _universe_weights(self, universe, rets_hist) -> pd.Series:
        """Pesos del ponderador único dentro de un universo (inverse-vol)."""
        return universe_weights(
            universe, rets_hist, self.tickers, self.span, self.weighting
        )

    def step(self, date, vol_p1: float, macro_p1: float,
             rets_hist: pd.DataFrame) -> pd.Series:
        """Pesos vigentes para el día `date`.

        Parameters
        ----------
        date : pd.Timestamp
            Día actual t (los pesos se aplican al retorno de t).
        vol_p1, macro_p1 : float
            Probabilidades filtradas de estrés/contracción (info <= t-1).
        rets_hist : pd.DataFrame
            Retornos diarios del universo HASTA t-1 inclusive (nunca t).
        """
        month = (date.year, date.month)
        if month != self.last_month:
            self._day_in_month = 1
        else:
            self._day_in_month += 1
        self.last_month = month

        due = self._day_in_month == self.offset + 1
        if due or self.weights.sum() == 0:
            probs = regime_probabilities(vol_p1, macro_p1)
            self.weights = sum(
                probs[r] * self._universe_weights(u, rets_hist)
                for r, u in self.universes.items()
            )
            self.rebalanced = True
        else:
            self.rebalanced = False
        return self.weights.copy()
