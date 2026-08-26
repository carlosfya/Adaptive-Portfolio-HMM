"""HMM dual con ensemble y label switching, en validación walk-forward.

Tres decisiones metodológicas clave (los tres pilares):

1. Causalidad: el escalado se ajusta SÓLO con el tramo de entrenamiento de
   cada ventana, y el estado de cada día se obtiene por FILTRADO forward
   (probabilidad del estado dado el pasado), nunca por Viterbi/suavizado,
   que usarían información futura.
2. Ensemble: cada ventana entrena n_init=10 inicializaciones EM y el estado
   diario se decide por voto mayoritario -> elimina el artefacto de semilla.
3. Label switching: tras cada ajuste, los estados se reordenan de forma
   determinista por la media de una feature-ancla monótona (log_vix para el
   HMM de volatilidad, baa10y_z para el macro): estado 0 = valor bajo
   (calma / expansión), estado 1 = valor alto (estrés / contracción).
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

# Orquestación 2x2 -> 4 regímenes: (estado_vol, estado_macro)
REGIME_MAP = {
    (0, 0): "Growth",      # calma + expansión
    (1, 1): "Crash",       # estrés + contracción
    (1, 0): "Sideways_A",  # estrés de mercado en expansión macro
    (0, 1): "Sideways_B",  # calma de mercado con deterioro macro
}


def fit_hmm(X: np.ndarray, cfg_hmm: dict, seed: int) -> GaussianHMM:
    """Ajusta un GaussianHMM de 2 estados con una semilla concreta.

    Si `sticky > 0`, se añade un prior de Dirichlet sobre la diagonal de la
    matriz de transición: la persistencia de los regímenes se codifica en
    el modelo (los regímenes reales duran meses) en vez de filtrarse a
    posteriori.
    """
    k = cfg_hmm["n_states"]
    sticky = cfg_hmm.get("sticky", 0)
    model = GaussianHMM(
        n_components=k,
        covariance_type=cfg_hmm["covariance_type"],
        n_iter=cfg_hmm["n_iter"],
        random_state=seed,
        transmat_prior=1.0 + sticky * np.eye(k),
    )
    model.fit(X)
    return model


def anchor_permutation(model: GaussianHMM, anchor_idx: int) -> np.ndarray:
    """Permutación determinista de estados por la media de la feature-ancla.

    Devuelve un array `perm` tal que `perm[estado_crudo] = estado_canónico`,
    con estado 0 = media baja del ancla y estado 1 = media alta.
    """
    order = np.argsort(model.means_[:, anchor_idx])  # crudo ordenado asc.
    perm = np.empty_like(order)
    perm[order] = np.arange(len(order))
    return perm


def forward_filter_states(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Estados filtrados causalmente: argmax de P(s_t | x_1..x_t).

    La recursión forward sólo mira el pasado, así que el estado del día t
    no depende de datos posteriores (a diferencia de Viterbi/posterior).
    """
    framelogprob = model._compute_log_likelihood(X)
    log_A = np.log(model.transmat_ + 1e-300)
    alpha = np.zeros_like(framelogprob)
    alpha[0] = np.log(model.startprob_ + 1e-300) + framelogprob[0]
    for t in range(1, len(X)):
        alpha[t] = logsumexp(alpha[t - 1][:, None] + log_A, axis=0) + framelogprob[t]
    return alpha.argmax(axis=1)


def forward_filter_proba(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Probabilidades filtradas P(s_t | x_1..x_t) por estado (causales)."""
    framelogprob = model._compute_log_likelihood(X)
    log_A = np.log(model.transmat_ + 1e-300)
    alpha = np.zeros_like(framelogprob)
    alpha[0] = np.log(model.startprob_ + 1e-300) + framelogprob[0]
    for t in range(1, len(X)):
        alpha[t] = logsumexp(alpha[t - 1][:, None] + log_A, axis=0) + framelogprob[t]
    return np.exp(alpha - logsumexp(alpha, axis=1, keepdims=True))


def _window_states(
    X_train: np.ndarray,
    X_all: np.ndarray,
    block_slice: slice,
    cfg_hmm: dict,
    anchor_idx: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estados y probabilidad filtrada (estado 1) del bloque out-of-sample."""
    votes = []
    probas = []
    loglik = []
    n_skipped = 0
    for i in range(cfg_hmm["n_init"]):
        model = fit_hmm(X_train, cfg_hmm, seed + i)
        # Control de calidad: una init que no convergió es ruido con signo
        # arbitrario y no debe entrar en el promedio del ensemble.
        if not model.monitor_.converged:
            n_skipped += 1
            continue
        perm = anchor_permutation(model, anchor_idx)
        proba = forward_filter_proba(model, X_all)[block_slice]
        proba = proba[:, np.argsort(perm)]  # columnas en orden canónico
        votes.append(proba.argmax(axis=1))
        probas.append(proba[:, 1])
        loglik.append(model.score(X_train))
    if not votes:  # ninguna init convergió: mejor fallar que inventar
        raise RuntimeError(
            f"Ninguna de las {cfg_hmm['n_init']} inicializaciones convergió"
        )
    if n_skipped:
        print(f"    (aviso: {n_skipped} init(s) sin converger, excluidas)")
    votes, probas = np.array(votes), np.array(probas)
    if cfg_hmm["selection"] == "max_likelihood":
        best = int(np.argmax(loglik))
        return votes[best], probas[best]
    # Voto mayoritario por día (con 2 estados: media > 0.5) y prob. media.
    # Nota: en días frontera el voto puede discrepar de (prob. media > 0.5);
    # el motor soft consume SÓLO las probabilidades, así que la etiqueta
    # discreta se usa únicamente para atribución y ablación (modo hard).
    return (votes.mean(axis=0) > 0.5).astype(int), probas.mean(axis=0)


def walk_forward_regimes(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Ejecuta el walk-forward completo de los dos HMM.

    Para cada ventana: se ajusta el escalador y el ensemble SÓLO con el
    tramo de entrenamiento, y se filtran los estados del bloque siguiente
    (refit_days días) de forma causal.

    Returns
    -------
    pd.DataFrame
        Columnas `vol_state`, `macro_state`, `regime` en fechas out-of-sample.
    """
    wf, hmm_cfg, fcfg = cfg["walk_forward"], cfg["hmm"], cfg["features"]
    seed = cfg["seed"]
    blocks = {
        "vol": (fcfg["vol_block"], hmm_cfg["anchor_vol"]),
        "macro": (fcfg["macro_block"], hmm_cfg["anchor_macro"]),
    }
    n = len(features)
    t0 = wf["train_min_days"]
    refit = wf["refit_days"]
    rolling = wf["scheme"] == "rolling"

    results = {name: np.full(n, -1, dtype=int) for name in blocks}
    proba = {name: np.full(n, np.nan) for name in blocks}
    starts = list(range(t0, n, refit))
    for k, s in enumerate(starts):
        e = min(s + refit, n)
        train_lo = max(0, s - wf["train_min_days"]) if rolling else 0
        for name, (cols, anchor) in blocks.items():
            raw = features[cols].to_numpy()
            scaler = StandardScaler().fit(raw[train_lo:s])  # SÓLO train
            X = scaler.transform(raw[train_lo:e])
            s_rel, e_rel = s - train_lo, e - train_lo
            results[name][s:e], proba[name][s:e] = _window_states(
                X[:s_rel], X, slice(s_rel, e_rel), hmm_cfg,
                cols.index(anchor), seed + 1000 * k,
            )
        if (k + 1) % 20 == 0 or k == len(starts) - 1:
            print(f"  ventana {k + 1}/{len(starts)} ({features.index[s].date()})")

    out = pd.DataFrame(
        {"vol_state": results["vol"], "macro_state": results["macro"],
         "vol_p1": proba["vol"], "macro_p1": proba["macro"]},
        index=features.index,
    ).iloc[t0:]
    out["regime"] = [
        REGIME_MAP[(v, m)] for v, m in zip(out["vol_state"], out["macro_state"])
    ]
    return out
