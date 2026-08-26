"""Precomputa (y cachea) los regímenes walk-forward de UNA variante.

Uso: python precompute.py <tag>
Permite paralelizar los experimentos caros por variante; los scripts
exp02/exp03/exp04 encuentran después el caché y sólo hacen el backtest.
"""

import os
import sys

# Con HMM de 2 estados el multihilo BLAS sólo añade contención: 1 hilo por
# proceso y paralelismo ENTRE variantes (debe fijarse antes de importar numpy).
for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(var, "1")

from common import get_regimes, variant_config  # noqa: E402

# Registro tag -> overrides de config (debe coincidir con los exp scripts)
REGISTRY: dict[str, dict] = {
    # --- exp02: ablación de features ---
    "exp02_sin_corr_spy_tlt": {"features": {"vol_block": ["log_vix", "rv21_spy"]}},
    "exp02_sin_log_vix": {"features": {"vol_block": ["rv21_spy", "corr_spy_tlt"]},
                          "hmm": {"anchor_vol": "rv21_spy"}},
    "exp02_sin_rv21_spy": {"features": {"vol_block": ["log_vix", "corr_spy_tlt"]}},
    "exp02_macro_sin_z-scores": {"features": {"macro_block": ["t10y2y", "baa10y"]},
                                 "hmm": {"anchor_macro": "baa10y"}},
    "exp02_macro_solo_z-scores": {"features": {"macro_block": ["t10y2y_z", "baa10y_z"]}},
    # --- exp03: sensibilidad de ventana ---
    "exp03_rolling_750": {"walk_forward": {"scheme": "rolling", "train_min_days": 750}},
    "exp03_rolling_1000": {"walk_forward": {"scheme": "rolling", "train_min_days": 1000}},
    "exp03_rolling_1250": {"walk_forward": {"scheme": "rolling", "train_min_days": 1250}},
    # --- exp04: ensembles con otras semillas base ---
    "exp04_ensemble_seed7": {"seed": 7},
    "exp04_ensemble_seed123": {"seed": 123},
    # --- exploración de arquitectura (no entra en la memoria de momento) ---
    # Features mínimas: hipótesis de que los z-scores (cambios) llevan la
    # señal macro y el bloque de vol depurado basta con 2 features
    "exp12_min_features": {"features": {"vol_block": ["log_vix", "rv21_spy"],
                                        "macro_block": ["t10y2y_z", "baa10y_z"]}},
    # Base con probabilidades filtradas guardadas (para asignación blanda causal)
    "exp13_proba": {},
    # Estructura temporal del VIX como feature de estrés agudo
    "exp14_vix_ts": {"features": {"vol_block": ["log_vix", "rv21_spy", "vix_ts"],
                                  "macro_block": ["t10y2y_z", "baa10y_z"]}},
    # HMM pegajoso: persistencia codificada en el modelo (prior diagonal)
    "exp15_sticky": {"features": {"vol_block": ["log_vix", "rv21_spy"],
                                  "macro_block": ["t10y2y_z", "baa10y_z"]},
                     "hmm": {"sticky": 100}},
    # ==== RE-VALIDACIÓN sobre la arquitectura final (base = config.yaml) ====
    # Ablación de señales (exp20): añadir/cambiar features respecto a la base
    "exp20_mas_corr": {"features": {"vol_block": ["log_vix", "rv21_spy",
                                                  "corr_spy_tlt"]}},
    "exp20_macro_niveles": {"features": {"macro_block": ["t10y2y", "baa10y"]},
                            "hmm": {"anchor_macro": "baa10y"}},
    "exp20_macro_ambos": {"features": {"macro_block": ["t10y2y", "baa10y",
                                                       "t10y2y_z", "baa10y_z"]}},
    # Sensibilidad de ventana (exp21)
    "exp21_rolling_750": {"walk_forward": {"scheme": "rolling",
                                           "train_min_days": 750}},
    "exp21_rolling_1000": {"walk_forward": {"scheme": "rolling",
                                            "train_min_days": 1000}},
    "exp21_rolling_1250": {"walk_forward": {"scheme": "rolling",
                                            "train_min_days": 1250}},
    # Estabilidad de semilla (exp22)
    "exp22_ensemble_seed7": {"seed": 7},
    "exp22_ensemble_seed123": {"seed": 123},
}
for s in range(10):
    REGISTRY[f"exp22_single_seed{s}"] = {"hmm": {"n_init": 1}, "seed": s}
# exp04: variantes de una sola inicialización, semillas 0..9
for s in range(10):
    REGISTRY[f"exp04_single_seed{s}"] = {"hmm": {"n_init": 1}, "seed": s}


def main() -> None:
    tag = sys.argv[1]
    get_regimes(tag, variant_config(REGISTRY[tag]))
    print(f"[{tag}] OK")


if __name__ == "__main__":
    main()
