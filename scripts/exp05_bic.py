"""Exp. 5 — Selección del número de estados por BIC.

Enseña honestamente que el BIC puede preferir K>2 y que los 2 estados se
mantienen POR DISEÑO: interpretabilidad (cada HMM responde una pregunta
económica), orquestación 2x2 y anti-overfitting.
"""

import numpy as np
import pandas as pd
from common import load_features, save_table, variant_config
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.regimes.walk_forward_hmm import fit_hmm  # noqa: E402


def n_params(k: int, d: int) -> int:
    """Parámetros libres de un GaussianHMM full: inicio + transición + medias + cov."""
    return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2


def main() -> None:
    cfg = variant_config()
    feats = load_features()
    t0 = cfg["walk_forward"]["train_min_days"]
    rows = {}
    for name, cols in [("vol", cfg["features"]["vol_block"]),
                       ("macro", cfg["features"]["macro_block"])]:
        X = StandardScaler().fit_transform(feats[cols].iloc[:t0])
        for k in range(1, 6):
            hmm_cfg = dict(cfg["hmm"], n_states=k)
            # Mejor log-verosimilitud de 5 inits (BIC del mejor ajuste)
            ll = max(fit_hmm(X, hmm_cfg, cfg["seed"] + i).score(X) for i in range(5))
            bic = n_params(k, X.shape[1]) * np.log(len(X)) - 2 * ll
            rows[(name, k)] = {"loglik": ll, "n_params": n_params(k, X.shape[1]),
                               "BIC": bic}
    df = pd.DataFrame(rows).T
    df.index.names = ["HMM", "K"]
    save_table(df, "exp05_bic")


if __name__ == "__main__":
    main()
