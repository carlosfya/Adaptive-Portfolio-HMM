"""Exp. 4 — Estabilidad de semilla (el experimento estrella metodológico).

Sobre la arquitectura final: con UNA inicialización EM el resultado depende
de la semilla; con el ensemble de 10 y voto mayoritario, no.
"""

import pandas as pd
from common import EXP_DIR, baseline_regimes, save_table, strategy_metrics, variant_config

SINGLE_SEEDS = list(range(10))
ENSEMBLE_SEEDS = [42, 7, 123]


def main() -> None:
    rows = {}
    for s in SINGLE_SEEDS:
        cfg = variant_config({"hmm": {"n_init": 1}, "seed": s})
        regimes = pd.read_parquet(EXP_DIR / f"regimes_exp22_single_seed{s}.parquet")
        rows[f"n_init=1, seed={s}"] = strategy_metrics(regimes, cfg)
    for s in ENSEMBLE_SEEDS:
        cfg = variant_config({"seed": s})
        if s == 42:
            regimes = baseline_regimes()
        else:
            regimes = pd.read_parquet(EXP_DIR / f"regimes_exp22_ensemble_seed{s}.parquet")
        rows[f"ensemble(10), seed={s}"] = strategy_metrics(regimes, cfg)

    df = pd.DataFrame(rows).T
    save_table(df, "exp04_estabilidad_semilla")
    single = df.iloc[: len(SINGLE_SEEDS)]["Sharpe"]
    ens = df.iloc[len(SINGLE_SEEDS):]["Sharpe"]
    print(f"\nSharpe n_init=1:  min={single.min():.3f} max={single.max():.3f} "
          f"rango={single.max() - single.min():.3f}")
    print(f"Sharpe ensemble:  min={ens.min():.3f} max={ens.max():.3f} "
          f"rango={ens.max() - ens.min():.3f}")


if __name__ == "__main__":
    main()
