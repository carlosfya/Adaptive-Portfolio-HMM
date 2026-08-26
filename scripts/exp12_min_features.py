"""Exp. 12 (exploratorio) — Features mínimas y combinación con suavizado.

Compara el conjunto mínimo (vol: log_vix+rv21; macro: sólo z-scores) con
la base y con las variantes de la ablación, y prueba la combinación
features mínimas + transición suave + persistencia 5 (los tres hallazgos
juntos).
"""

import pandas as pd
from common import EXP_DIR, baseline_regimes, load_returns, save_table, strategy_metrics, variant_config
from exp11_smooth_transitions import smoothed_backtest


def main() -> None:
    rets = load_returns()
    rows = {}

    # Experimento de la era del switch duro: fija mode=hard explícitamente
    # (la config base pasó a soft cuando la mezcla blanda se hizo final).
    base_cfg = variant_config({"allocation": {"mode": "hard"}})
    rows["base"] = strategy_metrics(baseline_regimes()["regime"], base_cfg)

    minf = pd.read_parquet(EXP_DIR / "regimes_exp12_min_features.parquet")["regime"]
    minf = minf.loc[minf.index.isin(rets.index)]
    rows["features mínimas"] = strategy_metrics(minf, base_cfg)

    zonly = pd.read_parquet(EXP_DIR / "regimes_exp02_macro_solo_z-scores.parquet")["regime"]
    zonly = zonly.loc[zonly.index.isin(rets.index)]
    rows["macro z-only (ablación)"] = strategy_metrics(zonly, base_cfg)

    # Combinaciones: suavizado y persistencia sobre las features mínimas
    rows["mínimas + suave a=0.1"] = smoothed_backtest(0.1, minf, rets, base_cfg)
    cfg_p5 = variant_config(
        {"allocation": {"mode": "hard", "persistence_days": 5}}
    )
    rows["mínimas + persistencia 5"] = strategy_metrics(minf, cfg_p5)
    rows["mínimas + p5 + suave a=0.1"] = smoothed_backtest(0.1, minf, rets, cfg_p5)
    # Y la misma combinación sobre la base (control)
    rows["base + p5 + suave a=0.1"] = smoothed_backtest(
        0.1, baseline_regimes()["regime"].loc[lambda s: s.index.isin(rets.index)],
        rets, cfg_p5)

    save_table(pd.DataFrame(rows).T, "exp12_features_minimas")


if __name__ == "__main__":
    main()
