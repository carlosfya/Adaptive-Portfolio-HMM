"""Exp. 3 — Sensibilidad a la ventana de entrenamiento (§5.3).

Expansiva (base) vs rolling de 750/1000/1250 días, bajo el motor blando.
Métricas sobre el periodo OoS común (el de la base).
"""

import pandas as pd
from common import EXP_DIR, baseline_regimes, save_table, strategy_metrics, variant_config

WINDOWS = {"rolling 750d": 750, "rolling 1000d": 1000, "rolling 1250d": 1250}


def main() -> None:
    base = baseline_regimes()
    start = base.index[0]  # periodo común = OoS de la expansiva
    rows = {"expansiva (base)": strategy_metrics(base, variant_config())}
    for name, days in WINDOWS.items():
        cfg = variant_config({"walk_forward": {"scheme": "rolling",
                                               "train_min_days": days}})
        regimes = pd.read_parquet(EXP_DIR / f"regimes_exp21_rolling_{days}.parquet")
        rows[name] = strategy_metrics(regimes, cfg, start=start)
    save_table(pd.DataFrame(rows).T, "exp03_sensibilidad_ventana")


if __name__ == "__main__":
    main()
