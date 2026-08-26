"""Exp. 2 — Ablación de señales (§5.2) sobre la arquitectura final.

La base es el conjunto mínimo (vol: log_vix+rv21; macro: z-scores). Cada
variante AÑADE o CAMBIA un bloque y re-corre el walk-forward completo:
- + correlación SPY-TLT (la hipótesis original del bloque de vol)
- + estructura temporal del VIX (vix_ts)
- macro en niveles en vez de z-scores
- macro con niveles Y z-scores (4 features)
- configuración inicial del trabajo (vol 3 + macro 4), como referencia
"""

import pandas as pd
from common import EXP_DIR, baseline_regimes, save_table, strategy_metrics, variant_config

VARIANTS = {
    "+ corr SPY-TLT": "exp20_mas_corr",
    "+ vix_ts": "exp14_vix_ts",
    "macro en niveles": "exp20_macro_niveles",
    "macro niveles+z": "exp20_macro_ambos",
    "config. inicial (vol3+macro4)": "exp13_proba",
}


def main() -> None:
    cfg = variant_config()
    rows = {"base (mínima)": strategy_metrics(baseline_regimes(), cfg)}
    for name, tag in VARIANTS.items():
        regimes = pd.read_parquet(EXP_DIR / f"regimes_{tag}.parquet")
        rows[name] = strategy_metrics(regimes, cfg)
    save_table(pd.DataFrame(rows).T, "exp02_ablacion_features")


if __name__ == "__main__":
    main()
