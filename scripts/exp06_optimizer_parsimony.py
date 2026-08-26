"""Exp. 6 — Parsimonia del motor de asignación (§5.6.1).

Con los MISMOS regímenes/probabilidades cacheados, comparar:
- el ponderador dentro de la mezcla blanda (inverse-vol vs equal-weight);
- el motor duro (switch discreto + anti-whipsaw), con sus tres ponderadores;
- el control sin switch de universo (todos los activos siempre).

Conclusión esperada: el ponderador es intercambiable; el valor está en el
switch (blando) de universo.
"""

import pandas as pd
from common import baseline_regimes, save_table, strategy_metrics, variant_config

ALL_TICKERS_UNIVERSE = None  # se rellena en main


def main() -> None:
    regimes = baseline_regimes()
    rows = {}

    # Motor blando (sistema final) con cada ponderador
    for w in ["inverse_vol", "equal_weight", "risk_parity"]:
        cfg = variant_config({"allocation": {"weighting": w}})
        rows[f"blando + {w}"] = strategy_metrics(regimes, cfg)

    # Motor duro (ablación de motor) con cada ponderador
    for w in ["inverse_vol", "equal_weight", "risk_parity"]:
        cfg = variant_config({"allocation": {"mode": "hard", "weighting": w}})
        rows[f"duro + {w}"] = strategy_metrics(regimes, cfg)

    # Control: sin switch de universo (cartera de todos los activos siempre)
    base = variant_config()
    todos = sorted({t for u in base["allocation"]["regime_universes"].values()
                    for t in u})
    cfg = variant_config({"allocation": {"regime_universes": {
        r: todos for r in base["allocation"]["regime_universes"]}}})
    rows["sin switch (todos los activos)"] = strategy_metrics(regimes, cfg)

    save_table(pd.DataFrame(rows).T, "exp06_parsimonia_optimizador")


if __name__ == "__main__":
    main()
