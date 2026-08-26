"""Exp. 7 — Ablación de activos (§5.6.2), leave-one-out por universo.

Quitar cada ETF de todos los universos donde aparece y medir el impacto
bajo el motor blando final. Un activo cuyo borrado no daña es candidato a
recorte (así salió LQD del sistema).
"""

import pandas as pd
from common import baseline_regimes, save_table, strategy_metrics, variant_config


def main() -> None:
    base_cfg = variant_config()
    regimes = baseline_regimes()
    universes = base_cfg["allocation"]["regime_universes"]
    tickers = sorted({t for u in universes.values() for t in u})

    rows = {"base (todos)": strategy_metrics(regimes, base_cfg)}
    for t in tickers:
        pruned = {r: [x for x in u if x != t] or u for r, u in universes.items()}
        cfg = variant_config({"allocation": {"regime_universes": pruned}})
        rows[f"sin {t}"] = strategy_metrics(regimes, cfg)
    # LQD ya no está en el sistema: medir también su RE-inclusión (control)
    plus_lqd = {r: u + (["LQD"] if r in ("Sideways_A", "Sideways_B") else [])
                for r, u in universes.items()}
    cfg = variant_config({"allocation": {"regime_universes": plus_lqd}})
    rows["re-añadiendo LQD"] = strategy_metrics(regimes, cfg)
    save_table(pd.DataFrame(rows).T, "exp07_ablacion_activos")


if __name__ == "__main__":
    main()
