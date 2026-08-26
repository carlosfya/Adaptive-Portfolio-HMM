"""Exp. 20 — Sensibilidad al día de rebalanceo y correlación vol-macro (§5.6).

Dos preguntas del consejo de revisión:

1. El motor blando fotografía las probabilidades filtradas UN día al mes
   (el primero hábil). ¿Depende el resultado de esa foto? Se repite el
   backtest rebalanceando el día hábil 1, 4, 8, 12 y 16 de cada mes
   (offset 0/3/7/11/15) con los MISMOS regímenes cacheados.
2. La mezcla asume independencia entre los dos HMM. ¿Cómo de correladas
   están sus probabilidades filtradas en la práctica? Se reporta la
   correlación incondicional y la condicionada a estrés (vol_p1 > 0.5).
"""

import pandas as pd
from common import baseline_regimes, load_returns, save_table, strategy_metrics, variant_config

OFFSETS = [0, 3, 7, 11, 15]  # día hábil del mes: 1º, 4º, 8º, 12º, 16º


def main() -> None:
    rets = load_returns()
    regimes = baseline_regimes()
    regimes = regimes.loc[regimes.index.isin(rets.index)]

    rows = {}
    for off in OFFSETS:
        cfg = variant_config({"allocation": {"rebalance_offset_days": off}})
        name = "base (día hábil 1)" if off == 0 else f"día hábil {off + 1}"
        rows[name] = strategy_metrics(regimes, cfg)
    save_table(pd.DataFrame(rows).T, "exp20_dia_rebalanceo")

    # Correlación empírica entre las probabilidades de los dos HMM
    corr_all = regimes["vol_p1"].corr(regimes["macro_p1"])
    stress = regimes[regimes["vol_p1"] > 0.5]
    corr_stress = stress["vol_p1"].corr(stress["macro_p1"])
    resumen = pd.DataFrame({"valor": {
        "corr(vol_p1, macro_p1) incondicional": corr_all,
        "corr(vol_p1, macro_p1) | vol_p1 > 0.5": corr_stress,
        "% días con vol_p1 > 0.5": len(stress) / len(regimes),
    }})
    save_table(resumen, "exp20_corr_probabilidades")


if __name__ == "__main__":
    main()
