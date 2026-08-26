"""Exp. 1 — Validación económica de los regímenes (§6.1).

¿Los 4 regímenes se corresponden con episodios reales (2020, 2022...)?
Estadísticas de SPY por régimen, matriz de transición y regímenes
dominantes en episodios conocidos.
"""

import numpy as np
import pandas as pd
from common import baseline_regimes, load_returns, save_table

EPISODES = {
    "COVID crash (feb-abr 2020)": ("2020-02-20", "2020-04-30"),
    "Bear 2022 (ene-oct 2022)": ("2022-01-01", "2022-10-31"),
    "Bull 2013-2014": ("2013-06-01", "2014-12-31"),
    "Vol-shock ago 2015": ("2015-08-15", "2015-10-15"),
    "Q4 2018": ("2018-10-01", "2018-12-31"),
}


def main() -> None:
    regimes = baseline_regimes()["regime"]
    rets = load_returns()
    common = regimes.index.intersection(rets.index)
    regimes, spy = regimes.loc[common], rets.loc[common, "SPY"]

    # --- Estadísticas de SPY condicionadas al régimen (mismo día) ---
    stats = spy.groupby(regimes).agg(["mean", "std", "count"])
    stats["ret_anual"] = stats["mean"] * 252
    stats["vol_anual"] = stats["std"] * np.sqrt(252)
    stats["frecuencia"] = stats["count"] / len(regimes)
    save_table(stats[["ret_anual", "vol_anual", "frecuencia"]], "exp01_spy_por_regimen")

    # --- Matriz de transición diaria entre regímenes ---
    trans = pd.crosstab(regimes.shift(), regimes, normalize="index")
    save_table(trans, "exp01_matriz_transicion")

    # --- Régimen dominante en episodios históricos conocidos ---
    rows = {}
    for name, (a, b) in EPISODES.items():
        seg = regimes.loc[a:b]
        if len(seg):
            rows[name] = seg.value_counts(normalize=True).round(2).to_dict()
    save_table(pd.DataFrame(rows).T.fillna(0.0), "exp01_episodios")


if __name__ == "__main__":
    main()
