"""Atribución de rentabilidad y riesgo por régimen activo (§5.5).

Desglosa los retornos netos de la estrategia según el régimen que la
cartera tenía activo cada día (tras el anti-whipsaw).
"""

import numpy as np
import pandas as pd
from common import save_table

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import OUTPUT_DIR  # noqa: E402


def main() -> None:
    strat = pd.read_parquet(OUTPUT_DIR / "strategy_returns.parquet")["strategy"]
    active = pd.read_parquet(OUTPUT_DIR / "active_regimes.parquet")["active_regime"]
    g = strat.groupby(active)
    out = pd.DataFrame({
        "dias": g.count(),
        "frecuencia": g.count() / len(strat),
        "ret_anual": g.mean() * 252,
        "vol_anual": g.std() * np.sqrt(252),
        "contrib_total": g.sum() / strat.sum(),  # aportación al retorno acumulado
    })
    out["sharpe_condicional"] = out["ret_anual"] / out["vol_anual"]
    save_table(out, "exp10_atribucion_regimen")


if __name__ == "__main__":
    main()
