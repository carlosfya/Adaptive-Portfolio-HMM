"""Exp. 18 — PSR y Deflated Sharpe Ratio del sistema final (§5.7).

Disciplina de multiple testing llevada a número (Bailey y López de Prado,
2014): el funnel probó decenas de configuraciones, así que el Sharpe
reportado debe descontarse contra el máximo que producirían N trials por
puro azar. Trials = variantes de CONFIGURACIÓN con métricas comparables:

- Se incluyen las tablas de ablación y variantes (exp02, exp03, exp06,
  exp07, exp09, exp11, exp12, exp13, exp13b, exp16, exp17).
- Se excluyen: exp01/exp10 (diagnósticos, no estrategias), exp04 (semillas:
  el ensemble elimina esa elección, no es selección de configuración) y
  exp08 (réplicas bootstrap, no trials).

Salida: PSR (P(SR>0)), SR0 (Sharpe esperado del mejor trial por azar) y
DSR = P(SR > SR0), con N y la dispersión del funnel.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EXP_DIR, save_table  # noqa: E402

from src.backtest.metrics import (  # noqa: E402
    deflated_sharpe,
    probabilistic_sharpe,
)
from src.config.settings import OUTPUT_DIR  # noqa: E402

TRIAL_TABLES = [
    "exp02_ablacion_features",
    "exp03_sensibilidad_ventana",
    "exp06_parsimonia_optimizador",
    "exp07_ablacion_activos",
    "exp09_overlays",
    "exp11_transiciones_suaves",
    "exp12_features_minimas",
    "exp13_asignacion_blanda",
    "exp13b_blanda_minimas",
    "exp16_motor_blando",
    "exp17_robustez_ejecucion",
    "exp19_sensibilidad_costes",
    "exp20_dia_rebalanceo",
]


def collect_trial_sharpes() -> pd.Series:
    """Sharpe de cada variante probada en el funnel (sin duplicados)."""
    frames = []
    for name in TRIAL_TABLES:
        path = EXP_DIR / f"{name}.csv"
        if not path.exists():
            print(f"  (aviso: falta {name}.csv, se omite)")
            continue
        df = pd.read_csv(path, index_col=0)
        if "Sharpe" in df.columns:
            s = df["Sharpe"].dropna()
            s.index = [f"{name}: {i}" for i in s.index]
            frames.append(s)
    all_sr = pd.concat(frames)
    # Las filas "base" se repiten en casi todas las tablas: una sola cuenta
    return all_sr[~all_sr.round(6).duplicated()]


def main() -> None:
    rets = pd.read_parquet(OUTPUT_DIR / "strategy_returns.parquet")["strategy"]
    trials = collect_trial_sharpes()
    n = len(trials)
    psr = probabilistic_sharpe(rets)
    dsr, sr0 = deflated_sharpe(rets, trials.tolist())

    resumen = pd.DataFrame(
        {
            "valor": {
                "N trials (funnel)": n,
                "Sharpe medio de los trials": trials.mean(),
                "Sharpe std de los trials": trials.std(),
                "Sharpe max de los trials": trials.max(),
                "PSR  P(SR>0)": psr,
                "SR0 (max esperado por azar, anual)": sr0,
                "DSR  P(SR>SR0)": dsr,
            }
        }
    )
    save_table(resumen, "exp18_dsr")
    trials.sort_values(ascending=False).to_csv(EXP_DIR / "exp18_trials.csv")
    print(f"\nTrials guardados en output/experiments/exp18_trials.csv ({n} filas)")


if __name__ == "__main__":
    main()
