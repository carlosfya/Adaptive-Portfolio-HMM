"""Utilidades comunes de los experimentos de validación (§6).

Cada experimento es un script independiente que reutiliza estas funciones:
- `variant_config`: config base + overrides (para no tocar config.yaml).
- `get_regimes`: walk-forward de una variante, cacheado en output/experiments/.
- `strategy_metrics`: backtest + métricas de una serie de regímenes.

Disciplina de multiple testing: TODAS las variantes probadas se guardan y
reportan; nunca se borra una fila porque salga mal.
"""

import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import run_backtest  # noqa: E402
from src.backtest.metrics import compute_metrics  # noqa: E402
from src.config.settings import DATA_DIR, OUTPUT_DIR, load_config  # noqa: E402
from src.regimes.walk_forward_hmm import walk_forward_regimes  # noqa: E402

EXP_DIR = OUTPUT_DIR / "experiments"


def deep_update(base: dict, overrides: dict) -> dict:
    """Fusión recursiva de diccionarios (overrides gana)."""
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def variant_config(overrides: dict | None = None) -> dict:
    """Config base con overrides aplicados."""
    return deep_update(load_config(), overrides or {})


def load_returns() -> pd.DataFrame:
    """Retornos diarios del universo completo (precios cacheados)."""
    prices = pd.read_parquet(DATA_DIR / "prices.parquet").dropna()
    return prices.pct_change().dropna()


def load_features() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "features.parquet")


def get_regimes(tag: str, cfg: dict, force: bool = False) -> pd.DataFrame:
    """Walk-forward de una variante, con caché por etiqueta de experimento."""
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    path = EXP_DIR / f"regimes_{tag}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    print(f"[{tag}] walk-forward...")
    regimes = walk_forward_regimes(load_features(), cfg)
    regimes.to_parquet(path)
    return regimes


def strategy_metrics(
    regimes: pd.Series | pd.DataFrame, cfg: dict,
    start: pd.Timestamp | None = None
) -> dict:
    """Backtest de la estrategia y métricas (opcionalmente desde `start`).

    `regimes` puede ser la Series de regímenes (modo hard) o el DataFrame
    completo del walk-forward con probabilidades (necesario en modo soft).
    """
    rets = load_returns()
    regimes = regimes.loc[regimes.index.isin(rets.index)]
    if start is not None:
        regimes = regimes.loc[regimes.index >= start]
    bt = run_backtest(rets, regimes, cfg)
    return compute_metrics(bt["returns"], bt["turnover"])


def baseline_regimes() -> pd.DataFrame:
    """Regímenes del sistema base (caché principal de run_walk_forward)."""
    return pd.read_parquet(OUTPUT_DIR / "regimes.parquet")


def save_table(df: pd.DataFrame, name: str) -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXP_DIR / f"{name}.csv")
    print(f"\n{df.round(3).to_string()}\nGuardado: output/experiments/{name}.csv")
