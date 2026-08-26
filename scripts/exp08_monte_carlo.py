"""Exp. 8 — Monte Carlo por block-bootstrap (§5.7).

Bootstrap de bloques (21 días) de los retornos diarios CONJUNTOS de
estrategia y benchmarks (se remuestrean los mismos días para preservar la
dependencia transversal). Reporta P(Sharpe>0), P(Sharpe>SPY), P(Sharpe>60/40)
y el intervalo de confianza del Sharpe. Honestidad: si con SPY hay empate
estadístico, se dice.
"""

import numpy as np
import pandas as pd
from common import load_returns, save_table, variant_config

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest.benchmarks import sixty_forty, spy_buy_hold  # noqa: E402
from src.config.settings import OUTPUT_DIR  # noqa: E402

N_BOOT = 5000
BLOCK = 21


def sharpe(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=-1) / x.std(axis=-1) * np.sqrt(252)


def main() -> None:
    cfg = variant_config()
    rng = np.random.default_rng(cfg["seed"])
    strat = pd.read_parquet(OUTPUT_DIR / "strategy_returns.parquet")["strategy"]
    rets = load_returns()
    spy = spy_buy_hold(rets, strat.index).to_numpy()
    b6040 = sixty_forty(rets, strat.index, cfg["backtest"]["cost_bps"]).to_numpy()
    s = strat.to_numpy()

    n = len(s)
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(N_BOOT, -1)[:, :n]

    sh_s, sh_spy, sh_64 = sharpe(s[idx]), sharpe(spy[idx]), sharpe(b6040[idx])
    df = pd.DataFrame({
        "P(Sharpe>0)": [(sh_s > 0).mean()],
        "P(Sharpe>SPY)": [(sh_s > sh_spy).mean()],
        "P(Sharpe>60/40)": [(sh_s > sh_64).mean()],
        "Sharpe IC5%": [np.percentile(sh_s, 5)],
        "Sharpe mediana": [np.median(sh_s)],
        "Sharpe IC95%": [np.percentile(sh_s, 95)],
    }, index=["sistema"])
    save_table(df, "exp08_monte_carlo")


if __name__ == "__main__":
    main()
