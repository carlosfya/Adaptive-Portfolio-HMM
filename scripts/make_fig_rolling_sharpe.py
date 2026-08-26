"""Figura extra §5.4: Sharpe rolling a 3 años (¿es estable el 1,20?).

Mismas convenciones que make_figures.py: sistema en azul, benchmarks en
grises, etiquetas directas al final, un solo eje, rejilla recesiva.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_returns, variant_config  # noqa: E402

from figlang import T, fig_dir  # noqa: E402
from src.backtest.benchmarks import sixty_forty, spy_buy_hold  # noqa: E402
from src.config.settings import OUTPUT_DIR, ROOT  # noqa: E402

FIG_DIR = fig_dir(ROOT)
WINDOW = 3 * 252  # 3 años hábiles

C_SISTEMA = "#2a78d6"
C_6040 = "#52514e"
C_SPY = "#9e9d99"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
})


def rolling_sharpe(rets: pd.Series) -> pd.Series:
    mu = rets.rolling(WINDOW).mean()
    sd = rets.rolling(WINDOW).std()
    return (mu / sd * np.sqrt(252)).dropna()


def main() -> None:
    cfg = variant_config()
    rets = load_returns()
    strat = pd.read_parquet(OUTPUT_DIR / "strategy_returns.parquet")["strategy"]
    dates = strat.index
    cost_bps = cfg["backtest"]["cost_bps"]

    series = [
        (rolling_sharpe(spy_buy_hold(rets, dates)), C_SPY, "SPY B&H"),
        (rolling_sharpe(sixty_forty(rets, dates, cost_bps)), C_6040, "60/40"),
        (rolling_sharpe(strat), C_SISTEMA, T("System")),
    ]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.axhline(0, color="#9e9d99", lw=0.8)
    handles = []
    for s, color, name in series:
        (line,) = ax.plot(s.index, s, color=color, lw=1.6)
        handles.append(line)
        ax.annotate(name, (s.index[-1], s.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold", va="center")
    ax.legend(handles, [n for _, _, n in series], ncol=3, loc="upper left")
    ax.set_ylabel(T("Rolling Sharpe (3-year window)"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rolling_sharpe.pdf")
    plt.close(fig)
    print(f"OK: {FIG_DIR / 'rolling_sharpe.pdf'}")


if __name__ == "__main__":
    main()
