"""Figuras de la memoria (PDF vectorial para LaTeX).

Criterios de diseño (skill dataviz):
- La estrategia lleva el azul de la paleta; los benchmarks son contexto y
  van en grises (el color sigue a la entidad, no al ranking).
- Regímenes: 4 tonos categóricos validados CVD (orden fijo de la paleta).
- Marcas finas, rejilla recesiva, etiquetas directas al final de las líneas
  (además de la leyenda), un solo eje por figura, nada de arcoíris.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_returns, variant_config  # noqa: E402
from src.backtest.benchmarks import sixty_forty, spy_buy_hold  # noqa: E402
from figlang import T, fig_dir  # noqa: E402
from src.config.settings import OUTPUT_DIR, ROOT  # noqa: E402

FIG_DIR = fig_dir(ROOT)

# Paleta (referencia dataviz, modo claro)
C_SISTEMA = "#2a78d6"          # azul: la entidad protagonista
C_6040 = "#52514e"             # gris oscuro: benchmark
C_SPY = "#9e9d99"              # gris claro: benchmark
REGIME_COLORS = {              # slots categóricos validados (1,2,3,6)
    "Growth": "#2a78d6",
    "Sideways_A": "#1baf7a",
    "Sideways_B": "#eda100",
    "Crash": "#e34948",
}

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
})


def _label_end(ax, series, text, color):
    """Etiqueta directa al final de una línea."""
    ax.annotate(text, (series.index[-1], series.iloc[-1]),
                xytext=(6, 0), textcoords="offset points",
                color=color, fontsize=9, fontweight="bold", va="center")


def fig_regimes(active: pd.Series, spy_price: pd.Series) -> None:
    """SPY con el fondo sombreado por el régimen activo del sistema."""
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    spy = spy_price.loc[active.index]
    ax.plot(spy.index, spy, color="#0b0b0b", lw=1.2)
    # Sombrear tramos contiguos del mismo régimen
    change = (active != active.shift()).cumsum()
    for _, seg in active.groupby(change):
        ax.axvspan(seg.index[0], seg.index[-1],
                   color=REGIME_COLORS[seg.iloc[0]], alpha=0.18, lw=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4)
               for c in REGIME_COLORS.values()]
    ax.legend(handles, REGIME_COLORS.keys(), ncol=4, loc="upper left")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}")
    ax.yaxis.set_minor_formatter(lambda x, _: "")
    ax.set_ylabel(T("SPY (log scale)"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "regimes_timeline.pdf")
    plt.close(fig)


def fig_equity(strat, b6040, spy) -> None:
    """Crecimiento de 1$ (escala log): sistema vs benchmarks."""
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for rets, color, name in [(spy, C_SPY, "SPY B&H"),
                              (b6040, C_6040, "60/40"),
                              (strat, C_SISTEMA, T("System"))]:
        eq = (1 + rets).cumprod()
        ax.plot(eq.index, eq, color=color, lw=1.6)
        _label_end(ax, eq, name, color)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:g}$")
    ax.yaxis.set_minor_formatter(lambda x, _: "")
    ax.set_ylabel(T("Growth of $1 (log)"))
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "equity_curves.pdf")
    plt.close(fig)


def fig_drawdown(strat, b6040, spy) -> None:
    """Drawdown comparado: la reducción de riesgo es el relato del TFG."""
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    for rets, color, name in [(spy, C_SPY, "SPY B&H"),
                              (b6040, C_6040, "60/40"),
                              (strat, C_SISTEMA, T("System"))]:
        eq = (1 + rets).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd, color=color, lw=1.4, label=name)
    # Las tres series terminan cerca de 0%: leyenda en vez de etiquetas
    # directas para evitar el solape al final de las líneas.
    ax.legend(loc="lower left", ncol=3)
    ax.set_ylabel(T("Drawdown"))
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "drawdowns.pdf")
    plt.close(fig)


def fig_seed_stability() -> None:
    """Dot plot: Sharpe por semilla (n_init=1) vs ensemble (n_init=10)."""
    df = pd.read_csv(OUTPUT_DIR / "experiments" / "exp04_estabilidad_semilla.csv",
                     index_col=0)
    single = df[df.index.str.startswith("n_init=1")]["Sharpe"]
    ens = df[df.index.str.startswith("ensemble")]["Sharpe"]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    rng = np.random.default_rng(0)  # jitter vertical sólo estético
    ax.scatter(single, 1 + rng.uniform(-0.08, 0.08, len(single)),
               s=45, color=C_SPY, zorder=3)
    ax.scatter(ens, np.zeros(len(ens)), s=45, color=C_SISTEMA, zorder=3)
    x0 = min(single.min(), ens.min())  # etiqueta anclada al rango de datos
    for y, lbl, c in [
        (1, T("n_init=1 (range {})").format(f"{single.max()-single.min():.2f}"), C_6040),
        (0, T("ensemble 10 (range {})").format(f"{ens.max()-ens.min():.2f}"), C_SISTEMA)]:
        ax.text(x0, y + 0.25, lbl, color=c, fontsize=9, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel(T("Sharpe out-of-sample"))
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "seed_stability.pdf")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = variant_config()
    rets = load_returns()
    strat = pd.read_parquet(OUTPUT_DIR / "strategy_returns.parquet")["strategy"]
    active = pd.read_parquet(OUTPUT_DIR / "active_regimes.parquet")["active_regime"]
    dates = strat.index
    spy = spy_buy_hold(rets, dates)
    b6040 = sixty_forty(rets, dates, cfg["backtest"]["cost_bps"])
    prices = (1 + rets["SPY"]).cumprod() * 100

    fig_regimes(active, prices)
    fig_equity(strat, b6040, spy)
    fig_drawdown(strat, b6040, spy)
    fig_seed_stability()
    print(f"Figuras guardadas en {FIG_DIR}")


if __name__ == "__main__":
    main()
