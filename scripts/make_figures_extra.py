"""Figuras adicionales de apoyo visual para la memoria.

- weights_evolution.pdf : área apilada de los pesos de la cartera.
- transition_heatmap.pdf: matriz de transición diaria entre regímenes.
- monte_carlo.pdf       : distribución bootstrap del Sharpe del sistema.
- ablation_sharpe.pdf   : barras comparativas de Sharpe de las ablaciones.
- normality_features.pdf: normalidad dentro del estado, rendimientos vs log-VIX.
- prob_dependence.pdf   : dependencia entre los dos HMM (producto vs realidad).
- dsr_trials.pdf        : los 56 intentos frente al umbral de azar (DSR).

Mismos criterios que make_figures.py (paleta validada, marcas finas,
rejilla recesiva, etiquetas directas, un solo eje).
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
EXP_DIR = OUTPUT_DIR / "experiments"

C_SISTEMA = "#2a78d6"
C_GRIS = "#9e9d99"
# Un color por activo, agrupado por papel económico (renta variable en
# azules, bonos en verdes/aguas, defensivos/reales en cálidos)
ASSET_COLORS = {  # sólo los activos que el sistema final puede tener en cartera
    "SPY": "#2a78d6", "QQQ": "#86b6ef",             # renta variable
    "TLT": "#1baf7a", "SHY": "#8fd9bf",             # bonos
    "GLD": "#eda100", "DBC": "#eb6834", "XLP": "#4a3aa7",  # reales/defensivos
}

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
})


def fig_weights() -> None:
    """Área apilada de los pesos: el switch de universo hecho visible."""
    w = pd.read_parquet(OUTPUT_DIR / "weights.parquet")
    w = w[list(ASSET_COLORS)]
    # Remuestreo semanal para un PDF ligero sin cambiar la lectura
    w = w.resample("W").last().fillna(0.0)
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.stackplot(w.index, w.T.to_numpy(),
                 colors=[ASSET_COLORS[c] for c in w.columns],
                 labels=w.columns, lw=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel(T("Portfolio weight"))
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=8)
    ax.grid(visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weights_evolution.pdf")
    plt.close(fig)


def fig_transition() -> None:
    """Mapa de calor de la matriz de transición (secuencial, un solo tono)."""
    t = pd.read_csv(EXP_DIR / "exp01_matriz_transicion.csv", index_col=0)
    order = ["Growth", "Sideways_A", "Sideways_B", "Crash"]
    t = t.loc[order, order]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(t.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), order, rotation=30, ha="right")
    ax.set_yticks(range(4), order)
    ax.set_xlabel(T("Regime on day $t$"))
    ax.set_ylabel(T("Regime on day $t-1$"))
    for i in range(4):
        for j in range(4):
            v = t.iloc[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.5 else "#0b0b0b")
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, shrink=0.8, label=T("Daily probability"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "transition_heatmap.pdf")
    plt.close(fig)


def fig_monte_carlo() -> None:
    """Histograma bootstrap del Sharpe con los benchmarks como referencia."""
    cfg = variant_config()
    rng = np.random.default_rng(cfg["seed"])
    strat = pd.read_parquet(OUTPUT_DIR / "strategy_returns.parquet")["strategy"]
    rets = load_returns()
    spy = spy_buy_hold(rets, strat.index).to_numpy()
    b64 = sixty_forty(rets, strat.index, cfg["backtest"]["cost_bps"]).to_numpy()
    s = strat.to_numpy()
    n, block, n_boot = len(s), 21, 5000
    starts = rng.integers(0, n - block, size=(n_boot, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]

    def sharpe(x):
        return x.mean(axis=-1) / x.std(axis=-1) * np.sqrt(252)

    sh = sharpe(s[idx])
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.hist(sh, bins=60, color=C_SISTEMA, alpha=0.75, lw=0)
    # SPY y 60/40 tienen Sharpe casi idéntico: leyenda con estilos de
    # línea distintos en vez de etiquetas junto a las líneas.
    ax.axvline(sharpe(spy[None, :])[0], color="#52514e", lw=1.4, ls="--",
               label=T("SPY (observed)"))
    ax.axvline(sharpe(b64[None, :])[0], color="#9e9d99", lw=1.6, ls=":",
               label=T("60/40 (observed)"))
    ax.legend(loc="upper left", fontsize=8)
    ax.axvline(0, color="#0b0b0b", lw=1)
    ax.set_xlabel(T("Bootstrapped Sharpe ratio of the system"))
    ax.set_ylabel(T("Replicas"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monte_carlo.pdf")
    plt.close(fig)


def fig_ablation_bars() -> None:
    """Barras horizontales: Sharpe de las variantes de ablación de señales."""
    ab = pd.read_csv(EXP_DIR / "exp02_ablacion_features.csv", index_col=0)
    labels = {
        "base (mínima)": T("base (minimal set)"),
        "+ corr SPY-TLT": T("+ SPY-TLT correlation"),
        "+ vix_ts": T("+ VIX term structure"),
        "macro en niveles": T("macro in levels (no z-scores)"),
        "macro niveles+z": T("macro levels + z-scores"),
        "config. inicial (vol3+macro4)": T("initial configuration (7 features)"),
    }
    sh = ab["Sharpe"].rename(index=labels).iloc[::-1]
    colors = [C_SISTEMA if n == T("base (minimal set)") else C_GRIS
              for n in sh.index]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    bars = ax.barh(sh.index, sh, color=colors, height=0.62)
    for b, v in zip(bars, sh):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=8, color="#52514e")
    ax.set_xlabel(T("Sharpe ratio (out-of-sample)"))
    ax.set_xlim(0, 1.25)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_sharpe.pdf")
    plt.close(fig)


def fig_probabilities() -> None:
    """Área apilada de P(régimen) filtradas: el corazón del motor blando."""
    from src.allocation.allocator import regime_probabilities
    df = pd.read_parquet(OUTPUT_DIR / "regimes.parquet")
    probs = pd.DataFrame(
        [regime_probabilities(v, m) for v, m in zip(df["vol_p1"], df["macro_p1"])],
        index=df.index,
    )[["Growth", "Sideways_A", "Sideways_B", "Crash"]]
    probs = probs.resample("W").last().dropna()
    colors = ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    ax.stackplot(probs.index, probs.T.to_numpy(), colors=colors,
                 labels=probs.columns, lw=0, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel(T("Filtered regime probability"))
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=8)
    ax.grid(visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "regime_probabilities.pdf")
    plt.close(fig)


def fig_hard_vs_soft() -> None:
    """Motor blando vs brusco con los MISMOS regimenes, en dos paneles.

    El panel de riqueza por si solo ENGANA: el motor brusco va por encima
    el 87% de los dias y solo pierde al final. La ventaja real del blando
    es de riesgo, asi que el segundo panel (caida desde maximos) es el que
    sostiene el argumento y se muestra junto al primero.
    """
    from src.backtest.engine import run_backtest
    cfg = variant_config()
    cfg_hard = variant_config({"allocation": {"mode": "hard"}})
    rets = load_returns()
    regimes = pd.read_parquet(OUTPUT_DIR / "regimes.parquet")
    regimes = regimes.loc[regimes.index.isin(rets.index)]
    series = [
        (run_backtest(rets, regimes, cfg_hard)["returns"], C_GRIS,
         T("hard switch"), -10),
        (run_backtest(rets, regimes, cfg)["returns"], C_SISTEMA,
         T("soft mixture"), 8),
    ]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.0), sharex=True)
    for r, color, name, dy in series:
        eq = (1 + r).cumprod()
        ax.plot(eq.index, eq, color=color, lw=1.6)
        ax.annotate(name, (eq.index[-1], eq.iloc[-1]), xytext=(6, dy),
                    textcoords="offset points", color=color, fontsize=9,
                    fontweight="bold", va="center")
        ax2.plot(eq.index, eq / eq.cummax() - 1, color=color, lw=1.3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:g}$")
    ax.yaxis.set_minor_formatter(lambda x, _: "")
    ax.set_ylabel(T("Growth of $1 (log)"))
    ax2.set_ylabel(T("Drawdown from peak"))
    ax2.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.margins(x=0.10)
    ax2.margins(x=0.10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hard_vs_soft.pdf")
    plt.close(fig)


def fig_dsr() -> None:
    """Cap.5 5.7: el Sharpe deflactado, con los 56 intentos reales."""
    t = pd.read_csv(EXP_DIR / "exp18_trials.csv", index_col=0)["Sharpe"]
    d = pd.read_csv(EXP_DIR / "exp18_dsr.csv", index_col=0)["valor"]
    sr0 = d.loc["SR0 (max esperado por azar, anual)"]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.hist(t, bins=22, color=C_GRIS, alpha=0.85, lw=0)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)  # aire para las anotaciones
    top = ax.get_ylim()[1]
    ax.axvline(sr0, color="#e34948", lw=1.6)
    ax.annotate(T("threshold to beat\nby pure chance\n$SR_0$ = {}")
                .format(f"{sr0:.2f}"),
                xy=(sr0, top * 0.97), xytext=(7, 0),
                textcoords="offset points", color="#e34948", fontsize=8,
                va="top")
    # Etiquetas de los extremos por ARRIBA: debajo del eje chocan con el
    # rotulo del eje x.
    # El maximo de los trials (1.25) es la variante SIN costes, no el
    # sistema: se marca el Sharpe del sistema por separado para no
    # atribuirle una cifra que no es la suya. La fila base de la ablacion
    # de features ES la configuracion final.
    sistema = t.loc[[i for i in t.index if i.startswith("exp02") and "base" in i]].iloc[0]
    # Etiquetas cortas junto a su propia linea y a alturas distintas: con
    # flechas o rotadas se solapaban entre si y con el rotulo del eje x.
    for val, color, key, ha, dx, y in [
            (t.min(), "#52514e", "worst: {}", "right", -4, 0.80),
            (sistema, C_SISTEMA, "system: {}", "right", -4, 0.60)]:
        ax.axvline(val, color=color, lw=1.0, ls=":")
        ax.annotate(T(key).format(f"{val:.2f}"), xy=(val, top * y),
                    xytext=(dx, 0), textcoords="offset points", ha=ha,
                    va="center", fontsize=8, color=color)
    ax.set_xlim(0, 1.45)
    ax.set_xlabel(T("Sharpe of the 56 variants tried"))
    ax.set_ylabel(T("Number of variants"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dsr_trials.pdf")
    plt.close(fig)


def fig_normality() -> None:
    """Por qué los HMM no observan rendimientos: normalidad dentro del estado.

    Compara, dentro del estado de calma del HMM de volatilidad, la
    distribucion de los rendimientos diarios de SPY con la del log-VIX.
    Ambas se contrastan con la normal ajustada (misma media y desviacion).
    """
    from scipy import stats

    feats = pd.read_parquet(OUTPUT_DIR / "data" / "features.parquet")
    regimes = pd.read_parquet(OUTPUT_DIR / "regimes.parquet")
    spy = pd.read_parquet(OUTPUT_DIR / "data" / "prices.parquet")["SPY"]
    df = pd.concat([feats["log_vix"], regimes["vol_state"],
                    spy.pct_change().rename("spy")], axis=1).dropna()
    calm = df[df["vol_state"] == 0]

    panels = [("spy", T("SPY daily return"), C_SISTEMA),
              ("log_vix", "log-VIX", "#1baf7a")]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for ax, (col, title, color) in zip(axes, panels):
        x = calm[col].to_numpy()
        # Estandarizado: las dos escalas son incomparables en bruto
        z = (x - x.mean()) / x.std()
        ax.hist(z, bins=70, density=True, color=color, alpha=0.55, lw=0)
        grid = np.linspace(-6, 6, 400)
        ax.plot(grid, stats.norm.pdf(grid), color="#3d3929", lw=1.4)
        ax.set_xlim(-6, 6)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel(T("standard deviations from the mean"))
        ax.annotate(T("skew {}\nexcess kurtosis {}").format(
                        f"{stats.skew(x):+.2f}", f"{stats.kurtosis(x):+.2f}"),
                    xy=(0.03, 0.95), xycoords="axes fraction", va="top",
                    fontsize=8, color="#3d3929")
    axes[0].set_ylabel(T("Density"))
    axes[1].annotate(T("fitted normal"), xy=(1.5, 0.22), xytext=(2.6, 0.33),
                     fontsize=8, color="#3d3929",
                     arrowprops={"arrowstyle": "-", "color": "#3d3929",
                                 "lw": 0.7})
    fig.tight_layout()
    fig.savefig(FIG_DIR / "normality_features.pdf")
    plt.close(fig)


def fig_prob_dependence() -> None:
    """La independencia entre los dos HMM: cuanto se aparta de la realidad."""
    df = pd.read_parquet(OUTPUT_DIR / "regimes.parquet")
    v, m = df["vol_p1"], df["macro_p1"]
    stress = v > 0.5
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.3),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    ax.scatter(v[~stress], m[~stress], s=4, alpha=0.25, color=C_GRIS, lw=0)
    ax.scatter(v[stress], m[stress], s=4, alpha=0.35, color="#e34948", lw=0)
    ax.set_xlabel(T("P(market stress)"))
    ax.set_ylabel(T("P(macro contraction)"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate(T("all days: corr {}\nstress days (red): corr {}").format(
                    f"{v.corr(m):.2f}", f"{v[stress].corr(m[stress]):.2f}"),
                xy=(0.03, 0.97), xycoords="axes fraction", va="top",
                fontsize=8, color="#3d3929")

    emp = (stress & (m > 0.5)).mean()
    prod = stress.mean() * (m > 0.5).mean()
    ax2.bar([T("actual\n(observed)"), T("independence\n(product)")], [emp, prod],
            color=["#e34948", C_GRIS], width=0.55)
    for i, val in enumerate([emp, prod]):
        ax2.annotate(f"{val:.1%}", xy=(i, val), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=8,
                     fontweight="bold")
    ax2.set_ylabel(T("Share of days with both signals on"))
    ax2.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax2.set_ylim(0, max(emp, prod) * 1.28)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prob_dependence.pdf")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_weights()
    fig_transition()
    fig_monte_carlo()
    fig_ablation_bars()
    fig_probabilities()
    fig_hard_vs_soft()
    fig_normality()
    fig_prob_dependence()
    fig_dsr()
    print(f"Figuras extra guardadas en {FIG_DIR}")


if __name__ == "__main__":
    main()
