"""Figuras conceptuales y de apoyo adicionales para la memoria.

Cap. 1: vol_clustering.pdf, stock_bond_corr.pdf
Cap. 2: filtered_vs_smoothed.pdf, risk_contribution_6040.pdf
Cap. 3: features_timeline.pdf
Cap. 4: label_switching.pdf
Cap. 5: case_study_2020.pdf, annual_returns.pdf
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_features, load_returns, variant_config  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.backtest.benchmarks import sixty_forty, spy_buy_hold  # noqa: E402
from figlang import MESES, T, fig_dir  # noqa: E402
from src.config.settings import DATA_DIR, OUTPUT_DIR, ROOT  # noqa: E402
from src.regimes.walk_forward_hmm import (  # noqa: E402
    anchor_permutation, fit_hmm, forward_filter_proba,
)

FIG_DIR = fig_dir(ROOT)
C_SIS, C_GRIS, C_NEGRO = "#2a78d6", "#9e9d99", "#0b0b0b"
C_ROJO, C_VERDE, C_AMBAR = "#e34948", "#1baf7a", "#eda100"

CRISES = [("2008-09-01", "2009-03-31"), ("2011-07-15", "2011-10-15"),
          ("2015-08-15", "2015-10-15"), ("2018-10-01", "2018-12-31"),
          ("2020-02-20", "2020-04-30"), ("2022-01-01", "2022-10-31")]

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
})


def _shade_crises(ax):
    for a, b in CRISES:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=C_ROJO,
                   alpha=0.12, lw=0)


def fig_vol_clustering() -> None:
    """Cap.1: la volatilidad llega en rachas -> no estacionariedad."""
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    rv = prices["SPY"].pct_change().rolling(21).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.plot(rv.index, rv, color=C_NEGRO, lw=1.0)
    _shade_crises(ax)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_ylabel(T("SPY 21-day realized volatility"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vol_clustering.pdf")
    plt.close(fig)


def fig_stock_bond_corr() -> None:
    """Cap.1: la correlación acciones-bonos es un régimen, no una ley."""
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    rets = prices[["SPY", "TLT"]].pct_change()
    corr = rets["SPY"].rolling(126).corr(rets["TLT"])
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.plot(corr.index, corr, color=C_NEGRO, lw=1.0)
    ax.axhline(0, color=C_GRIS, lw=1)
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
               color=C_ROJO, alpha=0.15, lw=0)
    ax.annotate(T("2022: both fall together"), xy=(pd.Timestamp("2022-06-01"), 0.45),
                fontsize=9, color=C_ROJO, ha="center")
    ax.set_ylabel(T("6-month SPY-TLT correlation"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stock_bond_corr.pdf")
    plt.close(fig)


def fig_filtered_vs_smoothed() -> None:
    """Cap.2/4: la probabilidad suavizada 'sabe antes' porque mira el futuro."""
    cfg = variant_config()
    feats = load_features()
    cols = cfg["features"]["vol_block"]
    a_idx = cols.index(cfg["hmm"]["anchor_vol"])
    end = feats.index.searchsorted(pd.Timestamp("2020-12-31"))
    X = StandardScaler().fit_transform(feats[cols].iloc[:end])
    model = fit_hmm(X, cfg["hmm"], cfg["seed"])
    perm = anchor_permutation(model, a_idx)
    inv = np.argsort(perm)
    filt = forward_filter_proba(model, X)[:, inv][:, 1]
    smooth = model.predict_proba(X)[:, inv][:, 1]
    idx = feats.index[:end]
    # Zoom fino a la TRANSICIÓN (feb-mar 2020), con marcador por día:
    # es donde se ve que la suavizada sube DÍAS ANTES que la filtrada
    lo = idx.searchsorted(pd.Timestamp("2020-02-03"))
    hi = idx.searchsorted(pd.Timestamp("2020-04-15"))
    fig, ax = plt.subplots(figsize=(8.5, 2.9))
    ax.plot(idx[lo:hi], smooth[lo:hi], color=C_GRIS, lw=1.6, ls="--",
            marker="o", ms=3.5, label=T("smoothed (uses the future)"))
    ax.plot(idx[lo:hi], filt[lo:hi], color=C_SIS, lw=1.6,
            marker="o", ms=3.5, label=T("filtered (past only)"))
    ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-15"),
               color=C_ROJO, alpha=0.10, lw=0)
    ax.axvline(pd.Timestamp("2020-02-20"), color=C_ROJO, lw=1, ls=":")
    ax.text(pd.Timestamp("2020-02-21"), 0.55, T(" market top,\n crash begins"),
            color=C_ROJO, fontsize=8)
    ax.set_ylabel(T("P(stressed state)"))
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "filtered_vs_smoothed.pdf")
    plt.close(fig)


def fig_risk_contribution() -> None:
    """Cap.2: el 60/40 reparte capital, no riesgo."""
    rets = load_returns()[["SPY", "TLT"]]
    cov = rets.cov() * 252
    w = np.array([0.6, 0.4])
    rc = w * (cov.to_numpy() @ w)
    rc = rc / rc.sum()
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    labels = [T("Capital allocation"), T("Risk contribution")]
    spy_vals, tlt_vals = [0.6, rc[0]], [0.4, rc[1]]
    ax.barh(labels, spy_vals, color=C_SIS, height=0.55, label=T("SPY (stocks)"))
    ax.barh(labels, tlt_vals, left=spy_vals, color=C_VERDE, height=0.55,
            label=T("TLT (bonds)"))
    for y, v in enumerate(spy_vals):
        ax.text(v / 2, y, f"{v:.0%}", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
        ax.text(v + (1 - v) / 2, y, f"{1 - v:.0%}", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "risk_contribution_6040.pdf")
    plt.close(fig)


def fig_features_timeline() -> None:
    """Cap.3: las 4 features del sistema con las crisis sombreadas."""
    feats = load_features()
    panels = [("log_vix", T("log VIX")), ("rv21_spy", T("SPY realized vol")),
              ("t10y2y_z", T("yield-curve z-score")),
              ("baa10y_z", T("credit-spread z-score"))]
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 6.0), sharex=True)
    for ax, (col, name) in zip(axes, panels):
        ax.plot(feats.index, feats[col], color=C_NEGRO, lw=0.8)
        _shade_crises(ax)
        ax.set_ylabel(name, fontsize=8)
    axes[0].set_title(T("What the two HMMs observe (crisis episodes shaded)"),
                      fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "features_timeline.pdf")
    plt.close(fig)


def fig_label_switching() -> None:
    """Cap.4: los índices crudos del HMM se intercambian entre inits."""
    cfg = variant_config()
    feats = load_features()
    cols = cfg["features"]["vol_block"]
    a_idx = cols.index(cfg["hmm"]["anchor_vol"])
    t0 = cfg["walk_forward"]["train_min_days"]
    X = StandardScaler().fit_transform(feats[cols].iloc[:t0])
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for i in range(cfg["hmm"]["n_init"]):
        model = fit_hmm(X, cfg["hmm"], cfg["seed"] + i)
        means = model.means_[:, a_idx]
        for raw, m in enumerate(means):
            color = C_ROJO if m > 0 else C_SIS
            marker = "o" if raw == 0 else "s"
            ax.scatter(i, m, s=55, color=color, marker=marker, zorder=3)
    ax.axhline(0, color=C_GRIS, lw=1)
    ax.set_xticks(range(10), [f"{i}" for i in range(10)])
    ax.set_xlabel(T("EM initialization"))
    ax.set_ylabel(T("Anchor mean per raw state"))
    ax.scatter([], [], marker="o", color=C_NEGRO, label=T('raw "state 0"'))
    ax.scatter([], [], marker="s", color=C_NEGRO, label=T('raw "state 1"'))
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "label_switching.pdf")
    plt.close(fig)


def fig_case_study_2020() -> None:
    """Cap.5: el COVID en 3 paneles: SPY, P(Crash) y la cartera."""
    from src.allocation.allocator import regime_probabilities
    cfg = variant_config()
    rets = load_returns()
    strat = pd.read_parquet(OUTPUT_DIR / "strategy_returns.parquet")["strategy"]
    reg = pd.read_parquet(OUTPUT_DIR / "regimes.parquet")
    pc = pd.Series([regime_probabilities(v, m)["Crash"]
                    for v, m in zip(reg["vol_p1"], reg["macro_p1"])],
                   index=reg.index)
    lo, hi = "2020-01-01", "2020-12-31"
    spy = (1 + rets.loc[lo:hi, "SPY"]).cumprod()
    eq = (1 + strat.loc[lo:hi]).cumprod()
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 5.6), sharex=True)
    axes[0].plot(spy.index, spy, color=C_GRIS, lw=1.4)
    axes[0].set_ylabel(T("SPY (growth of 1)"))
    axes[1].fill_between(pc.loc[lo:hi].index, pc.loc[lo:hi], color=C_ROJO,
                         alpha=0.6, lw=0)
    axes[1].set_ylabel("P(Crash)")
    axes[1].set_ylim(0, 1)
    axes[2].plot(eq.index, eq, color=C_SIS, lw=1.4)
    axes[2].set_ylabel(T("System (growth of 1)"))
    for ax in axes:
        ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30"),
                   color=C_ROJO, alpha=0.10, lw=0)
    lim = (min(spy.min(), eq.min()) * 0.98, max(spy.max(), eq.max()) * 1.02)
    axes[0].set_ylim(lim)
    axes[2].set_ylim(lim)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case_study_2020.pdf")
    plt.close(fig)


def fig_annual_returns() -> None:
    """Cap.5: retorno por año natural: consistencia, no un año de suerte."""
    cfg = variant_config()
    rets = load_returns()
    strat = pd.read_parquet(OUTPUT_DIR / "strategy_returns.parquet")["strategy"]
    spy = spy_buy_hold(rets, strat.index)
    b64 = sixty_forty(rets, strat.index, cfg["backtest"]["cost_bps"])
    def annual(s):
        return (1 + s).groupby(s.index.year).prod() - 1
    df = pd.DataFrame({T("System"): annual(strat), "60/40": annual(b64),
                       "SPY": annual(spy)})
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    for k, (col, color) in enumerate([("SPY", C_GRIS), ("60/40", "#52514e"),
                                      (T("System"), C_SIS)]):
        ax.bar(x + (k - 1) * 0.27, df[col], width=0.25, color=color, label=col)
    ax.axhline(0, color=C_NEGRO, lw=0.8)
    ax.set_xticks(x, df.index, rotation=45)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel(T("Calendar-year return"))
    ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "annual_returns.pdf")
    plt.close(fig)


def fig_window_sensitivity() -> None:
    """Cap.5 §5.3: Sharpe y MaxDD por esquema de ventana (dos paneles)."""
    df = pd.read_csv(OUTPUT_DIR / "experiments" / "exp03_sensibilidad_ventana.csv",
                     index_col=0)
    labels = {"expansiva (base)": T("expanding\n(system)"),
              "rolling 750d": T("rolling\n750d"),
              "rolling 1000d": T("rolling\n1000d"),
              "rolling 1250d": T("rolling\n1250d")}
    df = df.rename(index=labels)
    colors = [C_SIS if i == 0 else C_GRIS for i in range(len(df))]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.6))
    axes[0].bar(df.index, df["Sharpe"], color=colors, width=0.6)
    axes[0].set_ylabel(T("Sharpe ratio"))
    axes[1].bar(df.index, df["MaxDD"], color=colors, width=0.6)
    axes[1].set_ylabel(T("Maximum drawdown"))
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    for ax in axes:
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "window_sensitivity.pdf")
    plt.close(fig)


def fig_gantt() -> None:
    """Cap.7: planificación temporal del proyecto (diagrama de Gantt)."""
    import matplotlib.dates as mdates
    # El ultimo campo marca la fase de redaccion (color distinto); antes se
    # detectaba buscando "writing" en el nombre, lo que se rompia al traducir.
    phases = [
        (T("Literature review & design"), "2026-01-12", "2026-02-28", False),
        (T("Data, features & HMM engine"), "2026-02-16", "2026-04-30", False),
        (T("Allocation & backtesting"), "2026-04-13", "2026-05-31", False),
        (T("Validation funnel"), "2026-05-18", "2026-07-15", False),
        (T("Prototype analysis & rebuild"), "2026-06-15", "2026-07-31", False),
        (T("Thesis writing"), "2026-06-01", "2026-09-10", True),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    for i, (name, a, b, redaccion) in enumerate(reversed(phases)):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        ax.barh(i, (b - a).days, left=a, height=0.55,
                color=C_AMBAR if redaccion else C_SIS)
    ax.set_yticks(range(len(phases)), [p[0] for p in reversed(phases)],
                  fontsize=8)
    # Dos hitos: entrega de la memoria y defensa ante el tribunal.
    for fecha, key, y in [("2026-09-10", " submission", len(phases) - 0.3),
                          ("2026-10-15", " defense", len(phases) - 1.1)]:
        h = pd.Timestamp(fecha)
        ax.axvline(h, color=C_ROJO, lw=1.4, ls="--")
        ax.text(h, y, T(key), color=C_ROJO, fontsize=8)
    ax.set_xlim(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-11-05"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # matplotlib abrevia los meses siempre en inglés, así que se formatean
    # a mano para que el eje siga el idioma de la figura.
    ax.xaxis.set_major_formatter(
        lambda v, _: MESES[mdates.num2date(v).month - 1])
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gantt.pdf")
    plt.close(fig)


def fig_max_sharpe_search() -> None:
    """Cap.2 §2.3.1: el mejor Sharpe de N estrategias ALEATORIAS crece solo."""
    n = np.arange(1, 1001)
    # E[max de N normales estándar] ~ sqrt(2 ln N) (Bailey et al. 2014);
    # en Sharpe anualizado de un backtest de 10 años: SR ~ z / sqrt(años)
    exp_max = np.sqrt(2 * np.log(n)) / np.sqrt(10)
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.plot(n, exp_max, color=C_SIS, lw=1.8)
    ax.axhline(0.85, color=C_GRIS, lw=1.2, ls="--")
    ax.text(1000, 0.87, T("Sharpe of the 60/40"), color="#52514e",
            fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.set_xlabel(T("Number of random strategies tried (N)"))
    ax.set_ylabel(T("Expected best Sharpe\n(pure luck, 10-year backtest)"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "max_sharpe_search.pdf")
    plt.close(fig)


def fig_ewma_invvol() -> None:
    """Cap.3 §3.4.2: de la vol EWMA a los pesos inverse-vol.

    Se usa el universo Sideways_A (SPY, GLD, DBC), cuyas volatilidades son
    comparables: así la re-asignación se VE. En universos con SHY (vol
    ~0.5%) el mecanismo es idéntico pero SHY domina siempre (~85%), y el
    cambio relativo entre los demás resulta invisible en un área apilada.
    """
    rets = load_returns()
    universe = ["SPY", "GLD", "DBC"]
    colors = {"SPY": "#2a78d6", "GLD": "#eda100", "DBC": "#eb6834"}
    vol = rets[universe].ewm(span=63).std() * np.sqrt(252)
    vol = vol.loc["2019":"2021"].resample("W").last()
    inv = 1.0 / vol
    w = inv.div(inv.sum(axis=1), axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.4), sharex=True)
    for c in universe:
        axes[0].plot(vol.index, vol[c], color=colors[c], lw=1.4, label=c)
    axes[0].set_ylabel(T("EWMA volatility (ann.)"))
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].legend(ncol=3, loc="upper left", fontsize=8)
    for c in universe:
        axes[1].plot(w.index, w[c], color=colors[c], lw=1.4)
    axes[1].set_ylabel(T("Inverse-vol weight"))
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[1].set_ylim(0, 0.65)
    for ax in axes:
        ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30"),
                   color=C_ROJO, alpha=0.10, lw=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ewma_invvol.pdf")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for f in [fig_vol_clustering, fig_stock_bond_corr, fig_filtered_vs_smoothed,
              fig_risk_contribution, fig_features_timeline, fig_label_switching,
              fig_case_study_2020, fig_annual_returns, fig_window_sensitivity,
              fig_gantt, fig_max_sharpe_search, fig_ewma_invvol]:
        f()
        print(f"OK {f.__name__}")


if __name__ == "__main__":
    main()
