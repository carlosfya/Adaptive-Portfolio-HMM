"""Idioma de los rótulos de las figuras.

Las dos versiones de la memoria (espanol/ e ingles/) comparten los mismos
scripts de figuras, pero cada una necesita los textos en su idioma. Este
módulo centraliza esa traducción:

    FIG_LANG=es python scripts/make_figures.py   -> memoria/figuras/
    FIG_LANG=en python scripts/make_figures.py   -> memoria/figuras_en/

El idioma por defecto es el español, que es la versión principal del TFG.
Las cadenas se escriben en inglés en el código y T() las traduce; así, si
falta una traducción, la figura sale en inglés en lugar de romperse.

Nota: no se traducen los tickers de los ETFs (SPY, TLT...), los nombres de
los cuatro regímenes (Growth, Crash...) ni las siglas técnicas (EWMA, EM,
HMM, Sharpe), porque en la memoria se usan igualmente en inglés.
"""

import os
from pathlib import Path

LANG = os.environ.get("FIG_LANG", "es").lower()

_ES = {
    # --- make_figures.py -------------------------------------------------
    "SPY (log scale)": "SPY (escala logarítmica)",
    "Growth of $1 (log)": "Crecimiento de 1$ (escala log.)",
    "Drawdown": "Caída desde máximos",
    "Sharpe out-of-sample": "Sharpe fuera de muestra",
    "Rolling Sharpe (3-year window)": "Sharpe móvil (ventana de 3 años)",
    "System": "Sistema",
    "n_init=1 (range {})": "n_init=1 (rango {})",
    "ensemble 10 (range {})": "ensemble de 10 (rango {})",
    # --- make_figures_thesis.py ------------------------------------------
    "SPY 21-day realized volatility": "Volatilidad realizada de SPY a 21 días",
    "6-month SPY-TLT correlation": "Correlación SPY-TLT a 6 meses",
    "2022: both fall together": "2022: caen las dos a la vez",
    "smoothed (uses the future)": "suavizada (usa el futuro)",
    "filtered (past only)": "filtrada (solo el pasado)",
    " market top,\n crash begins": " máximo del mercado,\n empieza el desplome",
    "P(stressed state)": "P(estado de estrés)",
    "SPY (stocks)": "SPY (acciones)",
    "TLT (bonds)": "TLT (bonos)",
    "Capital allocation": "Reparto del capital",
    "Risk contribution": "Reparto del riesgo",
    "log VIX": "log VIX",
    "SPY realized vol": "Vol. realizada SPY",
    "yield-curve z-score": "z-score curva de tipos",
    "credit-spread z-score": "z-score dif. de crédito",
    "expanding\n(system)": "expansiva\n(sistema)",
    "rolling\n750d": "móvil\n750d",
    "rolling\n1000d": "móvil\n1000d",
    "rolling\n1250d": "móvil\n1250d",
    "Literature review & design": "Revisión bibliográfica y diseño",
    "Data, features & HMM engine": "Datos, variables y motor HMM",
    "Allocation & backtesting": "Asignación y backtesting",
    "Validation funnel": "Cadena de validación",
    "Prototype analysis & rebuild": "Análisis del prototipo y reconstrucción",
    "Thesis writing": "Redacción de la memoria",
    "What the two HMMs observe (crisis episodes shaded)":
        "Lo que observan los dos HMM (episodios de crisis sombreados)",
    "EM initialization": "Inicialización de EM",
    "Anchor mean per raw state":
        "Media de la variable de referencia en cada estado bruto",
    'raw "state 0"': 'estado bruto "0"',
    'raw "state 1"': 'estado bruto "1"',
    "SPY (growth of 1)": "SPY (crecimiento de 1)",
    "System (growth of 1)": "Sistema (crecimiento de 1)",
    "Calendar-year return": "Rentabilidad por año natural",
    "Sharpe ratio": "Ratio de Sharpe",
    "Maximum drawdown": "Caída máxima",
    " defense": " defensa",
    " submission": " entrega",
    "Sharpe of the 60/40": "Sharpe de la 60/40",
    "Number of random strategies tried (N)":
        "Número de estrategias aleatorias probadas (N)",
    "Expected best Sharpe\n(pure luck, 10-year backtest)":
        "Mejor Sharpe esperado\n(pura suerte, backtest de 10 años)",
    "EWMA volatility (ann.)": "Volatilidad EWMA (anualizada)",
    "Inverse-vol weight": "Peso por volatilidad inversa",
    # --- make_figures_extra.py -------------------------------------------
    "Portfolio weight": "Peso en la cartera",
    "Regime on day $t$": "Régimen el día $t$",
    "Regime on day $t-1$": "Régimen el día $t-1$",
    "Daily probability": "Probabilidad diaria",
    "SPY (observed)": "SPY (observado)",
    "60/40 (observed)": "60/40 (observado)",
    "Bootstrapped Sharpe ratio of the system":
        "Sharpe del sistema por remuestreo bootstrap",
    "Replicas": "Réplicas",
    "Sharpe ratio (out-of-sample)": "Ratio de Sharpe (fuera de muestra)",
    "base (minimal set)": "base (conjunto mínimo)",
    "+ SPY-TLT correlation": "+ correlación SPY-TLT",
    "+ VIX term structure": "+ estructura temporal del VIX",
    "macro in levels (no z-scores)": "macro en niveles (sin z-scores)",
    "macro levels + z-scores": "macro niveles + z-scores",
    "initial configuration (7 features)": "configuración inicial (7 variables)",
    "Filtered regime probability": "Probabilidad filtrada de régimen",
    "hard switch": "cambio brusco",
    "soft mixture": "mezcla blanda",
    "Drawdown from peak": "Caída desde máximos",
    "Sharpe of the 56 variants tried": "Sharpe de las 56 variantes probadas",
    "Number of variants": "Número de variantes",
    "threshold to beat\nby pure chance\n$SR_0$ = {}": "umbral a batir\npor puro azar\n$SR_0$ = {}",
    "system: {}": "sistema: {}",
    "worst: {}": "peor variante: {}",
    "SPY daily return": "Rendimiento diario de SPY",
    "standard deviations from the mean": "desviaciones típicas desde la media",
    "skew {}\nexcess kurtosis {}": "asimetría {}\ncurtosis en exceso {}",
    "Density": "Densidad",
    "fitted normal": "normal ajustada",
    "P(market stress)": "P(estrés de mercado)",
    "P(macro contraction)": "P(contracción macro)",
    "all days: corr {}\nstress days (red): corr {}":
        "todos los días: corr {}\ndías de estrés (rojo): corr {}",
    "Share of days with both signals on":
        "Días con las dos señales activas",
    "actual\n(observed)": "real\n(observado)",
    "independence\n(product)": "independencia\n(producto)",
}

_TABLES = {"es": _ES, "en": {}}

_MESES = {
    "es": ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
           "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"],
    "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
}

#: Abreviaturas de los meses en el idioma activo (matplotlib solo sabe inglés).
MESES = _MESES.get(LANG, _MESES["en"])


def T(text: str) -> str:
    """Traduce un rótulo al idioma activo (inglés si no hay traducción)."""
    return _TABLES.get(LANG, {}).get(text, text)


def fig_dir(root: Path) -> Path:
    """Carpeta de salida de las figuras según el idioma activo."""
    return root / "memoria" / ("figuras" if LANG == "es" else "figuras_en")
