# Sistema de Trading Algorítmico Adaptativo Basado en Modelado de Regímenes de Mercado

TFG de Carlos Fernández-Yáñez Arce (UC3M). Asignador de cartera **defensivo**
que estima el régimen de mercado con un **HMM dual** (volatilidad + macro,
ensemble de 10 inicializaciones) y construye cada mes una **mezcla blanda
causal**: `w = Σ P(régimen) · w_régimen`, con probabilidades **filtradas**
(solo pasado) y ponderación única inverse-volatility por universo. El switch
duro de universo se conserva como ablación (`allocation.mode: hard`).

## Flujo mínimo (end-to-end)

```bash
python data_fetch.py        # precios ETF (yfinance) + macro (FRED) -> output/data/
python build_features.py    # features causales (shift(1) en todo)
python train_hmm.py         # diagnóstico: ajuste + label switching en 1ª ventana
python run_walk_forward.py  # walk-forward HMM dual -> output/regimes.parquet (caro, cacheado)
python run_backtest.py      # backtest causal con costes -> métricas vs SPY y 60/40
```

## Los tres pilares metodológicos

1. **Causalidad**: `shift(1)` en todas las features; escalado ajustado sólo con
   el train de cada ventana; estados por **filtrado forward** (nunca Viterbi ni
   suavizado, que miran el futuro).
2. **Backtesting realista**: ejecución diferida (se decide en t−1, se aplica en
   t), 10 bps de fricción sobre el turnover, benchmarks SPY B&H y 60/40 mensual.
3. **Label switching**: reordenación determinista de estados por feature-ancla
   (`log_vix` para el HMM de volatilidad, `baa10y_z` para el macro).

## Estructura

Configuración única en `config.yaml`; resultados cacheados en `output/`.
