"""Paso 4: walk-forward completo del HMM dual -> output/regimes.parquet.

Es el paso caro (ensemble de 10 inicializaciones x 2 HMM por ventana);
el resultado se cachea para que los experimentos de asignación no
reentrenen los HMM.
"""

import pandas as pd

from src.config.settings import DATA_DIR, OUTPUT_DIR, load_config
from src.regimes.walk_forward_hmm import walk_forward_regimes


def main() -> None:
    cfg = load_config()
    feats = pd.read_parquet(DATA_DIR / "features.parquet")
    print(f"Walk-forward ({cfg['walk_forward']['scheme']}, "
          f"n_init={cfg['hmm']['n_init']}, selection={cfg['hmm']['selection']})...")
    regimes = walk_forward_regimes(feats, cfg)
    regimes.to_parquet(OUTPUT_DIR / "regimes.parquet")
    print(f"\nRegímenes out-of-sample: {regimes.index[0].date()} -> "
          f"{regimes.index[-1].date()} ({len(regimes)} días)")
    print(regimes["regime"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
