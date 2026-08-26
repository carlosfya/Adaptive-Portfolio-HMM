"""Paso 3 (diagnóstico): ajusta el ensemble HMM dual en la primera ventana.

Sirve como comprobación de cordura del ajuste y del criterio de label
switching antes de lanzar el walk-forward completo: imprime las medias por
estado de la feature-ancla y verifica que estado 1 = valor alto del ancla.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config.settings import DATA_DIR, load_config
from src.regimes.walk_forward_hmm import anchor_permutation, fit_hmm


def main() -> None:
    cfg = load_config()
    feats = pd.read_parquet(DATA_DIR / "features.parquet")
    t0 = cfg["walk_forward"]["train_min_days"]
    hmm_cfg, fcfg = cfg["hmm"], cfg["features"]

    for name, cols, anchor in [
        ("Volatilidad", fcfg["vol_block"], hmm_cfg["anchor_vol"]),
        ("Macro", fcfg["macro_block"], hmm_cfg["anchor_macro"]),
    ]:
        X = StandardScaler().fit_transform(feats[cols].iloc[:t0])
        a_idx = cols.index(anchor)
        print(f"\nHMM {name} (ancla: {anchor}) — ensemble n_init={hmm_cfg['n_init']}")
        for i in range(hmm_cfg["n_init"]):
            model = fit_hmm(X, hmm_cfg, cfg["seed"] + i)
            perm = anchor_permutation(model, a_idx)
            means = model.means_[:, a_idx]
            # Medias del ancla ya reordenadas: canónico 0 (bajo), 1 (alto)
            canon = {perm[s]: means[s] for s in range(len(means))}
            print(f"  init {i}: media ancla estado0={canon[0]:+.2f} "
                  f"estado1={canon[1]:+.2f}  (perm cruda -> canónica: {list(perm)})")


if __name__ == "__main__":
    main()
