"""Curvas de aprendizaje y robustez a perturbaciones (solo VALIDACION; no consulta prueba)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsbase import config as C
from tsbase.data import Windows, load_energy
from tsbase.models import MODELS
from tsbase.train import fit, predict
from tsbase.metrics import summarize

W = Windows(load_energy()[C.TARGET].values); tr, va = W.get("train"), W.get("val"); sc = W.mase_scale()
best = {k: v["best"] for k, v in json.load(open(C.RESULTS / "search.json")).items()}

# 1) curvas de aprendizaje: ventanas de entrenamiento MAS RECIENTES (orden cronologico preservado)
rows = []
for name, cfg in best.items():
    for frac in (0.25, 0.5, 0.75, 1.0):
        n = int(round(frac * len(tr["X"])))
        sub = {k: (v[-n:] if hasattr(v, "__len__") else v) for k, v in tr.items()}
        for s in range(2000, 2005):
            m, h, meta = fit(name, cfg, s, sub, va)
            ptr = W.to_levels(predict(m, sub["X"]), sub["last"]); pva = W.to_levels(predict(m, va["X"]), va["last"])
            rows.append(dict(model=name, frac=frac, n_train=n, seed=s,
                             train_MASE=summarize(sub["truth"], ptr, sc)["MASE_1:12"], val_MASE=summarize(va["truth"], pva, sc)["MASE_1:12"]))
    print("curva", name, flush=True)
pd.DataFrame(rows).to_csv(C.RESULTS / "learning_curves.csv", index=False)

# 2) robustez: ruido gaussiano en la ventana de entrada (validacion), modelos finales guardados
r_std = float(np.std(np.diff(W.logy[: W.split.n_tr])) / W.scale)
rng = np.random.default_rng(7); rows = []
for name, cfg in best.items():
    kw = {k: v for k, v in cfg.items() if k not in ("lr", "weight_decay", "batch", "clip")}
    for s in range(1000, 1000 + 10):
        m = MODELS[name](**kw); m.load_state_dict(torch.load(C.RESULTS / "models" / f"{name}_{s}.pt")); m.eval()
        for k in (0.0, 0.5, 1.0, 2.0):
            X = va["X"].copy(); noise = rng.normal(0, k * r_std, X.shape); noise[:, -1] = 0.0   # el ultimo punto es la referencia
            p = W.to_levels(predict(m, X + noise), va["last"])
            rows.append(dict(model=name, seed=s, noise_k=k, val_MASE=summarize(va["truth"], p, sc)["MASE_1:12"]))
pd.DataFrame(rows).to_csv(C.RESULTS / "robustness_val.csv", index=False)
print("ok r_std", r_std)
