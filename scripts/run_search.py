"""Busqueda aleatoria de hiperparametros (mismo presupuesto para los 3 modelos). Solo usa train/val."""
import json, sys, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsbase import config as C
from tsbase.data import Windows, load_energy
from tsbase.models import SPACES
from tsbase.train import fit, predict
from tsbase.metrics import summarize

W = Windows(load_energy()[C.TARGET].values); tr, va = W.get("train"), W.get("val"); sc = W.mase_scale()
rng = np.random.default_rng(C.SPACE_SEED)
out = {}
for name, space in SPACES.items():
    keys = list(space); grid = list(itertools.product(*[space[k] for k in keys]))
    idx = rng.choice(len(grid), size=C.SEARCH_BUDGET, replace=False)
    rows = []
    for i in idx:
        cfg = dict(zip(keys, grid[i]))
        m, h, meta = fit(name, cfg, C.SEARCH_SEED, tr, va)
        s = summarize(va["truth"], W.to_levels(predict(m, va["X"]), va["last"]), sc)
        rows.append(dict(cfg=cfg, val_MASE=s["MASE_1:12"], epochs=meta["epochs"], n_params=meta["n_params"], time_s=meta["train_time_s"]))
        print(name, cfg, round(s["MASE_1:12"], 4), flush=True)
    best = min(rows, key=lambda r: r["val_MASE"])
    out[name] = dict(space_size=len(grid), trials=rows, best=best["cfg"], best_val_MASE=best["val_MASE"])
(C.RESULTS).mkdir(exist_ok=True)
json.dump(out, open(C.RESULTS / "search.json", "w"), indent=1)
print({k: (v["best"], round(v["best_val_MASE"], 4)) for k, v in out.items()})
