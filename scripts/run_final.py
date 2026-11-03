"""Ejecucion definitiva: R semillas por modelo. El conjunto de prueba se evalua UNA vez por corrida."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import Ridge
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsbase import config as C
from tsbase.data import Windows, load_energy
from tsbase.train import fit, predict
from tsbase.metrics import summarize

R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SEEDS = list(range(1000, 1000 + R))
W = Windows(load_energy()[C.TARGET].values)
tr, va, te = W.get("train"), W.get("val"), W.get("test", allow_test=True); sc = W.mase_scale()
best = {k: v["best"] for k, v in json.load(open(C.RESULTS / "search.json")).items()}
(C.RESULTS / "preds").mkdir(parents=True, exist_ok=True); (C.RESULTS / "models").mkdir(exist_ok=True); (C.RESULTS / "hist").mkdir(exist_ok=True)
rows = []
for name, cfg in best.items():
    for s in SEEDS:
        m, hist, meta = fit(name, cfg, s, tr, va)
        t0 = time.time(); pv = W.to_levels(predict(m, va["X"]), va["last"]); inf_val = time.time() - t0
        t0 = time.time(); pt = W.to_levels(predict(m, te["X"]), te["last"]); inf_te = time.time() - t0
        rv, rt = summarize(va["truth"], pv, sc), summarize(te["truth"], pt, sc)
        trp = W.to_levels(predict(m, tr["X"]), tr["last"]); rtr = summarize(tr["truth"], trp, sc)
        rows.append(dict(model=name, seed=s, **{f"val_{k}": v for k, v in rv.items()}, **{f"test_{k}": v for k, v in rt.items()},
                         train_MASE=rtr["MASE_1:12"], train_RMSE=rtr["RMSE_1:12"], epochs=meta["epochs"], best_epoch=meta["best_epoch"],
                         n_params=meta["n_params"], train_time_s=meta["train_time_s"], infer_ms_per_window=1000 * inf_te / len(te["X"])))
        np.save(C.RESULTS / "preds" / f"{name}_{s}.npy", pt)
        torch.save(m.state_dict(), C.RESULTS / "models" / f"{name}_{s}.pt")
        json.dump(hist, open(C.RESULTS / "hist" / f"{name}_{s}.json", "w"))
        print(name, s, round(rv["MASE_1:12"], 4), round(rt["MASE_1:12"], 4), flush=True)

# --- referencias (deterministas) ---
org = te["org"]; y = W.y
pers = np.repeat(te["last"][:, None], C.H, 1)
seas = np.stack([[y[t + h - C.SEASONAL_PERIOD] for h in range(1, C.H + 1)] for t in org])
refs = {"Persistencia": pers, "Ingenuo estacional (m=52)": seas}
# Ridge sobre la misma ventana (alpha elegido en validacion)
bestA, bestS = None, np.inf
for a in [1e-3, 1e-2, 1e-1, 1, 10, 100]:
    rg = Ridge(alpha=a).fit(tr["X"], tr["Y"]); s = summarize(va["truth"], W.to_levels(rg.predict(va["X"]), va["last"]), sc)["MASE_1:12"]
    if s < bestS: bestA, bestS = a, s
rg = Ridge(alpha=bestA).fit(tr["X"], tr["Y"]); refs[f"Ridge lineal (alpha={bestA})"] = W.to_levels(rg.predict(te["X"]), te["last"])
ref_rows = []
for k, p in refs.items():
    ref_rows.append(dict(model=k, **{f"test_{a}": b for a, b in summarize(te["truth"], p, sc).items()}))
    np.save(C.RESULTS / "preds" / (k.split(" ")[0].replace("(", "") + ".npy"), p)
pd.DataFrame(rows).to_csv(C.RESULTS / "final_runs.csv", index=False)
pd.DataFrame(ref_rows).to_csv(C.RESULTS / "reference_models.csv", index=False)
np.save(C.RESULTS / "preds" / "truth_test.npy", te["truth"]); np.save(C.RESULTS / "preds" / "origins_test.npy", te["org"])
print(pd.DataFrame(ref_rows)[["model", "test_MASE_1:12", "test_RMSE_1:12"]])
