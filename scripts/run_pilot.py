"""Piloto: 5 semillas por modelo sobre VALIDACION con la configuracion seleccionada. Estima sigma_d y R."""
import json, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsbase import config as C
from tsbase.data import Windows, load_energy
from tsbase.train import fit, predict
from tsbase.metrics import summarize
from tsbase import stats as S

W = Windows(load_energy()[C.TARGET].values); tr, va = W.get("train"), W.get("val"); sc = W.mase_scale()
best = {k: v["best"] for k, v in json.load(open(C.RESULTS / "search.json")).items()}
rows = []
for name, cfg in best.items():
    for s in C.PILOT_SEEDS:
        m, h, meta = fit(name, cfg, s, tr, va)
        r = summarize(va["truth"], W.to_levels(predict(m, va["X"]), va["last"]), sc)
        rows.append(dict(model=name, seed=s, **r, epochs=meta["epochs"], n_params=meta["n_params"]))
        print(name, s, round(r["MASE_1:12"], 4), flush=True)
df = pd.DataFrame(rows); df.to_csv(C.RESULTS / "pilot.csv", index=False)
P = df.pivot(index="seed", columns="model", values="MASE_1:12")
summ = P.agg(["mean", "std"]).T; summ["cv"] = summ["std"] / summ["mean"]
best_mean = P.mean().min(); delta_abs = C.DELTA_REL * best_mean
pairs = {}
for a, b in itertools.combinations(P.columns, 2):
    d = (P[a] - P[b]).values; sd = d.std(ddof=1)
    pairs[f"{a}-{b}"] = dict(mean_diff=float(d.mean()), sigma_d=float(sd), delta_over_sigma=float(delta_abs / sd),
                              R_m1=S.required_runs(sd, delta_abs, m=1), R_holm3=S.required_runs(sd, delta_abs, m=3),
                              mde_R5=S.min_detectable(sd, 5, m=3))
out = dict(best_mean_MASE=float(best_mean), delta_abs=float(delta_abs), summary=summ.round(5).to_dict(), pairs=pairs,
           shapiro_p={f"{a}-{b}": float(__import__("scipy").stats.shapiro(P[a] - P[b])[1]) for a, b in itertools.combinations(P.columns, 2)})
json.dump(out, open(C.RESULTS / "pilot.json", "w"), indent=1)
print(json.dumps(out, indent=1))
