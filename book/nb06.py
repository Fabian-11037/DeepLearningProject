from _nbtools import *
cells = [
("md", """# 6. Benchmark de los modelos base, definición del baseline e inferencia

Todas las cifras de este capítulo se calculan al ejecutar las celdas a partir de los resultados de `scripts/run_final.py` (10 semillas por modelo, prueba evaluada una sola vez por corrida)."""),
("code", "%matplotlib inline\n" + SETUP + """
import itertools
from scipy import stats as sst
from tsbase import stats as S
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import bds
runs = pd.read_csv(ROOT / "results/final_runs.csv"); refs = pd.read_csv(ROOT / "results/reference_models.csv")
M = ["MLP", "LSTM", "CNN1D"]; P = "test_MASE_1:12"
truth = np.load(ROOT / "results/preds/truth_test.npy"); org = np.load(ROOT / "results/preds/origins_test.npy")
preds = {m: np.stack([np.load(ROOT / f"results/preds/{m}_{s}.npy") for s in sorted(runs.seed.unique())]) for m in M}   # (R, N, H)
mat = lambda col: runs.pivot(index="seed", columns="model", values=col)[M]
R = runs.seed.nunique(); print("Semillas por modelo:", R, "| ventanas de prueba:", len(truth))"""),
("md", """## 6.1 Comparación tabular"""),
("code", """rows = []
for m in M:
    r = runs[runs.model == m]; x = r[P].values; lo, hi = S.t_ci(x); bl, bh = S.boot_ci(x)
    rows.append({"Modelo": m, "MASE₁:₁₂ (prueba)": f"{x.mean():.4f} ± {x.std(ddof=1):.4f}", "IC95% t": f"[{lo:.4f}, {hi:.4f}]", "IC95% bootstrap": f"[{bl:.4f}, {bh:.4f}]",
                 "RMSE₁:₁₂ (USD/gal)": f"{r['test_RMSE_1:12'].mean():.4f} ± {r['test_RMSE_1:12'].std(ddof=1):.4f}", "MAE₁:₁₂": f"{r['test_MAE_1:12'].mean():.4f} ± {r['test_MAE_1:12'].std(ddof=1):.4f}",
                 "sMAPE₁:₁₂ %": f"{r['test_sMAPE_1:12'].mean():.2f} ± {r['test_sMAPE_1:12'].std(ddof=1):.2f}",
                 "MASE val": f"{r['val_MASE_1:12'].mean():.4f} ± {r['val_MASE_1:12'].std(ddof=1):.4f}",
                 "Parámetros": int(r.n_params.iloc[0]), "Entren. (s)": f"{r.train_time_s.mean():.1f}", "Inferencia (ms/ventana)": f"{r.infer_ms_per_window.mean():.3f}"})
tab = pd.DataFrame(rows); display(tab.style.hide(axis="index"))
display(Markdown("**Referencias (deterministas, sin semillas):**"))
refs[["model", "test_MASE_1:12", "test_RMSE_1:12", "test_MAE_1:12", "test_sMAPE_1:12", "test_MASE_h1", "test_MASE_h12"]].round(4)"""),
("code", """# error por horizonte (media sobre semillas)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2)); hs = np.arange(1, C.H + 1)
sc = Windows(load_energy()[C.TARGET].values).mase_scale()
for m in M:
    mase_h = np.array([(np.abs(truth - p).mean(0) / sc) for p in preds[m]]); ax[0].errorbar(hs, mase_h.mean(0), mase_h.std(0, ddof=1), label=m, capsize=2)
for k, f in [("Persistencia", "Persistencia.npy"), ("Ridge", "Ridge.npy")]:
    ax[0].plot(hs, np.abs(truth - np.load(ROOT / "results/preds" / f)).mean(0) / sc, "--", label=k)
ax[0].set_xlabel("horizonte h (semanas)"); ax[0].set_ylabel("MASE_h (prueba)"); ax[0].legend(); ax[0].set_title("Error por horizonte (media ± desv. entre semillas)")
runs.boxplot(column=P, by="model", ax=ax[1]); ax[1].set_title("MASE₁:₁₂ por semilla"); plt.suptitle(""); plt.tight_layout(); plt.show()"""),
("md", """## 6.2 Ranking y trade-off"""),
("code", """rk = runs.groupby("model").agg(MASE=(P, "mean"), params=("n_params", "first"), t=("train_time_s", "mean")).sort_values("MASE"); rk["rango"] = range(1, 4); display(rk.round(4))
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
for m in M: ax[0].scatter(rk.loc[m, "params"], rk.loc[m, "MASE"], s=60); ax[0].annotate(m, (rk.loc[m, "params"], rk.loc[m, "MASE"]))
ax[0].set_xscale("log"); ax[0].set_xlabel("parámetros"); ax[0].set_ylabel("MASE prueba"); ax[0].set_title("Precisión vs. complejidad")
for m in M: ax[1].scatter(rk.loc[m, "t"], rk.loc[m, "MASE"], s=60); ax[1].annotate(m, (rk.loc[m, "t"], rk.loc[m, "MASE"]))
ax[1].set_xlabel("tiempo de entrenamiento (s)"); ax[1].set_title("Precisión vs. costo"); plt.tight_layout(); plt.show()"""),
("md", """## 6.3 Diferencias entre modelos base (ΔRMSE %, ΔMAE %, tamaño de efecto)
$\\Delta\\mathrm{RMSE}=\\frac{\\mathrm{RMSE}_{ref}-\\mathrm{RMSE}_{modelo}}{\\mathrm{RMSE}_{ref}}\\times100\\,\\%$ (positivo = el modelo mejora a la referencia)."""),
("code", """mr = runs.groupby("model")["test_RMSE_1:12"].mean(); ma = runs.groupby("model")["test_MAE_1:12"].mean(); mm = runs.groupby("model")[P].mean()
best, worst = mm.idxmin(), mm.idxmax(); pers = refs.set_index("model").loc["Persistencia"]
d = lambda ref, mod, s: 100 * (s[ref] - s[mod]) / s[ref]
out = []
for m in M:
    out.append({"modelo": m, "ΔRMSE% vs peor base": d(worst, m, mr), "ΔRMSE% vs mejor base": d(best, m, mr), "ΔMAE% vs peor base": d(worst, m, ma), "ΔMAE% vs mejor base": d(best, m, ma),
                "ΔRMSE% vs persistencia": 100 * (pers["test_RMSE_1:12"] - mr[m]) / pers["test_RMSE_1:12"], "ΔMASE% vs persistencia": 100 * (pers[P] - mm[m]) / pers[P]})
print(f"Mejor base por MASE₁:₁₂ de prueba: {best} | peor: {worst}"); pd.DataFrame(out).set_index("modelo").round(3)"""),
("md", """## 6.4 Verificación de supuestos y pruebas pareadas (apareadas por semilla)"""),
("code", """pairs = list(itertools.combinations(M, 2)); mt = mat(P)
fig, ax = plt.subplots(1, 3, figsize=(11, 2.8))
for a, (u, v) in zip(ax, pairs):
    sst.probplot((mt[u] - mt[v]).values, plot=a); a.set_title(f"Q-Q: {u}−{v}")
plt.tight_layout(); plt.show()
rep = {f"{u} − {v}": S.paired_report(mt[u].values, mt[v].values) for u, v in pairs}
tt = pd.DataFrame(rep).T; tt["t_p_Holm"] = S.holm(tt["t_p"].astype(float).values)
wp = tt["wilcoxon_p"].astype(float).values; tt["wilcoxon_p_Holm"] = S.holm(np.nan_to_num(wp, nan=1.0))
cols = ["mean_diff", "t_ci95", "boot_ci95", "shapiro_p", "t_p", "t_p_Holm", "wilcoxon_p", "wilcoxon_p_Holm", "cohen_dz", "cliff_delta", "prob_superiority"]
tt[cols]"""),
("md", """Diferencia = primer modelo − segundo en MASE₁:₁₂ (negativa = el primero es mejor). Se reportan **p exactos, ajustados (Holm, 3 comparaciones) y sin ajustar**, IC al 95 % (t y bootstrap), $d_z$ de Cohen pareado, $\\delta$ de Cliff y probabilidad de superioridad. Con $n=10$ pares el Wilcoxon bilateral exacto sí puede rechazar (p mínimo $2/2^{10}\\approx0.002$)."""),
("code", """fr = S.friedman_nemenyi(mt.values, M); print(f"Friedman (n={fr['n_blocks']} semillas, k=3): χ²={fr['chi2']:.3f}, p={fr['p']:.4f}")
print("Rangos medios:", {k: round(v, 2) for k, v in fr["mean_ranks"].items()}, "| Distancia crítica Nemenyi (α=.05):", round(fr["CD"], 3))
print("Pares con |Δ rango medio| > CD:", [(u, v) for u, v in pairs if abs(fr["mean_ranks"][u] - fr["mean_ranks"][v]) > fr["CD"]] or "ninguno")
vd = S.variance_decomposition(mt.values); print("Descomposición de varianza →", {k: round(v, 6) for k, v in vd.items()})"""),
("md", """**Potencia y diferencia mínima detectable con el diseño final.**"""),
("code", """sd = {f"{u} − {v}": float((mt[u] - mt[v]).std(ddof=1)) for u, v in pairs}
bm = float(mm.min()); dabs = C.DELTA_REL * bm
pd.DataFrame({"σ̂_d": sd, "δ absoluto (3% de MASE del mejor)": dabs,
  f"dif. mín. detectable (R={R}, m=3, pot. .80)": {k: S.min_detectable(v, R, m=3) for k, v in sd.items()},
  "potencia aprox. para detectar δ (m=3)": {k: float(sst.norm.cdf(abs(dabs) / (v / np.sqrt(R)) - sst.norm.ppf(1 - .05 / 3 / 2))) for k, v in sd.items()}}).round(4)"""),
("md", """Nota: δ se declaró como **3 % relativo**; su valor absoluto depende de la partición (≈0.029 en el piloto sobre validación y ≈0.049 sobre la prueba, donde la MASE es mayor).

**Equivalencia (TOST, margen δ).** La ausencia de significancia **no** es equivalencia; para afirmar que dos modelos base son equivalentes se usa TOST pareado con margen $\\delta$ (p < 0.05 ⇒ equivalencia dentro de ±δ)."""),
("code", """pd.DataFrame({f"{u} − {v}": {"TOST p": S.tost_paired((mt[u] - mt[v]).values, dabs)} for u, v in pairs}).T.round(4)"""),
("md", """## 6.5 Diebold–Mariano por horizonte (corrección HLN) y diagnóstico de residuos
Se usan los pronósticos **promediados sobre las 10 semillas** de cada modelo (un pronóstico por modelo) y pérdida cuadrática (coherente con el MSE de entrenamiento); entre paréntesis, el mismo contraste con pérdida absoluta. Convención: ΔMSE (u−v) > 0 y estadístico DM > 0 ⇒ el **segundo** modelo del par es más preciso. Por limitación de potencia, la comparación entre pronósticos promediados no equivale a comparar corridas individuales."""),
("code", """avg = {m: preds[m].mean(0) for m in M}
rows = []
for u, v in pairs:
    for h in (1, 4, 12):
        e1, e2 = (truth[:, h-1] - avg[u][:, h-1]), (truth[:, h-1] - avg[v][:, h-1])
        s2, p2 = S.dm_test(e1, e2, h=h, power=2); s1, p1 = S.dm_test(e1, e2, h=h, power=1)
        rows.append({"par": f"{u} vs {v}", "h": h, "ΔMSE (u−v)": float((e1**2).mean() - (e2**2).mean()), "DM-HLN (cuad.)": s2, "p": p2, "DM-HLN (abs.)": s1, "p (abs.)": p1})
dm = pd.DataFrame(rows); dm["p Holm (por horizonte, 3 pares)"] = np.nan
for h in (1, 4, 12):
    ix = dm.index[dm.h == h]; dm.loc[ix, "p Holm (por horizonte, 3 pares)"] = S.holm(dm.loc[ix, "p"].values)
dm.round(4)"""),
("code", """# residuos del mejor modelo base (pronostico promedio, h=1)
res1 = (truth[:, 0] - avg[best][:, 0]); resh = [(truth[:, h-1] - avg[best][:, h-1]) for h in (1, 4, 12)]
lb = acorr_ljungbox(res1, lags=[5, 10, 20], return_df=True, model_df=0)
b2 = bds(res1, max_dim=3)
print(f"Mejor base: {best} | residuos h=1 (n={len(res1)})")
display(lb.round(4)); print(f"BDS dim=2: est={b2[0][0]:.3f}, p={b2[1][0]:.4f} | dim=3: est={b2[0][1]:.3f}, p={b2[1][1]:.4f}")
print(f"Sesgo medio de residuo h=1: {res1.mean():.4f} USD (desv {res1.std():.4f}) | Shapiro p = {sst.shapiro(res1)[1]:.4g}")
fig, ax = plt.subplots(1, 3, figsize=(12, 2.9))
ax[0].plot(res1, lw=.7); ax[0].set_title("Residuo h=1 (prueba)")
from statsmodels.graphics.tsaplots import plot_acf; plot_acf(res1, lags=30, ax=ax[1]); ax[1].set_title("ACF del residuo h=1")
ax[2].scatter(avg[best][:, 0], res1, s=6); ax[2].set_xlabel("pronóstico h=1"); ax[2].set_title("Residuo vs. pronóstico (heterocedasticidad)"); plt.tight_layout(); plt.show()"""),
("md", """## 6.6 Robustez, curvas de aprendizaje y sobreajuste (solo validación)"""),
("code", """rb = pd.read_csv(ROOT / "results/robustness_val.csv"); g = rb.groupby(["model", "noise_k"]).val_MASE.agg(["mean", "std"]).unstack(0)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.1))
for m in M: ax[0].errorbar(g.index, g[("mean", m)], g[("std", m)], marker="o", label=m, capsize=2)
ax[0].set_xlabel("ruido en la entrada (múltiplo de la desv. de los retornos)"); ax[0].set_ylabel("MASE validación"); ax[0].legend(); ax[0].set_title("Robustez a ruido gaussiano")
lc = pd.read_csv(ROOT / "results/learning_curves.csv"); a = lc.groupby(["model", "n_train"])[["train_MASE", "val_MASE"]].mean()
for m, ls in zip(M, ["-", "--", ":"]):
    ax[1].plot(a.loc[m].index, a.loc[m].val_MASE, ls, c="C0", label=f"{m} val"); ax[1].plot(a.loc[m].index, a.loc[m].train_MASE, ls, c="C1", label=f"{m} train")
ax[1].set_xlabel("ventanas de entrenamiento (las más recientes)"); ax[1].set_title("Curvas de aprendizaje"); ax[1].legend(fontsize=6, ncol=2); plt.tight_layout(); plt.show()
gap = runs.groupby("model").apply(lambda r: pd.Series({"MASE train": r.train_MASE.mean(), "MASE val": r["val_MASE_1:12"].mean(), "MASE test": r[P].mean(),
   "brecha (val−train)/train %": 100 * (r["val_MASE_1:12"].mean() - r.train_MASE.mean()) / r.train_MASE.mean(), "brecha (test−val)/val %": 100 * (r[P].mean() - r["val_MASE_1:12"].mean()) / r["val_MASE_1:12"].mean()}))
gap.round(3)"""),
("md", """## 6.7 Error por segmento (prueba)"""),
("code", """dts = load_energy().index; tdates = dts[org + 1]
err = np.abs(truth - avg[best]).mean(1); ep = np.abs(truth - np.load(ROOT / "results/preds/Persistencia.npy")).mean(1)
seg = pd.DataFrame({"año": tdates.year, f"MAE {best}": err, "MAE persistencia": ep, "precio medio (USD)": truth.mean(1)}).groupby("año").mean().round(3)
seg["razón MAE modelo/persistencia"] = (seg[f"MAE {best}"] / seg["MAE persistencia"]).round(3); display(seg)
q = pd.qcut(truth.mean(1), 4, labels=["Q1 bajo", "Q2", "Q3", "Q4 alto"])
pd.DataFrame({f"MAE {best}": err, "MAE persistencia": ep, "MAE / precio medio (%)": 100 * err / truth.mean(1)}).groupby(q).mean().round(3)"""),
("md", """## 6.8 Definición del baseline y margen de mejora disponible"""),
("code", """x = runs[runs.model == best][P].values; lo, hi = S.t_ci(x); bl, bh = S.boot_ci(x)
pm = pers[P]; rid = refs.set_index("model").filter(like="Ridge", axis=0)[P].iloc[0]
mde = S.min_detectable(float(np.mean(list(sd.values()))), R, m=1)
base = pd.Series({"Baseline (mejor modelo base)": best, "MASE₁:₁₂ prueba (media ± desv.)": f"{x.mean():.4f} ± {x.std(ddof=1):.4f}", "IC95% t": f"[{lo:.4f}, {hi:.4f}]", "IC95% bootstrap": f"[{bl:.4f}, {bh:.4f}]",
  "RMSE₁:₁₂ (USD/gal)": round(float(mr[best]), 4), "MASE persistencia": round(float(pm), 4), "MASE Ridge lineal": round(float(rid), 4),
  "δ práctico (3 % de MASE) → MASE objetivo": round(float(x.mean() * (1 - C.DELTA_REL)), 4),
  "mejora mínima detectable con R=10 (aprox. 1 comparación)": round(mde, 4),
  "umbral para un modelo avanzado (el más exigente de los dos)": round(float(x.mean() - max(C.DELTA_REL * x.mean(), mde)), 4)})
display(base.to_frame("valor")); print("\\nNota: el umbral estadístico debe leerse junto con la variabilidad entre semillas; ver discusión (Cap. 7).")"""),
]
write("06_benchmark_baseline.ipynb", cells)
