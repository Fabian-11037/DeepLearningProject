from _nbtools import *
cells = [
("md", """# 4. Diseño experimental, búsqueda de hiperparámetros y estudio piloto"""),
("code", "%matplotlib inline\n" + SETUP + """
from tsbase.models import SPACES, DEFAULTS
from tsbase import stats as S
search = json.load(open(ROOT / "results/search.json")); pilot = json.load(open(ROOT / "results/pilot.json")); pdf = pd.read_csv(ROOT / "results/pilot.csv")"""),
("md", """## 4.1 Diseño

| Elemento | Valor |
|---|---|
| Entrada | ventana de 26 semanas de log-precio relativo al último valor, escalada por la desviación estándar de **entrenamiento** |
| Salida | 12 log-retornos acumulados (multi-salida directa) |
| Partición | cronológica 70/15/15; ventanas asignadas por el tiempo de sus etiquetas |
| Preprocesamiento | solo con estadísticos de entrenamiento (escala de entradas y escala de la MASE) |
| Entrenamiento | Adam, MSE, máx. 200 épocas, early stopping (paciencia 20) sobre MSE de validación, CPU, 1 hilo, algoritmos deterministas |
| Semillas | `random`, `numpy`, `torch` fijadas por corrida; `torch.use_deterministic_algorithms(True)`; determinismo estricto en CPU |
| Prueba | congelada (hash en `PREREGISTRO.md`); **una sola evaluación por corrida**, en `scripts/run_final.py` |

**Prevención de fuga:** (i) las ventanas se asignan por el tiempo de sus etiquetas; (ii) escalas ajustadas solo con entrenamiento; (iii) el módulo de datos lanza `PermissionError` si se pide la prueba sin `allow_test=True`; (iv) la búsqueda y el piloto solo usan entrenamiento/validación; (v) no se usan covariables contemporáneas ni texto."""),
("md", """## 4.2 Búsqueda de hiperparámetros (equidad)
Misma estrategia (búsqueda aleatoria sin reemplazo), mismo presupuesto (**12 configuraciones por modelo**), misma semilla de entrenamiento y mismos criterios de selección (MASE$_{1:12}$ en validación). Los espacios tienen amplitud comparable (5–7 dimensiones categóricas):"""),
("code", """rows = []
for m, sp in SPACES.items():
    rows.append({"modelo": m, "espacio": "; ".join(f"{k}∈{v}" for k, v in sp.items()), "tamaño del espacio": search[m]["space_size"], "evaluadas": len(search[m]["trials"]), "% explorado": round(100 * len(search[m]["trials"]) / search[m]["space_size"], 1)})
pd.DataFrame(rows).style.hide(axis="index")"""),
("code", """fig, ax = plt.subplots(figsize=(8, 3.2))
for i, (m, d) in enumerate(search.items()):
    v = [t["val_MASE"] for t in d["trials"]]; ax.scatter([i] * len(v), v, alpha=.7)
ax.set_xticks(range(3)); ax.set_xticklabels(search.keys()); ax.set_ylabel("MASE$_{1:12}$ validación"); ax.set_title("Resultados de la búsqueda (12 configuraciones por modelo)"); plt.show()
pd.DataFrame({m: {**d["best"], "val_MASE (1 semilla)": round(d["best_val_MASE"], 4)} for m, d in search.items()}).T.fillna("—")"""),
("md", """Las configuraciones seleccionadas se **fijan** y no se vuelven a tocar. Nota: la diferencia entre las mejores configuraciones de la búsqueda se midió con **una sola semilla**, por lo que parte de la ventaja aparente de un modelo puede ser ruido de semilla (el piloto lo cuantifica)."""),
("md", """## 4.3 Estudio piloto (5 semillas, sobre validación)
Se entrenan 5 semillas por modelo (semillas 100–104, distintas de la usada en la búsqueda) con la configuración seleccionada."""),
("code", """ps = pdf.pivot(index="seed", columns="model", values="MASE_1:12")
res = pd.DataFrame({"media": ps.mean(), "desv. estándar": ps.std(), "CV": ps.std() / ps.mean()}).round(4); display(res)
ps.round(4)"""),
("md", """### Margen δ y tamaño de muestra
El margen de relevancia práctica fijado a priori es **δ = 3 %** de reducción relativa de la MASE respecto del mejor modelo base. Con la media del mejor modelo en el piloto, eso equivale a un δ absoluto en unidades de MASE; se calcula $R\\approx\\left(\\frac{z_{1-\\alpha'/2}+z_{1-\\beta}}{\\delta}\\right)^2\\hat\\sigma_d^2$ con $\\alpha'=0.05/3$ (Bonferroni, 3 comparaciones) y potencia 0.80."""),
("code", """tab = pd.DataFrame(pilot["pairs"]).T
tab["R (α=0.05, m=1)"] = tab.pop("R_m1"); tab["R (Bonferroni m=3)"] = tab.pop("R_holm3"); tab["dif. mín. detectable con R=5 (m=3)"] = tab.pop("mde_R5")
tab["Shapiro-Wilk p (dif. pareadas)"] = pd.Series(pilot["shapiro_p"])
print(f"MASE media del mejor modelo en el piloto: {pilot['best_mean_MASE']:.4f}  →  δ absoluto = {pilot['delta_abs']:.4f}")
tab.round(4)"""),
("code", """Rneed = int(np.ceil(max(v["R_holm3"] for v in pilot["pairs"].values())))
print(f"R requerido por el piloto (máximo entre pares, Bonferroni m=3): {Rneed}")
print(f"R mínimo del enunciado (5; 6 si se usa Wilcoxon): 6   →   R definitivo = max(10 declarado a priori, {Rneed}, 6) = {max(10, Rneed, 6)}")"""),
("md", """**Lectura.** Con 5 semillas la diferencia mínima detectable es ≈0.018 (CNN1D–MLP), ≈0.024 (CNN1D–LSTM) y ≈0.036 (LSTM–MLP) de MASE, frente a δ ≈ 0.029: para el par LSTM–MLP (el más ruidoso) **no se alcanza** la potencia con 5 semillas. Por eso la ejecución definitiva se amplía a **10 semillas** (declarado en el preregistro) — lo que supera el R requerido por el piloto y el mínimo de 6 para Wilcoxon. Se verifica la normalidad de las diferencias (Shapiro–Wilk) en el Cap. 6.

**Registro:** el preregistro v1 (`PREREGISTRO.md`) está fechado **antes** de la búsqueda y del piloto; el piloto se ejecutó sobre validación y no consultó la prueba."""),
]
write("04_diseno_piloto.ipynb", cells)
