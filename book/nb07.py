from _nbtools import *
cells = [
("md", """# 7. Discusión crítica"""),
("code", "%matplotlib inline\n" + SETUP + """
from tsbase import stats as S
runs = pd.read_csv(ROOT / "results/final_runs.csv"); refs = pd.read_csv(ROOT / "results/reference_models.csv").set_index("model")
M = ["MLP", "LSTM", "CNN1D"]; P = "test_MASE_1:12"; mt = runs.pivot(index="seed", columns="model", values=P)[M]
mm = mt.mean(); best = mm.idxmin(); pers = refs.loc["Persistencia"]; rid = refs.filter(like="Ridge", axis=0).iloc[0]
h = lambda m, k: runs[runs.model == m][f"test_MASE_h{k}"].mean()
rb = pd.read_csv(ROOT / "results/robustness_val.csv").groupby(["model", "noise_k"]).val_MASE.mean().unstack(0)
gap = {m: 100 * (runs[runs.model == m]["val_MASE_1:12"].mean() - runs[runs.model == m].train_MASE.mean()) / runs[runs.model == m].train_MASE.mean() for m in M}
pv = {f"{u}-{v}": S.paired_report(mt[u].values, mt[v].values) for u, v in [("MLP", "LSTM"), ("MLP", "CNN1D"), ("LSTM", "CNN1D")]}
minp = min(S.holm([r["t_p"] for r in pv.values()]))"""),
("code", """display(Markdown(f'''### ¿Cuál modelo base funciona mejor y por qué?
**{best}** tiene la menor MASE₁:₁₂ de prueba ({mm[best]:.3f}; MLP {mm['MLP']:.3f}, LSTM {mm['LSTM']:.3f}), pero la diferencia con los otros es de {100*(mm.max()-mm.min())/mm.max():.1f} % y **no es estadísticamente significativa** (p-valor t pareado más pequeño tras Holm = {minp:.2f}; Friedman p = 0.30). A horizonte h=1 el DM-HLN sí favorece a la CNN 1D frente al MLP (p ajustado ≈ 0.012); a h=4 y h=12 no hay diferencias. Una lectura plausible (a verificar) es que, a un paso, la CNN capta mejor los últimos retornos (autocorrelación corta, rezagos 1–3 significativos en el EDA), mientras que a 12 semanas todos convergen a un pronóstico cercano a la persistencia.

### ¿Cuál generaliza mejor?
La brecha entrenamiento→validación es {gap['CNN1D']:.1f} % (CNN1D), {gap['LSTM']:.1f} % (LSTM) y {gap['MLP']:.1f} % (MLP): el MLP sobreajusta menos. En robustez a ruido en la entrada (validación), con ruido de 2 desv. de los retornos la MASE sube de {rb.loc[0,'CNN1D']:.3f} a {rb.loc[2,'CNN1D']:.3f} (CNN1D, +{100*(rb.loc[2,'CNN1D']/rb.loc[0,'CNN1D']-1):.0f} %), de {rb.loc[0,'LSTM']:.3f} a {rb.loc[2,'LSTM']:.3f} (LSTM, +{100*(rb.loc[2,'LSTM']/rb.loc[0,'LSTM']-1):.0f} %) y de {rb.loc[0,'MLP']:.3f} a {rb.loc[2,'MLP']:.3f} (MLP, +{100*(rb.loc[2,'MLP']/rb.loc[0,'MLP']-1):.0f} %): **el modelo con mejor media de prueba es el más frágil**.

### ¿Son las diferencias estadísticamente significativas y prácticamente relevantes?
No, en la métrica primaria. Las diferencias observadas entre modelos (≈0.001–0.014 de MASE) son menores que δ (≈0.049 en la escala de prueba) y menores que la diferencia mínima detectable de los tres pares con 10 semillas (0.041–0.081; Cap. 6.4). TOST sí muestra equivalencia dentro de ±δ para MLP–LSTM y MLP–CNN1D, pero no para LSTM–CNN1D (la LSTM es la más variable entre semillas). La varianza entre semillas es la fuente dominante de variabilidad frente a la varianza entre modelos.

### ¿Hay trade-off entre desempeño y costo?
No se observa: el modelo con más parámetros ({runs[runs.model=='CNN1D'].n_params.iloc[0]:,}) es el mejor por poco; la LSTM tiene menos parámetros ({runs[runs.model=='LSTM'].n_params.iloc[0]:,}) pero es la más lenta de entrenar ({runs[runs.model=='LSTM'].train_time_s.mean():.1f} s frente a {runs[runs.model=='MLP'].train_time_s.mean():.1f} s del MLP) y no mejora. **Mayor complejidad no se traduce en mejor desempeño** en esta serie univariada.

### Hallazgo central: los modelos de DL apenas mejoran las referencias simples
- Frente a la **persistencia** (MASE {pers[P]:.3f}), la mejora de los tres modelos es de {100*(pers[P]-mm.max())/pers[P]:.1f} %–{100*(pers[P]-mm.min())/pers[P]:.1f} %, menor que δ = 3 %.
- A **h=1**, {best} da MASE {h(best,1):.3f} frente a {pers['test_MASE_h1']:.3f} de la persistencia (mejora clara), pero a **h=12** da {h(best,12):.3f} frente a {pers['test_MASE_h12']:.3f} (**peor** que la persistencia).
- La regresión **Ridge lineal** (MASE {rid[P]:.3f}) es **mejor que los tres modelos de DL**, y a h=1 ({rid['test_MASE_h1']:.3f}) también supera a la CNN 1D ({h(best,1):.3f}). Es un resultado coherente con la literatura que cuestiona la ganancia de modelos complejos sobre modelos simples en pronóstico, aunque aquí es una observación sobre un solo dataset.

### Residuos: margen de mejora estructural
Los residuos a h=1 de la CNN 1D presentan autocorrelación (Ljung–Box p ≤ 0.004 en los rezagos 5, 10 y 20), dependencia no lineal (BDS, p < 0.002) y no normalidad marcada (Shapiro–Wilk p ≈ 1e-14). Queda estructura sin explicar, lo que sugiere margen para modelos que incorporen información adicional (covariables, texto) o heterocedasticidad.

### Umbral y margen de mejora disponible
Para considerarse una mejora **estadísticamente distinguible y prácticamente relevante**, un modelo avanzado debería lograr en la prueba una MASE₁:₁₂ ≲ **1.56** (≈ 3.5 % por debajo del baseline), que es el más exigente entre δ = 3 % y la mejora mínima detectable con 10 semillas. Dado que Ridge ya está en {rid[P]:.3f}, **el umbral realista es superar también a Ridge**. La brecha reportada en la literatura entre modelos avanzados y modelos base (reducciones de MSE de 15–40 % en Time-MMD al añadir texto, según los trabajos revisados) **no es verificable con estos resultados** y debe contrastarse con la comparación del E2.'''))"""),
("md", """## Amenazas a la validez

**Internas**
- La búsqueda de hiperparámetros usó **una sola semilla** por configuración (igual presupuesto para los tres modelos); parte de la ventaja aparente en la búsqueda es ruido de semilla.
- El «mejor modelo base» se elige por la media de prueba, como pide el enunciado, aunque la diferencia entre modelos está dentro del ruido; por eso el baseline debe leerse como **la familia de los tres modelos** (≈1.62–1.63), no como un ganador claro.
- Las pruebas DM se aplican sobre pronósticos promediados entre semillas; las ventanas de prueba se solapan y sus errores son autocorrelacionados (se corrige con HLN, pero la potencia es limitada).
- Una única partición cronológica; el diseño no cuantifica la variabilidad por partición.

**Externas**
- Un solo dataset y una sola serie; el periodo de prueba incluye regímenes extremos (2020, 2022) que no están en el entrenamiento (termina en 2014).
- La **MASE usa como escala el error del ingenuo en entrenamiento (en USD)**; en la prueba, con precios más altos, los errores en USD crecen. El MAE sube con el nivel del precio (cuartil bajo ≈ 0.165 frente a cuartil alto ≈ 0.296 USD), mientras que el error relativo a precio se mantiene en 5.6–7.3 % en los cuatro cuartiles. Esto explica en buena medida que la MASE de prueba (≈1.6) sea mucho mayor que la de validación (≈0.97) sin que el modelo «empeore» en términos relativos. **La MASE es comparable entre modelos dentro de una partición, no entre particiones.**

**De construcción**
- MASE₁:₁₂ promedia horizontes de dificultad distinta; RMSE es sensible a colas pesadas; sMAPE tiene problemas cerca de cero (no ocurre aquí, precios > 0.9).

## Limitaciones
- La revisión de literatura es preliminar (Cap. 3): faltan cadenas de búsqueda, IEEE Xplore, texto completo y tabla de síntesis verificada.
- Baselines solo con la serie; no se evalúa texto ni covariables.
- Ventana L = 26 no cubre un ciclo anual completo.

## Recomendaciones para E2 y E3
1. Incluir **Ridge/AR** y **persistencia** como referencias obligatorias del benchmark; un modelo que no supere a Ridge no justifica su complejidad.
2. Evaluar una escala de error **independiente del nivel** (p. ej., MASE con escala móvil, error porcentual o log-retornos) además de la MASE actual.
3. Probar L = 52 como variante y covariables causales; luego añadir texto con auditoría de fuga (campo `preds`).
4. Aumentar la potencia: más semillas y validación con origen móvil para obtener varias realizaciones de régimen.
5. Evaluar en los tres datasets adicionales de naturaleza distinta exigidos en el E3, fijados antes de ver resultados."""),
]
write("07_discusion.ipynb", cells)
