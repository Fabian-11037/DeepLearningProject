from _nbtools import *
cells = [
("md", """# Entregable 1 — Estado del arte, EDA y baselines de DL para series de tiempo con texto

**Proyecto final de Deep Learning · Maestría en Estadística Aplicada, Universidad del Norte**

Este Jupyter Book documenta el Entregable 1 para el problema de **pronosticar el precio semanal de la gasolina a 1–12 semanas** (Time-MMD, dominio *Energy*), como punto de partida de un proyecto cuyo objetivo final es un modelo original que combine series de tiempo y texto (noticias, informes). Los tres modelos base de este entregable usan **solo la serie**: el texto entra en los Entregables 2 y 3."""),
("code", "%matplotlib inline\n" + SETUP + """
runs = pd.read_csv(ROOT / "results/final_runs.csv"); refs = pd.read_csv(ROOT / "results/reference_models.csv").set_index("model")
P = "test_MASE_1:12"; g = runs.groupby("model")[P].agg(["mean", "std"]); best = g["mean"].idxmin()
pers = refs.loc["Persistencia", P]; rid = refs.filter(like="Ridge", axis=0)[P].iloc[0]"""),
("code", """display(Markdown(f'''## Resultado principal

- **Baseline (mejor modelo base por MASE₁:₁₂ de prueba): {best}**, con MASE = **{g.loc[best,'mean']:.3f} ± {g.loc[best,'std']:.3f}** (10 semillas).
- Los tres modelos quedan en {g['mean'].min():.3f}–{g['mean'].max():.3f}; la persistencia da {pers:.3f} y una regresión **Ridge lineal** sobre la misma ventana da **{rid:.3f}**, es decir, **los tres modelos de DL mejoran a la persistencia en ≤ {100*(pers-g['mean'].min())/pers:.1f} %** y **ninguno supera a la referencia lineal**.
- Ninguna diferencia entre los tres modelos base es estadísticamente significativa en MASE₁:₁₂ tras la corrección de Holm; ver Cap. 6.
- Un modelo avanzado debe llevar la MASE₁:₁₂ de prueba por debajo de ≈ **1.56** (≥ 3.5 % por debajo del baseline) para ser estadísticamente distinguible y prácticamente relevante (δ=3 %), y debería superar también la referencia lineal.
'''))"""),
("md", """## Cumplimiento de los requisitos del Entregable 1

| Requisito | Dónde | Estado |
|---|---|---|
| Definición formal, tipo de problema, métrica primaria, δ, unidad de independencia | Cap. 1, `PREREGISTRO.md` | Completo |
| Justificación y descripción del dataset, versión y hash | Cap. 1 | Completo (fuente exacta de la serie: verificar en Apéndice C de Time-MMD) |
| EDA general y específico de series de tiempo con *decisión · impacto · riesgos* | Cap. 2 | Completo |
| Partición cronológica, prueba congelada con hash, desplazamiento de covariables | Cap. 2 (2.8, 2.9) | Completo |
| Estrategia de búsqueda, PRISMA, tabla de síntesis, calidad de estudios | Cap. 3 | **Parcial**: faltan cadenas booleanas/fechas, IEEE Xplore, motivos de exclusión, revisión por texto completo y tabla de síntesis con valores verificados |
| Extracción de los tres modelos base y análisis por modelo | Cap. 3 | Completo a nivel de frecuencia en resúmenes; confirmar con texto completo |
| Piloto (≥5 semillas), δ, tamaño de muestra, diferencia mínima detectable, preregistro fechado | Cap. 4, `PREREGISTRO.md` | Completo |
| Implementación de los tres modelos con igual esfuerzo de ajuste | Cap. 4–5 | Completo |
| ≥5 ejecuciones (≥6 con rangos), media, desv., IC | Cap. 6 | 10 semillas |
| Benchmark, ranking, baseline, ΔRMSE %, tamaño de efecto | Cap. 6 | Completo |
| Pruebas: supuestos, t pareado/Wilcoxon, Friedman+Nemenyi, Holm, DM-HLN, Ljung–Box, BDS, TOST, potencia | Cap. 6 | Completo |
| Discusión crítica y amenazas a la validez | Cap. 7 | Completo |
| Código, dependencias, semillas, configuración por corrida, scripts de datos | repositorio | Completo |"""),
("md", """```{admonition} Cómo se produjo este entregable
:class: note
El código, los análisis y los textos de este libro fueron generados con asistencia de un modelo de IA (Claude) a partir de las pautas del curso. Los resultados numéricos se calculan al ejecutar los cuadernos y los scripts. Las decisiones metodológicas (conjunto de datos, tipo de problema, δ, ventana, horizonte) y la revisión de la literatura **deben ser revisadas y asumidas por el estudiante** antes de entregar.
```"""),
]
write("00_resumen.ipynb", cells)
