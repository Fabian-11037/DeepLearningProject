# Contexto del proyecto (para cualquier agente o persona que continúe el trabajo)

Idioma de trabajo: **español**. Autor: Fabian José Manrique Morelos (Maestría en Estadística Aplicada, Universidad del Norte).

## 1. Objetivo
Proyecto final del curso de Deep Learning. Meta global: un **modelo original y novedoso que combine modelos de series de tiempo con NLP** (variables temporales + texto: noticias, informes, foros) para mejorar predicciones temporales. Son tres entregables encadenados; el dataset, la partición, las métricas y el δ definidos en el E1 se heredan en E2 y E3.

| Entregable | Qué pide (resumen) | Estado |
|---|---|---|
| **E1** Estado del arte, EDA y 3 baselines | EDA exhaustivo, extracción de 3 modelos base desde la literatura, piloto, ≥5 semillas, benchmark, baseline e inferencia, Jupyter Book | **Hecho, con la revisión de literatura parcial** |
| **E2** Benchmark de 10 modelos | Top-10 + baselines, replicar un paper guía e identificar su limitación, preregistro, auditoría de fuga, búsqueda de 1000 arquitecturas en dos fases (1000 con 1 semilla y 20–30 % de épocas → k≥10 finalistas con ≥5 semillas), ablación (≥3 componentes), robustez (≥3 niveles), potencia, Friedman/Holm/MCS/DM | Pendiente |
| **E3** Modelo original | R1: nuevo (mecanismo/pérdida/entrenamiento, no solo aplicación); R2: implementación propia (sin `from_pretrained` ni código ajeno); R3: superior y significativo frente a todo el Top-10 y baselines fuertes (≥3 pruebas por contraste, Holm p<0.05, |Cliff δ|≥0.147, potencia 0.90, consistente en ≥2 de 3 datasets adicionales de naturaleza distinta, elegidos antes de ver resultados), demostración formal de una propiedad, dashboard público, informe, sustentación | Pendiente |

Los PDF de pautas (`Entregable{1,2,3}DeepLearningProject.pdf`) están en el proyecto de claude.ai "DL Project". Topes de nota del E3: 5.0 si R1+R2+R3, 4.0 si R1+R2 sin R3, 3.0 si falla R1 o R2.

## 2. Decisiones ya tomadas (E1) — no cambiar sin avisar, se heredan
- **Dataset:** Time-MMD (Liu et al. 2024, arXiv:2406.08627), dominio **Energy**, objetivo `OT` (precio semanal de gasolina, USD/gal), 1,622 semanas (1993-04-05 → 2024-04-29). Commit fijo `00281e2d…`; hashes SHA-256 en `scripts/get_data.py`. Time-MMD no declara licencia de redistribución: **los datos no se versionan**.
- **Tipo de problema:** **Series de tiempo** (usa rezagos del objetivo), no regresión temporal.
- **Baselines solo-serie** (sin covariables ni texto): MLP, LSTM de 1 capa, CNN 1D poco profunda. El texto entra en E2/E3.
- **Ventana L=26, horizonte H=12**, multi-salida directa. Entrada: log-precio relativo al último valor, escalada con la desviación de entrenamiento; salida: log(y_{t+h}/y_t).
- **Partición cronológica 70/15/15** (train 1993-04 → 2014-12; val 2015-01 → 2019-08; test 2019-09 → 2024-04). Ventanas asignadas por el tiempo de sus etiquetas.
- **Métrica primaria:** MASE₁:₁₂ (media sobre h=1..12 de MAE_h / MAE del ingenuo h-pasos en train). Secundarias: RMSE, MAE, sMAPE, y h∈{1,4,12}. **δ = 3 %** relativo. α=0.05, potencia 0.80, Holm con 3 comparaciones. Unidad de independencia: origen de pronóstico; semillas apareadas por índice.
- Presupuesto igual para los tres modelos: 12 configuraciones aleatorias (1 semilla), piloto con semillas 100–104, final con 10 semillas (1000–1009).
- Todo está fechado en `PREREGISTRO.md`.

## 3. Resultados del E1 (valores de referencia)
- MASE₁:₁₂ de prueba: CNN1D **1.618 ± 0.034** (baseline), MLP 1.632, LSTM 1.632; persistencia 1.636; ingenuo estacional 6.935; **Ridge lineal 1.579** (mejor que los tres de DL).
- Ninguna diferencia entre los tres es significativa tras Holm. DM-HLN solo favorece a CNN1D sobre MLP a h=1.
- A h=12 los modelos son peores que la persistencia. Residuos con autocorrelación y no linealidad (margen de mejora).
- **Umbral para modelos avanzados:** MASE₁:₁₂ de prueba ≲ 1.56 y superar también a Ridge.
- Caveat: la MASE usa escala en USD de train; en la prueba (precios altos) el error en USD crece, así que la MASE de prueba (≈1.6) no es comparable con la de validación (≈0.97). Comparar modelos solo dentro de una misma partición. Para E2/E3 conviene una escala independiente del nivel.

## 4. Reglas que NO se deben romper
1. **El conjunto de prueba está congelado** (hash en `PREREGISTRO.md`). Solo `scripts/run_final.py` lo consulta (`allow_test=True`). No usarlo para búsqueda, piloto, diseño ni EDA de decisión.
2. Escalas y transformaciones se ajustan **solo con train**. Nada de datos futuros en el preprocesamiento.
3. Los preregistros se escriben **antes** de buscar/piloto/ejecutar y se versionan (E2 y E3 necesitan los suyos).
4. Mismo esfuerzo de búsqueda para todos los modelos (|espacio|, k, R).
5. No interpretar «no significativo» como equivalencia (usar TOST/ROPE). Reportar p exactos, ajustados y sin ajustar, IC y tamaño de efecto.
6. Antes de afirmar algo sobre literatura, verificarlo en el texto completo. No inventar referencias ni DOI.
7. Los textos del libro con cifras deben salir de celdas ejecutadas, no copiarse a mano.

## 5. Estructura del repositorio
```
PREREGISTRO.md          declaraciones a priori v1 (fecha, hash de la prueba)
src/tsbase/             config.py (constantes a priori), data.py (ventanas, particion, guarda de prueba),
                        models.py (MLP/LSTM/CNN1D + espacios), train.py, metrics.py, stats.py (DM-HLN, Holm, TOST, ...)
scripts/                get_data, make_prereg, run_search, run_pilot, run_final, run_extras
results/                search.json, pilot.{csv,json}, final_runs.csv, reference_models.csv, preds/, hist/,
                        learning_curves.csv, robustness_val.csv   (results/models/ esta ignorado por git)
data/literature/        conteos PRISMA y lista de 83 estudios preseleccionados
book/                   Jupyter Book 1.x: nb00..nb07.py GENERAN los .ipynb (editar el .py y correrlo), 08_reproducibilidad.md
tests/                  pytest: fuga, acceso a prueba, escala, causalidad, reproducibilidad
```
Los cuadernos `.ipynb` se generan con `python book/nb0X.py`; para cambiar un capítulo edita el `.py` y regenera.

## 6. Cómo ejecutar en otro PC
```bash
pip install -r requirements.txt          # jupyter-book debe ser 1.0.4 (Sphinx); la v2 usa otro CLI
python scripts/get_data.py               # baja Time-MMD y verifica hashes
python scripts/run_search.py && python scripts/run_pilot.py && python scripts/run_final.py 10 && python scripts/run_extras.py
pytest -q tests
jupyter-book build book                  # ejecuta los cuadernos; salida en book/_build/html
```
CPU basta (≈ minutos). Los resultados son deterministas (`torch.use_deterministic_algorithms(True)`, 1 hilo).
Gotchas: en el contenedor de nube, PyTorch de PyPI exige instalar también las librerías `nvidia-*`/`cuda-toolkit` (aunque no haya GPU); `pip install torch` completo lo resuelve. `make_prereg.py` sobrescribe `PREREGISTRO.md`: **no volver a correrlo** para E1 (rompería la fecha del preregistro).

## 7. Pendientes y siguientes pasos
**Pendiente del E1 (responsabilidad del estudiante):** completar el Cap. 3 (cadenas booleanas con fecha, IEEE Xplore, motivos de exclusión por registro, segundo revisor, lectura de texto completo, tabla de síntesis con referencia/problema/dataset/modelos base/métricas/valores y DOI, brecha de desempeño avanzados vs base). Verificar la fuente de la serie en el Apéndice C del paper de Time-MMD. Revisar las decisiones metodológicas.

**E2 (siguiente):** elegir paper guía y su limitación; definir el Top-10 (incluir Ridge/AR y persistencia como referencias); auditoría de fuga del texto (campo `preds` de Time-MMD sospechoso: son proyecciones narrativas; usar solo texto con `end_date` anterior al origen); preregistro v2; implementar búsqueda de 1000 arquitecturas.

**Ideas de modelo original (E3), de la revisión de literatura** (por prioridad; confirmar novedad con búsqueda dirigida de 5–10 trabajos cercanos, p. ej. AIR, TGForecaster, TaTS, Time-MMD/MM-TSFlib):
1. Compuerta de utilidad del texto guiada por incertidumbre de la serie (ŷ = f_ts + g(u)·h(texto)), con propiedad de «no empeorar» respecto al modelo solo-serie.
2. Pérdida de **texto placebo** (penaliza que el texto no informativo cambie la predicción; anti-fuga/ruido). Se recomienda combinar 1+2.
3. Núcleo de respuesta al impulso de noticias (decaimiento aprendible por horizonte).
4. Pronóstico probabilístico/conformal condicionado por texto.
No basta apilar/promediar salidas ni «añadir una regresión»: el E3 exige regla de fusión o mecanismo propio, hipótesis falsable y ablación.

**Datasets adicionales para E3** (misma tarea y métrica, naturaleza distinta, ≥2 ejes de diferencia, fijados antes de ver resultados): otros dominios de Time-MMD (Environment diario, Agriculture/Economy/Health mensual o semanal, etc.), NSW-EPNews, subconjunto de FNSPID o FX/petróleo (verificar licencia). Fijar la tarea como pronóstico de valor (no clasificación).

## 8. Antecedentes de la revisión de literatura
Hay un documento «Estado del arte: Series de tiempo + NLP (arquitecturas)» (Claude Docs) y un doc en el proyecto de claude.ai «DL Project» (`claude/estado-del-arte-series-texto.md`). Familias: A) LLM/pretrained adaptado a series (Time-LLM, OFA, TimeCMA…); B) fusión explícita de texto (Time-MMD/MM-TSFlib, TaTS, TGForecaster, SpecTF, AIR, Dual-Forecaster…); C) noticias/sentimiento + RNN/Transformer en finanzas (ganancias pequeñas e inconsistentes); D) texto como covariable. Evidencia clave: Time-MMD reporta −15 % MSE promedio al añadir texto (hasta −40 %); Tan et al. (NeurIPS 2024) muestran que quitar el LLM iguala o supera a varios modelos con LLM.

## 9. Git y accesos
Repositorio: `github.com/Fabian-11037/DeepLearningProject` (rama `main`). Commits de Claude llevan el trailer `Co-Authored-By`. Los textos y el código del E1 se produjeron con asistencia de IA y deben ser revisados por el autor.
