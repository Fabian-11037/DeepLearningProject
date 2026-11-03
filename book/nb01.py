from _nbtools import *
cells = [
("md", """# 1. Definición del problema y dataset"""),
("code", "%matplotlib inline\n" + SETUP),
("md", """## 1.1 Definición formal

- **Tipo de problema (de los diez admitidos): 3. Series de tiempo.** El pronóstico usa los rezagos del propio objetivo (ventana de entrada), por lo que, según la definición del enunciado, no es regresión temporal.
- **Espacio de entrada** $\\mathcal{X}=\\mathbb{R}^{L}$ con $L=26$: los últimos 26 valores de $\\log y$ *relativos al último* ($x_j=\\log y_{t-L+1+j}-\\log y_t$).
- **Espacio de salida** $\\mathcal{Y}=\\mathbb{R}^{H}$ con $H=12$: $y_h=\\log(y_{t+h}/y_t)$, $h=1,\\dots,12$. El pronóstico en nivel es $\\hat y_{t+h}=y_t\\,e^{\\hat y_h}$.
- **Función objetivo** $f:\\mathcal{X}\\to\\mathcal{Y}$, estrategia **multi-salida directa**; pérdida $\\mathcal{L}=\\frac{1}{NH}\\sum_{i,h}(y_{i,h}-f(x_i)_h)^2$ sobre los valores escalados.
- **Pregunta de investigación:** ¿qué error mínimo alcanzan un MLP, una LSTM y una CNN 1D al pronosticar el precio semanal de la gasolina a 1–12 semanas, y qué margen debe superar un modelo avanzado (multimodal con texto) para justificar su complejidad?
- **Hipótesis operativa:** $H_0$: los tres modelos tienen igual MASE esperada; $H_1$: alguno difiere en más de $\\delta$.
- **Métrica primaria:** MASE$_{1:12}$ (promedio de la MASE por horizonte; escala = MAE del ingenuo $h$-pasos en entrenamiento). **Secundarias:** RMSE, MAE, sMAPE, y los tres horizontes $h\\in\\{1,4,12\\}$.
- **Margen de relevancia práctica:** $\\delta=3\\,\\%$ de reducción relativa de la MASE respecto del mejor modelo base.
- **Unidad de independencia:** el origen de pronóstico; las semillas se aparean por índice.

Estas declaraciones están fechadas y versionadas en `PREREGISTRO.md`, y se heredan en los Entregables 2 y 3."""),
("md", """## 1.2 Justificación del dataset

- **Relevancia aplicada:** el precio de la gasolina es una serie con impacto económico, sensible a eventos (geopolítica, huracanes, decisiones de la OPEP) que se describen en texto; por eso es un banco natural para modelos que combinan serie y texto.
- **Procedencia:** Time-MMD (Liu et al., 2024, arXiv:2406.08627) agrupa 9 dominios con series numéricas y texto alineado. En *Energy*, la serie es un precio semanal de gasolina minorista en EE. UU. y el texto son informes semanales y búsquedas (la fuente exacta de cada serie está en el Apéndice C del paper; **pendiente de verificar y citar**).
- **Uso previo en la literatura:** es el benchmark de referencia de la línea multimodal que se revisa en el Cap. 3 (Time-MMD / MM-TSFlib y trabajos posteriores), lo que permite comparar resultados.
- **Licencia:** el repositorio no declara una licencia de redistribución, por eso los datos **no se versionan aquí**; `scripts/get_data.py` los descarga de un commit fijo y verifica el hash."""),
("code", """df = load_energy()
info = {
 "Observaciones": len(df), "Inicio": str(df.index.min().date()), "Fin": str(df.index.max().date()),
 "Frecuencia": "semanal (lunes)", "Variable objetivo": C.TARGET,
 "Columnas numéricas": df.shape[1], "Archivo": str(energy_path().relative_to(ROOT)),
 "SHA-256": sha256(energy_path()),
 "Commit Time-MMD": "00281e2d86058286d5548b15a7670e8eda57ef62 (14-ene-2025)"}
pd.Series(info).to_frame("valor")"""),
("code", """cols = pd.DataFrame({"columna": df.columns, "tipo": df.dtypes.astype(str).values, "faltantes": df.isna().sum().values})
cols["columna"] = cols["columna"].str.replace(r"\\s+", " ", regex=True).str.slice(0, 90)
cols"""),
("md", """## 1.3 Descripción técnica

- **Unidad de observación:** una semana; **variable objetivo** `OT`: precio promedio semanal (USD/galón). Es una **medición instrumental/administrativa** (encuesta de precios), con ruido de medición pequeño frente a la variación del precio.
- **Estructura de dependencia:** temporal (autocorrelación fuerte, no estacionaria en nivel). Esto obliga a una **partición cronológica estricta** y a que la inferencia estadística tenga en cuenta que los errores de pronóstico a horizonte $h$ están autocorrelacionados (MA($h-1$)), de ahí el uso de Diebold–Mariano con corrección HLN.
- **Covariables numéricas:** las otras 9 columnas son precios regionales contemporáneos del mismo producto. **No se usan** en los baselines: están medidas en el mismo instante que el objetivo y mezclarlas sería casi una copia del objetivo, además de que su disponibilidad al momento del pronóstico no está garantizada (riesgo de fuga). Los baselines son **solo-serie**.
- **Texto:** informes semanales (`Energy_report.csv`) y búsquedas (`Energy_search.csv`). No entran en el E1.""")]
write("01_problema_dataset.ipynb", cells)
