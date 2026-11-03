"""Genera el preregistro v1 (previo a busqueda, piloto y ejecucion final) con fecha y hashes."""
import datetime as dt, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsbase import config as C
from tsbase.data import Windows, load_energy, energy_path, sha256
from tsbase.models import SPACES

df = load_energy(); W = Windows(df[C.TARGET].values); sp = W.split
test_bytes = np.ascontiguousarray(W.y[sp.bounds["test"][0]:]).tobytes()
src_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted((C.ROOT / "src/tsbase").glob("*.py")))).hexdigest()
now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
dates = df.index
txt = f"""# Preregistro v1 (Entregable 1)

Fecha de registro: **{now}** (previo a la busqueda de hiperparametros, al piloto y a la ejecucion final).

## Problema
- Tipo de problema (Seccion 'Tipos admitidos'): **3. Series de tiempo** (el pronostico usa rezagos del objetivo).
- Dataset: Time-MMD, dominio **Energy**, objetivo `OT` (precio semanal de la gasolina, USD/galon). {len(df)} observaciones semanales, {dates.min().date()} a {dates.max().date()}.
- Hash SHA-256 del archivo numerico: `{sha256(energy_path())}`
- Entrada de los baselines: **solo la propia serie** (sin covariables ni texto). El texto se incorpora en E2/E3.
- Pregunta de investigacion: ¿cual es el error minimo que un modelo de DL basico (MLP, LSTM, CNN 1D) alcanza al pronosticar el precio semanal a 1-12 semanas y que margen de mejora debe superar un modelo avanzado/multimodal?
- Hipotesis operativa: H0: los tres modelos base tienen igual MASE esperada; H1: al menos uno difiere en mas de δ.

## Declaraciones a priori
- Unidad de independencia: {C.INDEPENDENCE_UNIT}.
- Metrica primaria: **{C.PRIMARY_METRIC}** = promedio sobre h=1..{C.H} de MAE_h / MAE del ingenuo h-pasos en train. Secundarias: RMSE, MAE, sMAPE; reporte por horizonte h ∈ {C.HORIZONS_REPORT}.
- Margen de relevancia practica: **δ = {C.DELTA_REL:.0%}** de reduccion relativa de la MASE respecto del mejor modelo base.
- α = {C.ALPHA}; potencia objetivo 1-β = {C.POWER}; correccion de Holm para las 3 comparaciones pareadas.
- Comparaciones de interes (confirmatorias): MLP-LSTM, MLP-CNN1D, LSTM-CNN1D en MASE_1:12 (test). Todo lo demas es exploratorio.
- Referencias ingenuas: persistencia y ingenuo estacional (m={C.SEASONAL_PERIOD}); referencia de contexto: regresion lineal regularizada (Ridge) sobre la misma ventana.

## Particion cronologica (semilla no aplica: es determinista)
- Entrenamiento: indices [0, {sp.n_tr}) -> {dates[0].date()} a {dates[sp.n_tr-1].date()}
- Validacion: [{sp.n_tr}, {sp.n_tr+sp.n_va}) -> {dates[sp.n_tr].date()} a {dates[sp.n_tr+sp.n_va-1].date()}
- Prueba: [{sp.n_tr+sp.n_va}, {sp.n}) -> {dates[sp.n_tr+sp.n_va].date()} a {dates[-1].date()}
- Las ventanas se asignan por el tiempo de sus etiquetas: ninguna etiqueta de entrenamiento cae en validacion ni prueba.
- Ventana de entrada L={C.L}; horizonte H={C.H}; estrategia multi-salida directa. Entrada: log(y) relativo al ultimo valor de la ventana; salida: log(y_(t+h)/y_t). Escala = desviacion estandar de la entrada en entrenamiento.
- **Hash SHA-256 de los valores del conjunto de prueba (congelado):** `{hashlib.sha256(test_bytes).hexdigest()}`
- El conjunto de prueba solo se consulta en `scripts/run_final.py` (una vez por corrida, con modelo e hiperparametros fijos).

## Presupuesto y busqueda (igual para los tres modelos)
- Busqueda aleatoria de {C.SEARCH_BUDGET} configuraciones por modelo (semilla de espacio {C.SPACE_SEED}), 1 semilla de entrenamiento ({C.SEARCH_SEED}), seleccion por MASE_1:12 en validacion.
- Espacios: {json.dumps(SPACES)}
- Entrenamiento: Adam, perdida MSE, max {C.MAX_EPOCHS} epocas, early stopping (paciencia {C.PATIENCE}) en MSE de validacion, CPU, hilos=1, algoritmos deterministas.
- Piloto: semillas {C.PILOT_SEEDS} sobre validacion con la configuracion seleccionada.
- Ejecucion final: R = max({C.FINAL_SEEDS_MIN}, R requerido por el piloto, limitado por presupuesto; si se limita se declara como limitacion).

## Trazabilidad del codigo
- SHA-256 de `src/tsbase/*.py` al momento del registro: `{src_hash}`
"""
(C.ROOT / "PREREGISTRO.md").write_text(txt, encoding="utf-8")
print(txt[:600])
