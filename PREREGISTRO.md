# Preregistro v1 (Entregable 1)

Fecha de registro: **2026-11-03 21:02 UTC** (previo a la busqueda de hiperparametros, al piloto y a la ejecucion final).

## Problema
- Tipo de problema (Seccion 'Tipos admitidos'): **3. Series de tiempo** (el pronostico usa rezagos del objetivo).
- Dataset: Time-MMD, dominio **Energy**, objetivo `OT` (precio semanal de la gasolina, USD/galon). 1622 observaciones semanales, 1993-04-05 a 2024-04-29.
- Hash SHA-256 del archivo numerico: `94313cefb3459f04b58ec814da5d817a694e767cc52de0ce09d4de4efdad02ee`
- Entrada de los baselines: **solo la propia serie** (sin covariables ni texto). El texto se incorpora en E2/E3.
- Pregunta de investigacion: ¿cual es el error minimo que un modelo de DL basico (MLP, LSTM, CNN 1D) alcanza al pronosticar el precio semanal a 1-12 semanas y que margen de mejora debe superar un modelo avanzado/multimodal?
- Hipotesis operativa: H0: los tres modelos base tienen igual MASE esperada; H1: al menos uno difiere en mas de δ.

## Declaraciones a priori
- Unidad de independencia: origen de pronostico (ventana); las semillas se aparean por indice.
- Metrica primaria: **MASE_1:12** = promedio sobre h=1..12 de MAE_h / MAE del ingenuo h-pasos en train. Secundarias: RMSE, MAE, sMAPE; reporte por horizonte h ∈ (1, 4, 12).
- Margen de relevancia practica: **δ = 3%** de reduccion relativa de la MASE respecto del mejor modelo base.
- α = 0.05; potencia objetivo 1-β = 0.8; correccion de Holm para las 3 comparaciones pareadas.
- Comparaciones de interes (confirmatorias): MLP-LSTM, MLP-CNN1D, LSTM-CNN1D en MASE_1:12 (test). Todo lo demas es exploratorio.
- Referencias ingenuas: persistencia y ingenuo estacional (m=52); referencia de contexto: regresion lineal regularizada (Ridge) sobre la misma ventana.

## Particion cronologica (semilla no aplica: es determinista)
- Entrenamiento: indices [0, 1135) -> 1993-04-05 a 2014-12-29
- Validacion: [1135, 1378) -> 2015-01-05 a 2019-08-26
- Prueba: [1378, 1622) -> 2019-09-02 a 2024-04-29
- Las ventanas se asignan por el tiempo de sus etiquetas: ninguna etiqueta de entrenamiento cae en validacion ni prueba.
- Ventana de entrada L=26; horizonte H=12; estrategia multi-salida directa. Entrada: log(y) relativo al ultimo valor de la ventana; salida: log(y_(t+h)/y_t). Escala = desviacion estandar de la entrada en entrenamiento.
- **Hash SHA-256 de los valores del conjunto de prueba (congelado):** `e03e8f0402c53d4fcbf95e50c6f6a780aa35a6b3fd19cd5af2236aa1adab835a`
- El conjunto de prueba solo se consulta en `scripts/run_final.py` (una vez por corrida, con modelo e hiperparametros fijos).

## Presupuesto y busqueda (igual para los tres modelos)
- Busqueda aleatoria de 12 configuraciones por modelo (semilla de espacio 2026), 1 semilla de entrenamiento (0), seleccion por MASE_1:12 en validacion.
- Espacios: {"MLP": {"hidden": [32, 64, 128], "layers": [1, 2], "dropout": [0.0, 0.1, 0.2], "lr": [0.001, 0.003], "weight_decay": [0.0, 0.0001], "batch": [32, 64]}, "LSTM": {"hidden": [16, 32, 64], "dropout": [0.0, 0.1, 0.2], "lr": [0.001, 0.003], "weight_decay": [0.0, 0.0001], "batch": [32, 64], "clip": [0.5, 1.0]}, "CNN1D": {"channels": [16, 32, 64], "kernel": [3, 5], "convs": [1, 2], "dropout": [0.0, 0.1, 0.2], "lr": [0.001, 0.003], "weight_decay": [0.0, 0.0001], "batch": [32, 64]}}
- Entrenamiento: Adam, perdida MSE, max 200 epocas, early stopping (paciencia 20) en MSE de validacion, CPU, hilos=1, algoritmos deterministas.
- Piloto: semillas [100, 101, 102, 103, 104] sobre validacion con la configuracion seleccionada.
- Ejecucion final: R = max(10, R requerido por el piloto, limitado por presupuesto; si se limita se declara como limitacion).

## Trazabilidad del codigo
- SHA-256 de `src/tsbase/*.py` al momento del registro: `19680e8972fbacddd8101f7e090032dbedee9b3781c904ce97f73f5a82363b7c`
