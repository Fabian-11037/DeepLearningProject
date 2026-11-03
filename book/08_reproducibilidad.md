# 8. Reproducibilidad

## Pasos

```bash
pip install -r requirements.txt
python scripts/get_data.py          # descarga Time-MMD (commit fijo) y verifica los hashes SHA-256
python scripts/make_prereg.py       # (ya ejecutado) preregistro v1, previo a búsqueda y piloto
python scripts/run_search.py        # búsqueda: 12 configuraciones aleatorias por modelo (train/val)
python scripts/run_pilot.py         # piloto: 5 semillas por modelo sobre validación
python scripts/run_final.py 10      # 10 semillas por modelo; la prueba se evalúa una vez por corrida
python scripts/run_extras.py        # curvas de aprendizaje y robustez (validación)
pytest -q tests                     # salvaguardas contra fuga y reproducibilidad
jupyter-book build book             # ejecuta los cuadernos y compila el libro
```

Todo corre en CPU (2 núcleos) en pocos minutos. El orden importa: el preregistro es anterior a la búsqueda y al piloto.

## Qué hay en el repositorio

| Ruta | Contenido |
|---|---|
| `PREREGISTRO.md` | declaraciones a priori, fechadas, con hash de la prueba congelada |
| `src/tsbase/` | datos, modelos, entrenamiento, métricas, estadística |
| `scripts/` | descarga de datos y experimentos |
| `results/` | búsqueda, piloto, corridas finales (una fila por semilla y modelo), predicciones, historiales, curvas de aprendizaje y robustez |
| `data/literature/` | conteos PRISMA y lista de los 83 estudios preseleccionados |
| `book/` | Jupyter Book |

## Semillas y configuración por corrida
- Búsqueda: semilla de entrenamiento 0, semilla del espacio 2026.
- Piloto: 100–104. Final: 1000–1009. Curvas de aprendizaje: 2000–2004.
- La configuración (hiperparámetros) de cada corrida está en `results/search.json`; cada fila de `results/final_runs.csv` registra modelo, semilla, métricas de validación y prueba, épocas, parámetros y tiempos.

## Licencia de los datos
Time-MMD no declara en su repositorio una licencia de redistribución; por eso los datos no se versionan y se obtienen con `scripts/get_data.py` (commit `00281e2d…`).
