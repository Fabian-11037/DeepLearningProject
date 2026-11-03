# Entregable 1 · Series de tiempo + texto (Time-MMD *Energy*)

Proyecto final de Deep Learning (Maestría en Estadística Aplicada, Universidad del Norte).
Baselines de DL (MLP, LSTM, CNN 1D) para pronosticar el precio semanal de la gasolina a 1–12 semanas, EDA, protocolo estadístico y Jupyter Book.

- Libro: `jupyter-book build book` → `book/_build/html/index.html`
- Preregistro fechado: `PREREGISTRO.md`
- Pasos de reproducción: `book/08_reproducibilidad.md`

**Resultado principal:** los tres modelos quedan en MASE₁:₁₂ de prueba ≈ 1.62–1.63 (persistencia 1.636; Ridge lineal 1.579). Sin diferencias significativas entre ellos.

**Pendiente del estudiante:** completar la revisión de literatura (Cap. 3: cadenas de búsqueda, IEEE Xplore, texto completo, tabla de síntesis con valores verificados).

*Nota:* los cuadernos fuente se generan con `book/nb0*.py` (`python book/nb02.py`, etc.); el código y los textos se produjeron con asistencia de IA y deben ser revisados por el autor.
