# Fichas de datasets multimodales candidatos (verificadas el 2026-11-03)

Fuentes primarias consultadas ese día. "No verificado" = no apareció en la fuente leída; confirmar antes de usarlo.

## FNSPID (finanzas)
- **Paper:** FNSPID: A Comprehensive Financial News Dataset in Time Series. arXiv:2402.06698 — https://arxiv.org/abs/2402.06698v1
- **Tamaño:** 15.7 millones de noticias alineadas en el tiempo y 29.7 millones de precios; 4,775 empresas, todas miembros del S&P 500.
- **Fechas:** 1999–2023 (noticias y precios).
- **Fuente del texto:** cuatro sitios web de noticias bursátiles (el resumen no los nombra; la herramienta de scraping del repo recolecta de Nasdaq).
- **Alojamiento:** Hugging Face `Zihan1004/FNSPID` (enlazado desde el README de GitHub `Zdong104/FNSPID_Financial_News_Dataset`).
- **Licencia: AMBIGUA.** El README de GitHub dice que se liberan los derechos de uso comercial y de investigación, y en otra sección prohíbe el uso comercial del *código* sin autorización. No se pudo leer la tarjeta del dataset en Hugging Face (fuera de la lista de acceso) ni el archivo LICENSE. El repo está sin mantenimiento desde 07/2025. El texto proviene de scraping de sitios de terceros: revisar sus términos.

## NSW-EPNews (electricidad, Australia)
- **Paper:** NSW-EPNews: A News-Augmented Benchmark for Electricity Price Forecasting with LLMs. arXiv:2506.11050 — https://arxiv.org/abs/2506.11050v1
- **Mercado:** Nueva Gales del Sur, Mercado Nacional de Electricidad (NEM), Australia.
- **Tamaño:** más de 175,000 precios spot semihorarios; ≈3.6 mil artículos; más de 3,600 pares de registros (a nivel día). El artículo y los registros pueden diferir en número.
- **Fechas:** 2015–2024 (diez años). Precios cada 30 min hasta el 1-oct-2021 y cada 5 min después; los autores reducen la frecuencia a 30 min con la mediana de cada ventana de seis registros.
- **Fuentes:** precios de AEMO; noticias de WattClarity (scraping de su archivo); temperaturas diarias (fuente no nombrada en el texto leído). GPT-4o resume cada artículo y le asigna metadatos, entre ellos un nivel de impacto.
- **Alojamiento:** figshare (enlace del paper: https://figshare.com/s/e25f3a98679d347f2a2e).
- **Licencia: NO VERIFICADA.** El paper no la declara en el texto leído (el CC BY 4.0 de la página es del artículo). Los textos son de WattClarity: revisar derechos.

## MoTime (suite de ocho datasets)
- **Paper:** MoTime: A Dataset Suite for Multimodal Time Series Forecasting. arXiv:2505.15072 — https://arxiv.org/abs/2505.15072v1
- **Dominios:** comercio electrónico, tráfico web, medios y comportamiento de usuarios. **No es de energía ni finanzas.**
- **Datasets (forma de la serie tal como la imprime la Tabla 2; el paper no rotula cuál eje es nº de series y cuál longitud):** PixelRec 4,865 × 43,082; TaobaoFashion 365 × 890; MovieLens 10,505 × 84,518; AmazonReview 3,934 × 668,756; Tianchi 365 × 36,397; News 144 × 26,612; WikiPeople 550 × 3,856; VISUELLE 11 × 5,355 (irregular, sin experimentos).
- **Frecuencia y fechas:** mayoría diaria. PixelRec 2020-03-05 → 2022-09-12; TaobaoFashion 2014-06-15 → 2015-06-14; MovieLens 1995-01-09 → 2023-10-13; AmazonReview 2013-01-01 → 2023-10-01; News: 144 valores cada 20 min durante las primeras 48 h tras la publicación; WikiPeople vistas diarias. Tianchi: un año, sin fechas.
- **Texto:** metadatos asociados a cada serie por identificador (títulos, descripciones, géneros, etiquetas, resúmenes de Wikipedia, titulares de noticias). Es **texto estático por ítem**, no un flujo de noticias que cambia en el tiempo, salvo el subconjunto News. El paper no da fechas de captura del texto externo.
- **Alojamiento:** Kaggle `krissssss/multimodal-time-series-forecasting` (no se pudo abrir la página).
- **Licencia: NO VERIFICADA.** El CC BY 4.0 visible es del artículo. Los datos de origen (Amazon Reviews 2023, Tianchi, MovieLens) tienen términos propios.
- **Inconsistencias del paper:** las tablas 1 y 2 y la sección 8 dan cifras distintas para algunos totales; usar con cautela hasta verificar los archivos.

## Lectura para el proyecto
- **FNSPID:** el más grande y con texto fechado; la licencia es el mayor riesgo, y la literatura reporta ganancias pequeñas del texto en finanzas.
- **NSW-EPNews:** pequeño en artículos (≈3.6 mil) pero con fechas claras y mercado de energía; adecuado como dataset adicional del E3 si la licencia lo permite.
- **MoTime:** poco adecuado como fuente de *noticias* para pronóstico (texto mayormente estático por ítem y de dominios de comercio/medios); podría servir como dataset adicional de naturaleza distinta, no como principal.
