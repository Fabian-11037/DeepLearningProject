"""Declaraciones a priori del Entregable 1 (se heredan en E2 y E3)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

# --- Problema -------------------------------------------------------------
DOMAIN = "Energy"
TARGET = "OT"            # Weekly gasoline retail price (USD/gal), East Coast series
L = 26                   # ventana de entrada (semanas); PACF de dlog significativa hasta el rezago 23
H = 12                   # horizonte de pronostico (semanas), estrategia multi-salida directa
HORIZONS_REPORT = (1, 4, 12)
SEASONAL_PERIOD = 52

# --- Particion cronologica ------------------------------------------------
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15   # el resto (15 %) es prueba

# --- Metrica y margen -----------------------------------------------------
PRIMARY_METRIC = "MASE_1:12"   # promedio de MASE por horizonte h=1..12 (escala: ingenuo h-pasos en train)
DELTA_REL = 0.03               # margen de relevancia practica: 3 % de reduccion relativa de la MASE
ALPHA = 0.05
POWER = 0.80
INDEPENDENCE_UNIT = "origen de pronostico (ventana); las semillas se aparean por indice"

# --- Presupuesto ----------------------------------------------------------
SEARCH_BUDGET = 12             # configuraciones aleatorias por modelo (igual para los tres)
SEARCH_SEED = 0
SPACE_SEED = 2026
PILOT_SEEDS = [100, 101, 102, 103, 104]
FINAL_SEEDS_MIN = 10
MAX_EPOCHS = 200
PATIENCE = 20
