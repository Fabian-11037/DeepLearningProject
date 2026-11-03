from _nbtools import *
cells = [
("md", """# 5. Implementación de los tres modelos base"""),
("code", "%matplotlib inline\n" + SETUP + """
import inspect, torch
from tsbase import models as M
from tsbase.models import MODELS, SPACES, n_params
best = {k: v["best"] for k, v in json.load(open(ROOT / "results/search.json")).items()}
runs = pd.read_csv(ROOT / "results/final_runs.csv")"""),
("md", """## 5.1 Código de los modelos
Implementados desde cero con las capas primitivas de PyTorch (sin importar arquitecturas prediseñadas). Versiones de software en `requirements.txt` y en la Sección 5.4."""),
("code", """print(inspect.getsource(M.MLP)); print(inspect.getsource(M.LSTM1)); print(inspect.getsource(M.CNN1D))"""),
("md", """## 5.2 Configuraciones seleccionadas y número de parámetros
Las expresiones analíticas del Cap. 3 se verifican contra el conteo real de PyTorch:"""),
("code", """L, H = C.L, C.H
c = best["MLP"]; a_mlp = (L + 1) * c["hidden"] + (c["hidden"] + 1) * H if c["layers"] == 1 else None
d = best["LSTM"]["hidden"]; a_lstm = 4 * d * (d + 3) + (d + 1) * H
c = best["CNN1D"]; a_cnn = c["channels"] * (c["kernel"] + 1) + (c["channels"] * c["kernel"] * c["channels"] + c["channels"]) * (c["convs"] - 1) + (c["channels"] * L + 1) * H
rows = []
for name, an in [("MLP", a_mlp), ("LSTM", a_lstm), ("CNN1D", a_cnn)]:
    kw = {k: v for k, v in best[name].items() if k not in ("lr", "weight_decay", "batch", "clip")}
    rows.append({"modelo": name, "hiperparámetros": json.dumps(best[name]), "parámetros (PyTorch)": n_params(MODELS[name](**kw)), "parámetros (fórmula)": an})
pd.DataFrame(rows).style.hide(axis="index")"""),
("md", """## 5.3 Curvas de entrenamiento y validación (semilla 1000)"""),
("code", """fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for a, name in zip(ax, MODELS):
    h = np.array(json.load(open(ROOT / f"results/hist/{name}_1000.json"))); a.plot(h[:, 0], label="entrenamiento"); a.plot(h[:, 1], label="validación")
    a.axvline(runs.query("model==@name and seed==1000").best_epoch.iloc[0] - 1, ls=":", c="k"); a.set_title(name); a.set_xlabel("época")
ax[0].set_ylabel("MSE escalado"); ax[0].legend(); plt.tight_layout(); plt.show()"""),
("md", """La línea punteada marca la época elegida por early stopping (mínimo de validación). La brecha entrenamiento→validación en MASE es de 6.5 % (MLP) a 11 % (CNN 1D); se cuantifica en el Cap. 6.6. Las curvas son de una sola semilla y solo ilustran el procedimiento."""),
("md", """## 5.4 Entorno, tiempo y reproducibilidad"""),
("code", """import platform, sklearn, scipy, statsmodels
pd.Series({"Python": platform.python_version(), "PyTorch": torch.__version__.split("+")[0], "NumPy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
           "SciPy": scipy.__version__, "statsmodels": statsmodels.__version__, "Hardware": "CPU (2 núcleos, 1 hilo por entrenamiento), sin GPU", "Determinismo": "estricto (torch.use_deterministic_algorithms(True))"}).to_frame("versión")"""),
("code", """runs.groupby("model")[["epochs", "train_time_s", "infer_ms_per_window", "n_params"]].agg(["mean", "std"]).round(3)"""),
]
write("05_modelos_base.ipynb", cells)
