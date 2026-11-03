"""Carga, particion cronologica y ventanas. El conjunto de prueba solo se entrega con allow_test=True."""
import hashlib
import numpy as np
import pandas as pd
from . import config as C


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def energy_path():
    return C.DATA_DIR / "numerical" / "Energy" / "Energy.csv"


def load_energy() -> pd.DataFrame:
    df = pd.read_csv(energy_path(), parse_dates=["date"]).set_index("date")
    return df


class Split:
    """Indices de la particion (por tiempo del objetivo). Train < val < test sin solapamiento de etiquetas."""

    def __init__(self, n: int):
        self.n = n
        self.n_tr = int(C.TRAIN_FRAC * n)
        self.n_va = int(C.VAL_FRAC * n)
        self.n_te = n - self.n_tr - self.n_va
        self.bounds = {"train": (0, self.n_tr), "val": (self.n_tr, self.n_tr + self.n_va),
                       "test": (self.n_tr + self.n_va, n)}

    def origins(self, part: str, L=C.L, H=C.H) -> np.ndarray:
        a, b = self.bounds[part]
        lo = max(a - 1, L - 1)         # primer objetivo (t+1) cae dentro de la particion
        hi = b - 1 - H                 # ultimo objetivo (t+H) cae dentro de la particion
        return np.arange(lo, hi + 1)


class Windows:
    def __init__(self, y: np.ndarray, L=C.L, H=C.H):
        self.y, self.L, self.H = np.asarray(y, float), L, H
        self.logy = np.log(self.y)
        self.split = Split(len(y))
        Xtr, Ytr = self._xy(self.split.origins("train"))
        self.scale = float(Xtr.std())          # escala ajustada solo con train

    def _xy(self, org):
        lg, L, H = self.logy, self.L, self.H
        X = np.stack([lg[t - L + 1:t + 1] - lg[t] for t in org])
        Y = np.stack([lg[t + 1:t + H + 1] - lg[t] for t in org])
        return X, Y

    def get(self, part: str, allow_test: bool = False):
        if part == "test" and not allow_test:
            raise PermissionError("El conjunto de prueba se evalua una sola vez, en el script final.")
        org = self.split.origins(part)
        X, Y = self._xy(org)
        last = self.y[org]
        truth = np.stack([self.y[t + 1:t + self.H + 1] for t in org])
        return dict(org=org, X=X / self.scale, Y=Y / self.scale, last=last, truth=truth)

    def to_levels(self, out_scaled: np.ndarray, last: np.ndarray) -> np.ndarray:
        return last[:, None] * np.exp(out_scaled * self.scale)

    def mase_scale(self) -> np.ndarray:
        """Escala MASE por horizonte h: MAE del ingenuo h-pasos dentro de train (nivel)."""
        tr = self.y[: self.split.n_tr]
        return np.array([np.mean(np.abs(tr[h:] - tr[:-h])) for h in range(1, self.H + 1)])
