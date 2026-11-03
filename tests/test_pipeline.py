import sys
from pathlib import Path
import numpy as np, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsbase import config as C
from tsbase.data import Windows, Split, load_energy
from tsbase.train import fit, predict

W = Windows(load_energy()[C.TARGET].values)


def test_no_label_overlap_between_partitions():
    sp = W.split
    last_tr = sp.origins("train")[-1] + C.H
    first_va = sp.origins("val")[0] + 1
    last_va = sp.origins("val")[-1] + C.H
    first_te = sp.origins("test")[0] + 1
    assert last_tr < sp.n_tr <= first_va
    assert last_va < sp.n_tr + sp.n_va <= first_te


def test_test_requires_explicit_permission():
    with pytest.raises(PermissionError):
        W.get("test")
    assert len(W.get("test", allow_test=True)["X"]) > 0


def test_scale_uses_only_train():
    tr = W.get("train"); raw = tr["X"] * W.scale
    assert abs(raw.std() - W.scale) < 1e-9


def test_window_is_causal():
    d = W.get("val"); t = d["org"][0]
    assert np.allclose(d["X"][0] * W.scale, W.logy[t - C.L + 1:t + 1] - W.logy[t])
    assert d["X"].shape[1] == C.L and d["Y"].shape[1] == C.H
    assert np.allclose(d["truth"][0], W.y[t + 1:t + C.H + 1])


def test_same_seed_same_result():
    from tsbase.models import DEFAULTS
    tr, va = W.get("train"), W.get("val")
    a, _, _ = fit("MLP", DEFAULTS["MLP"], 7, tr, va, max_epochs=5)
    b, _, _ = fit("MLP", DEFAULTS["MLP"], 7, tr, va, max_epochs=5)
    assert np.array_equal(predict(a, va["X"]), predict(b, va["X"]))
