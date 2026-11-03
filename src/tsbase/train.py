import random, time
import numpy as np
import torch
import torch.nn as nn
from . import config as C
from .models import MODELS, n_params


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _t(a):
    return torch.as_tensor(a, dtype=torch.float32)


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(_t(X)).numpy()


def fit(name, cfg, seed, tr, va, max_epochs=C.MAX_EPOCHS, patience=C.PATIENCE):
    """Entrena con early stopping sobre MSE de validacion (escalado). Devuelve modelo, historial, meta."""
    seed_everything(seed)
    cfg = dict(cfg)
    kw = {k: v for k, v in cfg.items() if k not in ("lr", "weight_decay", "batch", "clip")}
    model = MODELS[name](**kw)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    clip = cfg.get("clip", 1.0)
    X, Y = _t(tr["X"]), _t(tr["Y"]); Xv, Yv = _t(va["X"]), _t(va["Y"])
    g = torch.Generator().manual_seed(seed)
    lossf = nn.MSELoss()
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        tl = 0.0
        for i in range(0, len(X), cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            loss = lossf(model(X[idx]), Y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tl += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xv), Yv).item()
        hist.append((tl / len(X), vl))
        if vl < best - 1e-8:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    meta = dict(epochs=len(hist), best_epoch=int(np.argmin([h[1] for h in hist])) + 1,
                train_time_s=time.time() - t0, n_params=n_params(model), best_val_mse=best)
    return model, hist, meta
