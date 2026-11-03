import numpy as np


def per_horizon(truth, pred, scale):
    """truth, pred: (N,H). Devuelve dict con vectores por horizonte."""
    e = truth - pred
    mae = np.abs(e).mean(0)
    rmse = np.sqrt((e ** 2).mean(0))
    smape = (200 * np.abs(e) / (np.abs(truth) + np.abs(pred))).mean(0)
    mase = mae / scale
    return dict(MAE=mae, RMSE=rmse, sMAPE=smape, MASE=mase)


def summarize(truth, pred, scale, horizons=(1, 4, 12)):
    m = per_horizon(truth, pred, scale)
    out = {"MASE_1:12": float(m["MASE"].mean()), "RMSE_1:12": float(m["RMSE"].mean()),
           "MAE_1:12": float(m["MAE"].mean()), "sMAPE_1:12": float(m["sMAPE"].mean())}
    for h in horizons:
        for k in ("MASE", "RMSE", "MAE"):
            out[f"{k}_h{h}"] = float(m[k][h - 1])
    return out
